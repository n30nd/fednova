import torch
from model import ResNet18
import torch.optim as optim 
import copy
import random 
import numpy as np

def federated_train(trainloaders, valloaders, testloader, config):
    model = ResNet18(num_classes=2)
    global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
    
    # Optimizer cho mô hình toàn cục
    optimizer = optim.SGD(global_model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    
    num_rounds = config.num_rounds  # Số vòng huấn luyện
    
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        
        # Chọn các client tham gia vào mỗi round
        selected_clients = select_clients(trainloaders, config.clients_per_round)
        
        # Huấn luyện trên các client đã chọn
        local_models = []
        local_deltas = []
        taus = []
        
        for client in selected_clients:
            local_model, delta_w, tau = local_train(client, global_model, config, trainloaders)
            local_models.append(local_model)
            local_deltas.append(delta_w)
            taus.append(tau)
        
        len_data_local_select = [len(trainloaders[i].dataset) for i in selected_clients]
        # Cập nhật mô hình toàn cục sử dụng FedNova
        global_model = fednova_update(global_model, len_data_local_select, local_models, local_deltas, taus, config.fraction_fit)
        
        # Đánh giá mô hình trên tập kiểm tra
        evaluate(global_model, testloader)


def select_clients(trainloaders, clients_per_round):
    """Chọn ngẫu nhiên một số client tham gia huấn luyện trong mỗi round."""
    # Số lượng client có sẵn
    total_clients = len(trainloaders)
    # Chọn ngẫu nhiên một số client
    selected_clients = random.sample(range(total_clients), clients_per_round)
    return selected_clients

def local_train(client, global_model, config, trainloader):
    """Huấn luyện mô hình trên một client cụ thể."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(global_model).to(device) # Sử dụng bản sao mô hình toàn cục
    model.train()
    
    # Sử dụng SGD với learning rate = 1e-3 và momentum = 0.9
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    tau = 0
    print(f"Training on client {client}, device: {device}")
    # Huấn luyện mô hình trên client
    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(trainloader[client]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            tau += 1  # Tăng số lần cập nhật
    # Tính toán sự thay đổi trọng số và số lượng cập nhật
    delta_w = {name: global_model.state_dict()[name] - param.cpu()  for name, param in model.state_dict().items()}
    # tau = len(trainloader[client]) * config.num_epochs  # Tính số lần cập nhật

    return model, delta_w, tau

def fednova_update(global_model, len_data_local_select, local_models, local_deltas, taus, fraction_fit):
    """Cập nhật mô hình toàn cục sử dụng FedNova."""

    
    # Số lượng client tham gia
    num_clients = len(local_models)
    len_data_local_select = np.array(len_data_local_select)
    taus = np.array(taus)
    local_deltas = np.array(local_deltas)
    c1 = np.sum(len_data_local_select * taus)/num_clients

    
    # Cập nhật mô hình toàn cục theo FedNova
    with torch.no_grad():
        for name, param in global_model.state_dict().items():
            # Bỏ qua các tham số có kiểu int64 (num_batches_tracked)
            if param.dtype == torch.int64:
                continue
            
            c2 = np.sum([
                (len_data_local_select[i] * local_deltas[i][name] / taus[i]) 
                for i in range(len(local_deltas))
            ]) / num_clients
            c3 = c1 * c2

            global_model.state_dict()[name] -= c3
            # # Tính tổng trọng số thay đổi cho mỗi layer
            # weighted_delta = sum(taus[i] * local_deltas[i][name] for i in range(num_clients)) / total_tau
            
            # # Đảm bảo weighted_delta có kiểu float
            # weighted_delta = weighted_delta.float()  # Ép kiểu về float nếu cần
            
            # # Ép kiểu toàn bộ các tham số trong global_model về float
            # global_model.state_dict()[name] = global_model.state_dict()[name].float()
            
            # # Cập nhật trọng số mô hình toàn cục
            # global_model.state_dict()[name] -= fraction_fit * weighted_delta
    
    return global_model




def evaluate(model, testloader):
    """Đánh giá mô hình trên tập kiểm tra."""
    model.eval()  # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")