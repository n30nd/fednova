import torch
from model import ResNet18, ResNet50, VGG11Model
import torch.optim as optim 
import copy
import random 
import numpy as np
import time 
import matplotlib.pyplot as plt

def federated_train(trainloaders, valloaders, testloader, config):
    model = ResNet18(num_classes=2)
    # model = ResNet50(num_classes=2)
    # model = VGG11Model(num_classes=2)
    global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
        
    num_rounds = config.num_rounds  # Số vòng huấn luyện
    accs = []

    accs.append(evaluate(global_model, testloader)) #huan luyen 1 lan trc o server
    for round_num in range(num_rounds):
        print(f"Round {round_num + 1}/{num_rounds}")
        start = time.time()
        a_list = []
        d_list = []
        n_list = []
        # Chọn các client tham gia vào mỗi round
        selected_clients = select_clients(trainloaders, config.clients_per_round)
        
        # Huấn luyện trên các client đã chọn
        for client in selected_clients:
            a_i, d_i = local_train(client, global_model, config, trainloaders)
            a_list.append(a_i)
            d_list.append(d_i)
            n_list.append(len(trainloaders[client].dataset))
            
        total_n = sum(n_list)


        d_total_round = copy.deepcopy(global_model.state_dict())
        for key in d_total_round:
            d_total_round[key] = 0.0
        
        for i in range(len(selected_clients)):
            d_para = d_list[i]
            for key in d_para:
                d_total_round[key] += d_para[key] * n_list[i] / total_n
        
        #Update global model
        coeff = 0.0
        for i in range(len(selected_clients)):
            coeff += a_list[i] * n_list[i] / total_n
        
        updated_model = global_model.state_dict()
        for key in updated_model:
            if updated_model[key].type() == 'torch.LongTensor':
                updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
            elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
            else:
                #print(updated_model[key].type())
                #print((coeff*d_total_round[key].type()))
                updated_model[key] -= coeff * d_total_round[key]
        global_model.load_state_dict(updated_model)    

        # # Cập nhật mô hình toàn cục sử dụng FedNova
        # global_model = fednova_update(global_model, len_data_local_select, local_deltas, taus)
        
        # Đánh giá mô hình trên tập kiểm tra
        acc = evaluate(global_model, testloader)
        accs.append(acc)
        # Điều chỉnh learning rate theo độ chính xác
        if acc > 70.0:
            config.learning_rate = 1e-5
            print(f"Accuracy > 80%, decreasing learning rate to {config.learning_rate}")
        elif acc > 65.0:
            config.learning_rate = 1e-4
            print(f"Accuracy > 70%, decreasing learning rate to {config.learning_rate}")
        end = time.time()
        print(f'Time for round {round_num + 1}: ', end-start)
    print('accuracies: ', accs)
    plt.plot(range(0, num_rounds + 1), accs, marker='o', label='Accuracy')
    plt.xlabel('Round')
    plt.xticks(range(0, num_rounds + 1, 10))
    plt.ylabel('Accuracy')
    plt.title('FedNova on ResNet18 over Rounds')
    plt.grid(True)
    plt.legend()
    plt.savefig('running_outputs/accuracy_summary.png')
    plt.close()



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
    net = copy.deepcopy(global_model).to(device) # Sử dụng bản sao mô hình toàn cục
    net.train()
    
    # Sử dụng SGD với learning rate = 1e-3 và momentum = 0.9
    optimizer = optim.SGD(
        # net.parameters(),
        filter(lambda p: p.requires_grad, net.parameters()), 
        lr=config.learning_rate, 
        momentum=config.momentum)
    tau = 0
    print(f"Training on client {client}, device: {device}, learning_rate={config.learning_rate}")
    # Huấn luyện mô hình trên client
    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(trainloader[client]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            tau += 1  # Tăng số lần cập nhật

    a_i = (tau-config.momentum * (1-pow(config.momentum, tau)) / (1 - config.momentum)) / (1 - config.momentum)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key].to('cpu'), a_i)
    

    return a_i, norm_grad

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
    return accuracy
