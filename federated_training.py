import torch
from model import ResNet18, ResNet50, VGG11Model
import torch.optim as optim 
import copy
import random 
import numpy as np
import time 
import matplotlib.pyplot as plt

from data_utils import get_val_dataloader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def federated_train(trainloaders, valloaders, testloader, config):
#     # model = ResNet18(num_classes=2)
#     # model = ResNet50(num_classes=2)
#     model = VGG11Model(num_classes=2)
#     global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
        
#     num_rounds = config.num_rounds  # Số vòng huấn luyện
#     accs_test = []
#     accs_val = []
#     accs_test.append(evaluate(global_model, testloader)) #huan luyen 1 lan trc o server
#     for round_num in range(num_rounds):
#         print(f"Round {round_num + 1}/{num_rounds}")
#         start = time.time()
#         a_list = []
#         d_list = []
#         n_list = []
#         len_val_list = []
#         acc_val_list = []
#         # Chọn các client tham gia vào mỗi round
#         selected_clients = select_clients(trainloaders, config.clients_per_round)
        
#         # Huấn luyện trên các client đã chọn
#         for client in selected_clients:
#             a_i, d_i, acc_val_i, len_val_i = local_train(client, global_model, config, trainloaders[client], valloaders[client])
#             a_list.append(a_i)
#             d_list.append(d_i)
#             n_list.append(len(trainloaders[client].dataset))
#             len_val_list.append(len_val_i)
#             acc_val_list.append(acc_val_i)
#         total_n = sum(n_list)
#         total_n_val = sum(len_val_list)

#         for i in range(len(len_val_list)):
#             len_val_list[i] = len_val_list[i] / total_n_val
#         acc_val_aggrated = sum([acc_val * len_val for acc_val, len_val in zip(acc_val_list, len_val_list)])
#         accs_val.append(acc_val_aggrated)

#         d_total_round = copy.deepcopy(global_model.state_dict())
#         for key in d_total_round:
#             d_total_round[key] = 0.0
        
#         for i in range(len(selected_clients)):
#             d_para = d_list[i]
#             for key in d_para:
#                 d_total_round[key] += d_para[key] * n_list[i] / total_n
        
#         #Update global model
#         coeff = 0.0
#         for i in range(len(selected_clients)):
#             coeff += a_list[i] * n_list[i] / total_n
        
#         updated_model = global_model.state_dict()
#         for key in updated_model:
#             if updated_model[key].type() == 'torch.LongTensor':
#                 updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
#             elif updated_model[key].type() == 'torch.cuda.LongTensor':
#                     updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
#             else:
#                 #print(updated_model[key].type())
#                 #print((coeff*d_total_round[key].type()))
#                 updated_model[key] -= coeff * d_total_round[key]
#         global_model.load_state_dict(updated_model)    

#         # # Cập nhật mô hình toàn cục sử dụng FedNova
#         # global_model = fednova_update(global_model, len_data_local_select, local_deltas, taus)
        
#         # Đánh giá mô hình trên tập kiểm tra
#         acc = evaluate(global_model, testloader)
#         accs_test.append(acc)
#         print(f"Accuracy on test set: {acc}")
#         print(f"Accuracy on validation set: {acc_val_aggrated}")
#         # Điều chỉnh learning rate theo độ chính xác
#         # if acc > 70.0:
#         #     config.learning_rate = 1e-5
#         #     print(f"Accuracy > 80%, decreasing learning rate to {config.learning_rate}")
#         # elif acc > 65.0:
#         #     config.learning_rate = 1e-4
#         #     print(f"Accuracy > 70%, decreasing learning rate to {config.learning_rate}")
#         if round_num >= 19:
#             config.learning_rate = 1e-4
#         if round_num >= 39:
#             config.learning_rate = 1e-5
#         end = time.time()
#         print(f'Time for round {round_num + 1}: ', end-start)
#     print('accuracies test: ', accs_test)
#     print('accuracies val: ', accs_val)
#     plt.plot(range(0, num_rounds + 1), accs_test, marker='o', label='Accuracy_test')
#     plt.plot(range(1, num_rounds + 1), accs_val, marker='x', label='Accuracy_val')
#     plt.xlabel('Round')
#     plt.xticks(range(0, num_rounds + 1, 10))
#     plt.ylabel('Accuracy')
#     plt.title('FedNova on ResNet18 over Rounds')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig('running_outputs/accuracy_summary.png')
#     plt.close()

def federated_train(trainloaders, valloaders, testloader, config):
    # model = ResNet18(num_classes=2)
    # model = ResNet50(num_classes=2)

    #Cai dat seed
    random.seed(config.dataset_seed)
    np.random.seed(config.dataset_seed)
    torch.manual_seed(config.dataset_seed)
    torch.cuda.manual_seed(config.dataset_seed)
    torch.cuda.manual_seed_all(config.dataset_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    model = VGG11Model(num_classes=2)
    global_model = copy.deepcopy(model)  # Bản sao mô hình toàn cục
    valloader_goc = get_val_dataloader()
    num_rounds = config.num_rounds  # Số vòng huấn luyện
    accs_test = []
    accs_val = []
    accs_test.append(evaluate(global_model, testloader)) #huan luyen 1 lan trc o server
    accs_val.append(evaluate(global_model, valloader_goc))
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
            a_i, d_i, acc_val_i, len_val_i = local_train(client, global_model, config, trainloaders[client], valloaders[client])
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
                updated_model[key] -= coeff * d_total_round[key]
        global_model.load_state_dict(updated_model)    

        # # Cập nhật mô hình toàn cục sử dụng FedNova
        # global_model = fednova_update(global_model, len_data_local_select, local_deltas, taus)
        
        # Đánh giá mô hình trên tập kiểm tra
        acc_test = evaluate(global_model, testloader)
        accs_test.append(acc_test)
        print(f"Accuracy on test set: {acc_test}")

        acc_val = evaluate(global_model, valloader_goc)
        accs_val.append(acc_val)
        print(f"Accuracy on validation set: {acc_val}")


        if round_num >= 3:
            if acc_val > 80.0:
                config.learning_rate = 1e-8
                print(f"Accuracy > 80%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 70.0:
                config.learning_rate = 1e-7
                print(f"Accuracy > 70%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 60.0:
                config.learning_rate = 1e-6
                print(f"Accuracy > 60%, decreasing learning rate to {config.learning_rate}")
            elif acc_val > 50.0:
                config.learning_rate = 1e-5
                print(f"Accuracy > 50%, decreasing learning rate to {config.learning_rate}")
            else :
                config.learning_rate = 1e-4
                print(f"Accuracy <= 50%, increasing learning rate to {config.learning_rate}")
        end = time.time()
        print(f'Time for round {round_num + 1}: ', end-start)
    print('accuracies test: ', accs_test)
    print('accuracies val: ', accs_val)
    plt.plot(range(0, num_rounds + 1), accs_test, marker='o', label='Accuracy_test')
    plt.plot(range(0, num_rounds + 1), accs_val, marker='x', label='Accuracy_val')
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

def local_train(client, global_model, config, trainloader, valloader):
    """Huấn luyện mô hình trên một client cụ thể."""
    net = copy.deepcopy(global_model).to(DEVICE) # Sử dụng bản sao mô hình toàn cục
    net.train()
    
    # Sử dụng SGD với learning rate = 1e-3 và momentum = 0.9
    optimizer = optim.SGD(
        # net.parameters(),
        filter(lambda p: p.requires_grad, net.parameters()), 
        lr=config.learning_rate, 
        momentum=config.momentum)
    tau = 0
    print(f"Training on client {client}, device: {DEVICE}, learning_rate={config.learning_rate}")
    # Huấn luyện mô hình trên client
    for epoch in range(config.num_epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = net(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            tau += 1  # Tăng số lần cập nhật

    acc_val_i = evaluate(net, valloader)

    a_i = (tau-config.momentum * (1-pow(config.momentum, tau)) / (1 - config.momentum)) / (1 - config.momentum)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key].to('cpu'), a_i)
    

    return a_i, norm_grad, acc_val_i, len(valloader.dataset)

def evaluate(model, testloader):
    """Đánh giá mô hình trên tập kiểm tra."""
    # print('evaluate on', device)
    model.to(DEVICE)
    model.eval()  # Chuyển sang chế độ đánh giá
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    model.to('cpu')
    return accuracy
