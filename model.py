import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # Output corresponds to num_classes
        )

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)  # Sử dụng ResNet50

        # Điều chỉnh lại lớp phân loại (fc) với cấu trúc mới
        self.model.fc = nn.Sequential(
            nn.Flatten(),                     # Flatten đầu vào
            nn.Linear(2048, 512),             # Chuyển từ 2048 (đặc trưng đầu ra của ResNet50) xuống 512
            nn.ReLU(inplace=True),            # Hàm kích hoạt ReLU
            nn.Linear(512, 128),              # Chuyển từ 512 xuống 128
            nn.ReLU(inplace=True),            # Hàm kích hoạt ReLU
            nn.Linear(128, num_classes)       # Output với số lớp bằng num_classes
        )

    def forward(self, x):
        return self.model(x)  # Truyền dữ liệu qua mô hình ResNet50
    
if __name__ == '__main__':
    model = ResNet50()
    print(model)

class VGG11Model(nn.Module):
    # Implement VGG11 model for transfer learning
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg11(pretrained=True)
        
        # Freeze the convolutional base
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        
        # Replace avgpool with AdaptiveAvgPool2d
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace the classifier with a new one
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # Output corresponds to num_classes
        )

    def forward(self, x):
        return self.model(x)