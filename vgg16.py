import torch
import torch.nn as nn
from torchsummary import summary

#vgg16_model
class Vgg16(nn.Module):
    def __init__(self,num_classes=3):
        super(Vgg16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(128,256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=1, ceil_mode=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Test_model(nn.Module):
    def __init__(self,num_classes=5):
        super(Test_model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.ReLU(True),
            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Vgg16(3).to(device)
    # x = torch.randn(1, 3, 224, 224).to(device)
    # res = model(x)
    summary(model,(3,224,224),device='cuda')
    # print(res)