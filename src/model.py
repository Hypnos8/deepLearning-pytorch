import torch
import torch.nn.functional

from src.ResBlock import ResBlock


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resNet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),  # input_tensor is output of ResBlock from before
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            torch.nn.AvgPool2d(kernel_size=(10, 10)),  # not sure how to implement global average pooling
            #torch.nn.AdaptiveAvgPool2d(output_size=512),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=512, out_features=2),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.resNet(x)


