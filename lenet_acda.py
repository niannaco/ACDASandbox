import torch.nn as nn
import sys
import torch.nn.functional as F
from acda import *
from torchsummary import summary

class LeNet_ACDA(nn.Module):
    def __init__(self):
        super(LeNet_ACDA, self).__init__()
        self.conv1 = Conv_ACDA(3, 6, kernel_size = 3, padding = 1, stride = 1, bias = True)
        self.conv2 = Conv_ACDA(6, 16, kernel_size = 3, padding = 1, stride = 1, bias = True)
        self.fc1   = nn.Linear(16*8*8, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 100)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2) # 6x16x16
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2) # 16x8x8
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
def test():
    net = LeNet_ACDA()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
    print(summary(net,(3,32,32),device = "cpu"))

if __name__ == '__main__':
    test()