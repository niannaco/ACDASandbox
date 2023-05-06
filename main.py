from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os, shutil
import argparse

from lenet import *
from lenet_acda import *
from adresnets import *
from progress_bar import *

BATCH_SIZE = 128
OPTIMIZER = 'adam'
gpu_index = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 
start_epoch = 0 

# Data
print('Loading data...')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
#trainset = torchvision.datasets.ImageFolder(root='/kaggle/input/cifake-real-and-ai-generated-synthetic-images/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
#testset = torchvision.datasets.ImageFolder(root='/kaggle/input/cifake-real-and-ai-generated-synthetic-images/test', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('Building model...')

# CIFAR-10/100
#net = LeNet()
#net = LeNet_DCF()
#net = AdResNetS_Conv(3,100)
net = AdResNetS(3,100)

#net = torchvision.models.resnet18(weights=None)

# CIFAKE
# net = AdResNetS(3,2)

print(net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

for epoch in range(start_epoch, start_epoch+30):
    if epoch in [22, 27]:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
        print('In epoch %d, decaying LR to %f' %(epoch, optimizer.param_groups[0]['lr']))
    train(epoch)
    test(epoch)