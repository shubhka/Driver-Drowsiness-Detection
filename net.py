# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 14:14:08 2023

@author: Shubhi Kant
"""

#%% Importing Libraries
import torch
import torch.nn as nn
from torchvision import models

#%% Architecture

class Net (nn.Module):
    def __init__ (self):
        super (Net, self).__init__()
        self.conv1 = nn.Conv2d(32, 3, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.5)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64*64*128, 512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Dense(512, 64)
        self.fc3 = nn.Dense(64, 1)
        
    def forward (self, x):
        x = torch.relu(self.conv1(x))
        x = self.drop1(self.pool1(self.bn1(x)))
        x = torch.relu(self.conv2(x))
        x = self.drop2(self.pool2(self.bn2(x)))
        x = torch.relu(self.conv3(x))
        x = self.drop3(self.pool3(self.bn3(x)))
        x = self.flatten(x)
        x = self.drop4(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
        