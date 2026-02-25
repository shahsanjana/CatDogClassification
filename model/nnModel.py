# %%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import sys 

# %%
# 50 X 50 pixels
img_size = 50

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)     # consider padding=2? what does this do
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # infer flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 50, 50)
            out = self._forward_conv(dummy)
            n = out.view(1, -1).size(1)
            
        self.fc1 = nn.Linear(n, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, img_size=50):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # infer flattened conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            out = self._forward_conv(dummy)
            n = out.flatten(1).shape[1]

        self.fc1 = nn.Linear(n, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def _forward_conv(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x

#net = Net()

# %%
# test_img = torch.randn(img_size, img_size).view(-1,1, img_size, img_size)  #batch size, channels, height, width
# output = net(test_img)
# print(output)
# cv 1 torch.Size([1, 32, 23, 23])
# cv 2 torch.Size([1, 64, 9, 9])
# cv 3 torch.Size([1, 128, 2, 2])



