'''Simplified version of DLA in PyTorch.

Note this implementation is not identical to the original paper version.
But it seems works fine.

See dla.py for the original paper version.

Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class DLA(nn.Module):
    def __init__(self, block=BasicBlock):
        super(DLA, self).__init__()
        self.base = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=True)

        self.layer1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer3 = Tree(block, 32, 256, level=1, stride=3)
        self.linear7 = nn.Linear(1024, 512)
        self.linear8 = nn.Linear(512, 256)
        self.linear9 = nn.Linear(256, 64)
        self.linear10 = nn.Linear(64, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        out = self.base(x)
        out = F.dropout(out, p=self.dropout_input, training=self.is_training)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!! out shape should be {out.shape}')
        out = self.linear7(out)
        out = self.linear8(out)
        out = self.linear9(out)
        out = self.linear10(out)
        return out


class PartConv(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(PartConv, self).__init__()
        #self.base = nn.Conv2d(3, 16, kernel_size=1, stride=1, bias=True)

        self.layer1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

        self.layer3 = Tree(block, 32, 256, level=1, stride=3)
        self.linear7 = nn.Linear(1024, 512)
        self.linear8 = nn.Linear(512, 256)
        self.linear9 = nn.Linear(256, 64)
        #self.linear10 = nn.Linear(64, 10)
        self.dropout_input = 0.0
        self.dropout_hidden = 0.0
        self.is_training = True

    def forward(self, x):
        #out = self.base(x)
        out = F.dropout(x, p=self.dropout_input, training=self.is_training)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!! out shape should be {out.shape}')
        out = self.linear7(out)
        out = self.linear8(out)
        out = self.linear9(out)
        #out = self.linear10(out)
        return out
