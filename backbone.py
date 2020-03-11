import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from abc import abstractmethod
import math


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class ConvBlock(nn.Module):
    """Convolution, optionally followed by batch normalization and a ReLU non-linearity"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act_fun=True, bn=True, bias=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_fun:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """Transposed convolution, optionally followed by batch normalization and a ReLU non-linearity"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn=True, act_fun=True, bias=True):
        super(ConvTransposeBlock, self).__init__()
        layers = [nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_fun:
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    """Residual block as used in ResNet architectures
    This block contains two convolutions each followed by batch normalization and a ReLU non-linearity"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """Backbone architecture shared by all models
    Currently supports Conv-4 and ResNet-18 architectures"""
    def __init__(self, backbone, in_channels, image_size, out_channels, add_fc=False, last_relu=True):
        super(Encoder, self).__init__()
        if backbone == 'Conv4':
            num_channels = 64
            batch_norm = True
            self.layers = nn.ModuleList()
            self.layers.append(ConvBlock(in_channels, num_channels, bn=batch_norm))
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(ConvBlock(num_channels, num_channels, bn=batch_norm))
            self.layers.append(nn.MaxPool2d(2))
            self.layers.append(ConvBlock(num_channels, num_channels, bn=batch_norm))
            self.layers.append(nn.MaxPool2d(2))
            if add_fc:  # add fully connected layer on top of 4th convolution
                self.layers.append(ConvBlock(num_channels, num_channels, bn=batch_norm))
                self.layers.append(nn.MaxPool2d(2))
                feat_size = self.get_feature_size(backbone, image_size, num_channels)
                if last_relu:  # end with ReLU activation function
                    last_layer = nn.Sequential(View([-1, feat_size]), nn.Linear(feat_size, out_channels), nn.ReLU())
                else:
                    last_layer = nn.Sequential(View([-1, feat_size]), nn.Linear(feat_size, out_channels))
                self.layers.append(last_layer)
            else:
                self.layers.append(ConvBlock(num_channels, out_channels, act_fun=last_relu, bn=batch_norm))
                self.layers.append(nn.MaxPool2d(2))
        elif backbone == 'ResNet18':
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]
            strides = [1, 2, 2, 2]
            self.num_channels = 64
            self.layers = nn.ModuleList()
            self.layers.append(ConvBlock(in_channels, self.num_channels, kernel_size=7, stride=2, padding=3, bias=False))
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.layers.append(self._make_layer(block, 64, num_blocks[0], strides[0]))
            self.layers.append(self._make_layer(block, 128, num_blocks[1], strides[1]))
            self.layers.append(self._make_layer(block, 256, num_blocks[2], strides[2]))
            self.layers.append(self._make_layer(block, 512, num_blocks[3], strides[3]))
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            if add_fc:
                if last_relu:
                    self.layers.append(nn.Sequential(View([-1, 512]), nn.Linear(512, out_channels), nn.ReLU()))
                else:
                    self.layers.append(nn.Sequential(View([-1, 512]), nn.Linear(512, out_channels)))

            for m in self.modules():
                # Initialization using fan-in
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            raise NotImplementedError
        self.num_layers = len(self.layers)

    def _make_layer(self, block, num_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.num_channels, num_channels, stride))
            self.num_channels = num_channels
        return nn.Sequential(*layers)

    @staticmethod
    def get_feature_size(backbone, image_size, z_dim):
        if backbone == 'Conv4':
            size = image_size
            for _ in range(4):
                size //= 2
            size = z_dim * size * size
        elif backbone == 'ResNet18':
            size = z_dim
        else:
            raise NotImplementedError
        return size

    def forward(self, x, last_layer):
        if last_layer is None:
            last_layer = self.num_layers
        out = x
        for i in range(last_layer):
            out = self.layers[i](out)
        return out


class Decoder(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, image_size):
        super(Decoder, self).__init__()
        if backbone == 'Conv4':
            num_channels = 64
            batch_norm = True
            self.image_size = image_size
            size = self.get_feature_size(backbone, image_size)
            first_layer = nn.Sequential(
                nn.Linear(in_channels, num_channels * size * size),
                View((-1, num_channels, size, size)),
                nn.BatchNorm2d(num_channels),
                nn.ReLU()
            )
            self.layers = nn.ModuleList()
            self.layers.append(first_layer)
            self.layers.append(ConvTransposeBlock(num_channels, num_channels, bn=batch_norm))
            self.layers.append(ConvTransposeBlock(num_channels, num_channels, bn=batch_norm))
            self.layers.append(ConvTransposeBlock(num_channels, num_channels, bn=batch_norm))
            self.layers.append(ConvTransposeBlock(num_channels, out_channels, bn=False, act_fun=False))
        elif backbone == 'ResNet18':
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]
            strides = [1, 1, 1, 1]
            self.num_channels = 512
            self.image_size = image_size
            size = self.get_feature_size(backbone, image_size)
            first_layer = nn.Sequential(
                nn.Linear(in_channels, self.num_channels * size * size),
                View((-1, self.num_channels, size, size)),
                nn.BatchNorm2d(self.num_channels),
                nn.ReLU()
            )
            self.layers = nn.ModuleList()
            self.layers.append(first_layer)
            self.layers.append(self._make_layer(block, 512, num_blocks[0], strides[0]))
            self.layers.append(Interpolate(scale_factor=2))
            self.layers.append(self._make_layer(block, 256, num_blocks[1], strides[1]))
            self.layers.append(Interpolate(scale_factor=2))
            self.layers.append(self._make_layer(block, 128, num_blocks[2], strides[2]))
            self.layers.append(Interpolate(scale_factor=2))
            self.layers.append(self._make_layer(block, 64, num_blocks[3], strides[3]))
            self.layers.append(Interpolate(scale_factor=2))
            self.layers.append(nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            raise NotImplementedError

    def _make_layer(self, block, num_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.num_channels, num_channels, stride))
            self.num_channels = num_channels
        return nn.Sequential(*layers)

    @staticmethod
    def get_feature_size(backbone, image_size):
        if backbone == 'Conv4':
            size = image_size
            offset = 0
            for _ in range(4):
                if (size // 2) < (size / 2.0):
                    offset += 1
                size //= 2
            size += max(offset, 1)
        elif backbone == 'ResNet18':
            size = round(image_size / 16.0)
        else:
            raise NotImplementedError
        return size

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = torch.sigmoid(out)
        out = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)
        return out


class BaseClass(nn.Module):
    """Base class for all models"""
    def __init__(self):
        super(BaseClass, self).__init__()

    @abstractmethod
    def encode(self, x, last_layer=None):
        pass

    @staticmethod
    @abstractmethod
    def update(**kwargs):
        pass

    @staticmethod
    def train_classifier(z_all, n_way, n_support, n_query):
        """
        Train linear classfier for downstream tasks
        :param z_all: embeddings for all classes, support and query examples (n_way x (n_support + n_query) x z_dim)
        :param n_way: number of classes
        :param n_support: number of support examples per class
        :param n_query: number of query examples per class
        :return: scores for query example
        """
        z_dim = z_all[0, 0].size()

        z_support = z_all[:, :n_support]
        z_query = z_all[:, n_support:]

        z_support = z_support.contiguous().view(n_way * n_support, *z_dim)
        z_query = z_query.contiguous().view(n_way * n_query, *z_dim)

        y_support = torch.from_numpy(np.repeat(range(n_way), n_support))
        y_support = Variable(y_support.to(z_all.device))

        classifier = Classifier(z_dim, n_way, num_layers=1).to(z_all.device)

        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0005)

        batch_size = 4
        support_size = n_way * n_support
        for epoch in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).to(z_all.device)
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]

                scores = classifier(z_batch)
                loss = F.cross_entropy(scores, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        scores = classifier(z_query)
        return scores


class Classifier(nn.Module):
    """Simple NN-classifier, supports up to 2 layers
    For a 2-layer network, the first layer is a 3x3 convolution if the input tensor is of order 3."""
    def __init__(self, input_dim, output_dim, num_layers, num_channels=64):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            if len(input_dim) == 1:
                self.layers.append(nn.Linear(input_dim[0], output_dim))
            elif len(input_dim) == 3:
                in_dim = input_dim[0] * input_dim[1] * input_dim[2]
                self.layers.append(View((-1, in_dim)))
                self.layers.append(nn.Linear(in_dim, output_dim))
            else:
                raise NotImplementedError
        elif num_layers == 2:
            if len(input_dim) == 1:
                self.layers.append(nn.Linear(input_dim[0], num_channels))
                self.layers.append(nn.Linear(num_channels, output_dim))
            elif len(input_dim) == 3:
                pooling_dim = [2, 2]
                width = num_channels * pooling_dim[0] * pooling_dim[1]
                self.layers.append(nn.Conv2d(input_dim[0], num_channels, kernel_size=3, stride=1, padding=0))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.AdaptiveAvgPool2d(pooling_dim))
                self.layers.append(View((-1, width)))
                self.layers.append(nn.Linear(width, output_dim))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == '__main__':
    pass
