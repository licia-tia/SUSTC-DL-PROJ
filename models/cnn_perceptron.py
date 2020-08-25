"""
@author: Jiajun Yu
@contact: 11711102@mail.sustech.edu.cn
"""

import pdb
import torch
from torch import nn
from torchvision import models

hidden = '20'
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
    def __init__(self, input_dim, output_dim=2, meta_dim=8, hidden_layer=hidden):
        '''
            initialize the CNN plus MLP model
            @param: input_dim: the dimension of the input vector
            @param: output_dim: optional, the dimension of the output
            @param: meta_dim: optional, the dimension of the meta data
            @param: hidden_layer: optional, the structure of hidden layer, a string value
            @return: none
        '''
        super().__init__()
        layer_dims = [64, 'pool', 128, 'pool', 256, 256, 'pool', 512, 'pool', 1024, 'pool', 1280]  # the layer of the CNN
        self.view_num = 1280  # the view shape should be (-1, view_num)
        layers = []
        in_dim = input_dim
        for i in layer_dims:
            if i == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                layers += self._make_layer(in_dim, i)
                in_dim = i
        self.cnn = nn.Sequential(*layers)
        model = nn.Linear(self.view_num, 2)
        nn.init.normal_(model.weight, 0, 0.01)
        nn.init.zeros_(model.bias)
        self.cnn_linear = nn.Sequential(nn.Dropout(0.2), model)  # the linear part of CNN, using a Dropout model here
        perceptron_layers = []
        in_dim = meta_dim + 2
        for dim in hidden_layer.split():
            out_dim = int(dim)
            perceptron_layers.append(nn.Linear(in_dim, out_dim))
            perceptron_layers.append(nn.ReLU())
            in_dim = out_dim
        perceptron_layers.append(nn.Linear(in_dim, output_dim))
        perceptron_layers.append(nn.Softmax(1))
        self.perceptron = nn.Sequential(*perceptron_layers)


    def _make_layer(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1):
        '''
            method to create layer of CNN
            @param: in_channel: input dimension of the layer
            @param: out_channel: output dimension of the layer
            @params: kernel_size, stride, padding: hyperparameters of the Conv2d module in the layer
            @return: the layer with a Conv2d module, a batch normalization module, and an activation module(ReLU)
        '''
        layer = [nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
                 nn.BatchNorm2d(out_channel), nn.ReLU()]
        return layer
        
    def forward(self, img, meta):
        # first send the img to CNN
        out = self.cnn(img)
        out = out.view(-1, self.view_num)
        score = self.cnn_linear(out)
        # score sent out from CNN
        # combine the score with meta data(onehot)
        score = torch.cat((score, meta), 1)
        # print(score.size(), meta.size())
        score = score.float()
        # send the score with meta to MLP
        out = self.perceptron(score)
        return out
