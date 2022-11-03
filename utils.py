import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import os
import torchvision.models as models
import inspect
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_vgg = models.vgg16(pretrained=True).features.to(device).eval()
VGG_MEAN = [103.939, 116.779, 123.68]

# 파라미터 freeze
for param in model_vgg.parameters():
    param.requires_grad_(False)

# 모델의 중간 레이어의 출력값을 얻는 함수를 정의
def get_features(x, model, layers):
    features = {}
    x = x*255.0
   # x = torch.permute(x,[0,2,3,1])
    red,green,blue = torch.split(x,1,dim=1)
    bgr = torch.cat([blue-VGG_MEAN[2],green - VGG_MEAN[1],red - VGG_MEAN[0]],1)

    for name, layer in enumerate(model.children()): # 0, conv
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features


feature_layers = {'2': 'conv1_2',
                  '7': 'conv2_2',
                  '14': 'conv3_3',
                  '21': 'conv4_3',
                  '27': 'conv5_3'}

'''
content_tensor = torch.Tensor(3,128,128)
content_tensor = torch.unsqueeze(content_tensor,0).to(device)  # 3,128,128
content_features = get_features(content_tensor, model_vgg, feature_layers)
print(content_features)
'''


def content_loss(output, content):
    masked = (1-content) * torch.square(output - content)

    return torch.mean(masked)

import torchvision.transforms.functional as T


def style_loss(output, style):
    style = torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))(style)
    output = torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))(output)
    # style = (cv2.cvtColor(style, cv2.COLOR_GRAY2RGB)+1)/2
    # output = torchvision.transforms.functional.to_grayscale(num_output_channels=3)
    style = (style + 1)/2
    output = (output + 1)/2
    # output = (cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)+1)/2

    # style = T.invert(style)
    # output = T.invert(output)

    def gram_matrix(x):
        n, c, h, w = x.size()
        features = x.view(n, c, h*w)
        G = torch.matmul(features, torch.permute(features, (0, 2, 1)))
        return G.div(n*c*h*w)

    # output_tensor = torch.unsqueeze(output, 0).to(device)
    output_features = get_features(output, model_vgg, feature_layers)
    feature_o = [output_features.get('conv1_2'),output_features.get('conv2_2'),output_features.get('conv3_3'),output_features.get('conv4_3'),output_features.get('conv5_3')]
    gram_o = [gram_matrix(l) for l in feature_o]

    # style_tensor = torch.unsqueeze(style, 0).to(device)
    style_features = get_features(style, model_vgg, feature_layers)
    feature_s = [style_features.get('conv1_2'), style_features.get('conv2_2'), style_features.get('conv3_3'),
                 style_features.get('conv4_3'), style_features.get('conv5_3')]
    gram_s = [gram_matrix(l) for l in feature_s]

    loss_s = torch.zeros(4, dtype=torch.float32).to(device)

    for g, g_ in zip(gram_o, gram_s):
        loss_s += torch.mean(torch.mean(torch.subtract(g, g_) ** 2, 1), 1)
        # loss_s += torch.mean(torch.subtract(g, g_) ** 2, 1)
    return torch.mean(loss_s)



