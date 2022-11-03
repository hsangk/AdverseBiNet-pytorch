import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchsummary




class Encoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder, self).__init__()
        model = [
            nn.Conv2d(in_size, out_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_size,0.9),
            nn.LeakyReLU(0.2,inplace=True)
        ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)

class Encoder2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Encoder2, self).__init__()
        model = [
            nn.Conv2d(in_size, out_size, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_size,0.9),
        ]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Decoder, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(out_size, 0.9),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)

class Decoder2(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(Decoder2, self).__init__()
        model = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(out_size, 0.9)
        ]
        if dropout:
            model.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)

# TANet
class Generator1(nn.Module):
    def __init__(self, content_input_shape, style_input_shape):
    #def __init__(self):
        super(Generator1, self).__init__()
        _,content_channel, _, _ = content_input_shape
        _,style_channel, _, _ = style_input_shape
        self.content_enc1 = Encoder(content_channel, 32)
        self.content_enc2 = Encoder(32, 64)
        self.content_enc3 = Encoder(64, 128)
        self.content_enc4 = Encoder(128, 256)
        self.content_enc5 = Encoder(256, 256)
        self.content_enc6 = Encoder(256, 256)
        self.content_enc7 = Encoder(256, 256)
        self.content_enc8 = Encoder2(256, 256)

        self.style_enc1 = Encoder(style_channel, 32)
        self.style_enc2 = Encoder(32, 64)
        self.style_enc3 = Encoder(64, 128)
        self.style_enc4 = Encoder(128, 256)
        self.style_enc5 = Encoder(256, 256)
        self.style_enc6 = Encoder(256, 256)
        self.style_enc7 = Encoder(256, 256)
        self.style_enc8 = Encoder2(256, 256)

        self.dec1 = Decoder(512, 256, dropout=0.5)
        self.dec2 = Decoder(512, 256, dropout=0.5)
        self.dec3 = Decoder(512, 256, dropout=0.5)
        self.dec4 = Decoder(512, 256, dropout=0.5)
        self.dec5 = Decoder(512, 128)
        self.dec6 = Decoder(256, 64)
        self.dec7 = Decoder(128, 32)
        self.dec8 = Decoder2(64, 1)

        self.initialize_weights()
        self.Tanh = nn.Tanh()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,0.0,0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0.0,0.02)
                nn.init.constant_(m.bias,0)


    def forward(self, x,y):
        c1 = self.content_enc1(x)
        c2 = self.content_enc2(c1)
        c3 = self.content_enc3(c2)
        c4 = self.content_enc4(c3)
        c5 = self.content_enc5(c4)
        c6 = self.content_enc6(c5)
        c7 = self.content_enc7(c6)
        c8 = self.content_enc8(c7)

        s1 = self.style_enc1(y)
        s2 = self.style_enc2(s1)
        s3 = self.style_enc3(s2)
        s4 = self.style_enc4(s3)
        s5 = self.style_enc5(s4)
        s6 = self.style_enc6(s5)
        s7 = self.style_enc7(s6)
        s8 = self.style_enc8(s7)

        c8_s8_cat = torch.cat((c8,s8),1)

        d1 = self.dec1(c8_s8_cat)
        d1 = torch.cat((d1, c7), 1)
        d2 = self.dec2(d1)
        d2 = torch.cat((d2, c6), 1)
        d3 = self.dec3(d2)
        d3 = torch.cat((d3, c5), 1)
        d4 = self.dec4(d3)
        d4 = torch.cat((d4, c4), 1)
        d5 = self.dec5(d4)
        d5 = torch.cat((d5, c3), 1)
        d6 = self.dec6(d5)
        d6 = torch.cat((d6, c2), 1)
        d7 = self.dec7(d6)
        d7 = torch.cat((d7, c1), 1)
        d8 = self.dec8(d7)

        return self.Tanh(d8)


class Discriminator1(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator1, self).__init__()
        _, input_channels,_,_ = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2,2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(64, momentum=0.9)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2,2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128, momentum=0.9)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1,2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256, momentum=0.9)
        )

        self.linear = nn.Linear(256*32*32,1, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,0.0,0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0.0,0.02)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        bs = x.size(0)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.reshape(bs, -1)
        x5 = self.linear(x4)
    #    sig = nn.Sigmoid(x5)

        return nn.Sigmoid()(x5), x5
  #      return nn.Sigmoid(x5),x5



'''
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Generator1((1,256,256),(3,256,256))
model2 = Discriminator1((3,256,256))
model.to(device)
model2.to(device)
print(summary(model,[(1,256,256),(3,256,256)]))
print(summary(model2,(3,256,256)))
'''


# Noise Remover Network (BiNet)
class Generator2(nn.Module):
    def __init__(self, image_input_shape):
    #def __init__(self):
        super(Generator2, self).__init__()
        _,image_channel, _, _ = image_input_shape

        self.img_enc1 = Encoder(image_channel, 32)
        self.content_enc2 = Encoder(32, 64)
        self.content_enc3 = Encoder(64, 128)
        self.content_enc4 = Encoder(128, 256)
        self.content_enc5 = Encoder(256, 256)
        self.content_enc6 = Encoder(256, 256)
        self.content_enc7 = Encoder(256, 256)
        self.content_enc8 = Encoder2(256, 256)

        self.dec1 = Decoder(256, 256, dropout=0.5)
        self.dec2 = Decoder(512, 256, dropout=0.5)
        self.dec3 = Decoder(512, 256, dropout=0.5)
        self.dec4 = Decoder(512, 256)
        self.dec5 = Decoder(512, 128)
        self.dec6 = Decoder(256, 64)
        self.dec7 = Decoder(128, 32)
        self.dec8 = Decoder2(64, 1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,0.0,0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0.0,0.02)
                nn.init.constant_(m.bias,0)


    def forward(self, x):
        e1 = self.img_enc1(x)
        e2 = self.content_enc2(e1)
        e3 = self.content_enc3(e2)
        e4 = self.content_enc4(e3)
        e5 = self.content_enc5(e4)
        e6 = self.content_enc6(e5)
        e7 = self.content_enc7(e6)
        e8 = self.content_enc8(e7)

        d1 = self.dec1(e8)
        d1 = torch.cat((d1, e7), 1)
        d2 = self.dec2(d1)
        d2 = torch.cat((d2, e6), 1)
        d3 = self.dec3(d2)
        d3 = torch.cat((d3, e5), 1)
        d4 = self.dec4(d3)
        d4 = torch.cat((d4, e4), 1)
        d5 = self.dec5(d4)
        d5 = torch.cat((d5, e3), 1)
        d6 = self.dec6(d5)
        d6 = torch.cat((d6, e2), 1)
        d7 = self.dec7(d6)
        d7 = torch.cat((d7, e1), 1)
        d8 = self.dec8(d7)

        return nn.Tanh()(d8)

class Discriminator2(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator2, self).__init__()
        _, input_channels,_,_ = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 2,2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(64, momentum=0.9)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2,2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128, momentum=0.9)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1,2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256, momentum=0.9)
        )
        self.linear = nn.Linear(256*32*32,1, bias=True)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight,0.0,0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0.0,0.02)
                nn.init.constant_(m.bias,0)

    def forward(self, x):
        bs = x.size(0)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = x4.reshape(bs, -1)
        x5 = self.linear(x4)
    #    sig = nn.Sigmoid(x5)

        return nn.Sigmoid()(x5), x5
  #      return nn.Sigmoid(x5),x5