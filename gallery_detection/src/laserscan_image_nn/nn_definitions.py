from torch import nn
import torch


class gallery_detector_v1(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(1,1),padding_mode="circular"),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Conv2d(8,16,[3,3],padding=(1,1),padding_mode="circular"),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Conv2d(16,32,[3,3],padding=(1,1),padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(1,1),padding_mode="circular"),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2880,1440),
            nn.ReLU(),
            nn.Linear(1440,720),
            nn.ReLU(),
            nn.Linear(720, 360),
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits


class gallery_detector_v2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,1,[16,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(360,360*2),
            nn.ReLU(),
            nn.Linear(360*2,360),
            nn.Sigmoid()
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits


class gallery_detector_v3(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(1,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,3],padding=(1,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.Conv2d(16,16,[3,3],padding=(1,1),padding_mode="zeros"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,3],padding=(1,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Conv2d(32,32,[3,3],padding=(1,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(2880,2880),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(2880,1440),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(1440,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v3_v2_v2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3_v2_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d([2,6]),

            nn.Flatten(),
            nn.Linear(1440,1440),
            nn.Dropout(p=0.2),
            nn.ReLU(),

            nn.Linear(1440,720),
            nn.Dropout(p=0.2),
            nn.ReLU(),

            nn.Linear(720,720),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v3_v2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(16,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.AvgPool2d([2,6]),

            nn.Flatten(),
            nn.Linear(1440,1440),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(1440,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(720,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v3_v3(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3_v3, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(16,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.AvgPool2d([2,6]),

            nn.Flatten(),
            nn.Linear(1440,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(720,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v3_v4(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3_v4, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(16,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.AvgPool2d([2,6]),

            nn.Flatten(),
            nn.Linear(1440,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v3_v5(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v3_v5, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d([2,6]),

            nn.Flatten(),
            nn.Linear(1440,720),
            nn.ReLU(),

            nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class iñigo_mas_te_vale_que_funcione(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(iñigo_mas_te_vale_que_funcione, self).__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv2d(1,32,[5,5],padding=(0,2),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([1,3]),

            nn.Conv2d(32,64,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1,3]),

            nn.Conv2d(64,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d([1,3]),

            nn.Conv2d(128,256,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(256,256),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(256, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.convolutional_block(x)
        return logits

class gallery_detector_v4_convolutional_bogaloo(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v4_convolutional_bogaloo, self).__init__()

        self.encoder = nn.Sequential(
             nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,2),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(32,64,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(64,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(128,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d([4,5]),


            nn.Flatten(),
            nn.Linear(256,32),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.Dropout(p=0.05),
            nn.ReLU(),
        )
        nz = 64
        ngf = 100
        self.generator = nn.Sequential(
            nn.ConvTranspose1d(nz,          ngf * 8,    3, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose1d(ngf * 8,     ngf * 4,    3, 3, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose1d(ngf * 4,     ngf * 2,    4, 3, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose1d( ngf * 2,    ngf,        4, 3, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose1d( ngf,        1,          5, 4, 1, bias=False),
            nn.BatchNorm1d(1),

        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        features = self.encoder(x)
        features = torch.reshape(features,(1,-1,1))
        generated = self.generator(features)

        return generated

class gallery_detector_v5_more_conv(nn.Module):
    def __init__(self):
        super(gallery_detector_v5_more_conv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(32,64,[3,3],padding=(0,2),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(64,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(128,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(128,256,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(256,256,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d([2,4]),

            nn.Flatten(),
            nn.Linear(256,512),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(512,360),
            nn.Dropout(p=0.05)
            
        )
       
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        features = self.encoder(x)

        return features
class gallery_detector_v6_no_avg_pooling(nn.Module):
    def __init__(self):
        super(gallery_detector_v6_no_avg_pooling, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([1,3]),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d([1,3]),

            nn.Conv2d(32,64,[3,3],padding=(0,2),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(64,128,[3,3],padding=(0,1),padding_mode="circular"),
            nn.Dropout(p=0.05),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d([2,4]),

            nn.Flatten(),
            nn.Linear(2560,2048),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(2048,1024),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(1024,512),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(512,360),
            nn.Dropout(p=0.05),

 
        )
       
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        features = self.encoder(x)

        return features

class lets_get_chonky(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(lets_get_chonky, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,4,[7,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(4,8,[5,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d([1,2]),

            nn.Conv2d(8,16,[5,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d([2,2]),

            nn.Flatten(),
            nn.Linear(720,720),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(720,720),
            nn.ReLU(),
            nn.Linear(720,720),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(720,360),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            #nn.Linear(1440,720),
            #nn.Dropout(p=0.2),
            #nn.ReLU(),
#
            #nn.Linear(720,720),
            #nn.Dropout(p=0.2),
            #nn.ReLU(),
            #nn.Linear(720, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class lets_get_chonky_2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(lets_get_chonky_2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,4,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4,8,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,8,[3,15],padding=(0,7),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,1,[3,15],padding=(0,7),padding_mode="circular"),
            nn.ReLU(),
            nn.AvgPool2d([2,1]),
            nn.Flatten(),

        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits


if __name__ == "__main__":
    from torchsummary import summary
    NET = lets_get_chonky_2


    import matplotlib.pyplot as plt
    print("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = NET().to(device)
    summary(net,(1,16,360))


class lets_get_chonky_2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(lets_get_chonky_2, self).__init__()

        kernel_widht = 21
        padding = int(kernel_widht / 2)

        self.layers = nn.Sequential(
            nn.Conv2d(1,4,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4,8,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(128),
            nn.ReLU(),


            nn.Conv2d(128,64,[3,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,32,[1,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,[1,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,8,[1,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,1,[1,kernel_widht],padding=(0,padding),padding_mode="circular"),
            nn.ReLU(),
            nn.AvgPool2d([2,1]),
            nn.Flatten(),

            nn.Linear(360,720),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(720,360),
            nn.Dropout(0.2),
            nn.ReLU()

        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits
NET = lets_get_chonky_2