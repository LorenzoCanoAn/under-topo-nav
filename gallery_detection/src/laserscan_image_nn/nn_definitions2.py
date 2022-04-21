from torch import dropout, nn
from torchsummary import summary
import torch

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
            nn.Linear(2880*2,2880),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(2880,1440),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(1440,720),
            nn.Dropout(p=0.05),
            nn.ReLU(),

            nn.Linear(720, 360),
            nn.ReLU()
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits

class gallery_detector_v4(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v4, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.MaxPool2d([1,2]),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),

            nn.Conv2d(32,64,[3,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),

            nn.Conv2d(64,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),

            nn.Conv2d(32,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.ReLU(),
            
            nn.Conv2d(16,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Flatten(),

            nn.Linear(1440,720),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(720,360),
            nn.ReLU()
            
            
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits
class gallery_detector_v4_1(nn.Module):
    """Added BatchNorm2D to the convolutional layers and dropuout"""
    def __init__(self):
        super(gallery_detector_v4_1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.MaxPool2d([1,2]),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Flatten(),

            nn.Linear(1440,720),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(720,360),
            nn.ReLU(),           
            
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits
class gallery_detector_v4_1_small(nn.Module):
    """Added BatchNorm2D to the convolutional layers and dropuout"""
    def __init__(self):
        super(gallery_detector_v4_1_small, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.MaxPool2d([1,2]),
            nn.ReLU(),

            nn.Conv2d(8,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,32,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,16,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16,8,[3,3],padding=(0,1),padding_mode="circular"),
            nn.BatchNorm2d(8),
            nn.MaxPool2d([2,2]),
            nn.ReLU(),
            nn.Flatten(),

            nn.Linear(720,720),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(720,360),
            nn.ReLU(),           
            
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        # X should be an image with floats from 0 to 1
        logits = self.layers(x)
        return logits
if __name__ == "__main__":
    model = gallery_detector_v4()
    print(summary(model.to(torch.device("cuda")), (1, 16, 720)))
