from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.modules.activation import Sigmoid


class gallery_detector_v1(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3]),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(8,16,[3,3]),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(16,32,[3,3]),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3]),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 2560 *2),
            nn.ReLU(),
            nn.Linear(2560*2, 2560),
            nn.ReLU(),
            nn.Linear(2560, 360),
            nn.Sigmoid()
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        logits = self.layers(x)
        return logits


class gallery_detector_v2(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(gallery_detector_v2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1,8,[3,3],padding_mode="circular"),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(8,16,[3,3],padding_mode="circular"),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(16,32,[3,3],padding_mode="circular"),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Conv2d(32,32,[3,3],padding_mode="circular"),
            nn.MaxPool2d([2,2],[2,2]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2560, 2560 *2),
            nn.ReLU(),
            nn.Linear(2560*2, 2560),
            nn.ReLU(),
            nn.Linear(2560, 360)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        x /= 255.0
        logits = self.layers(x)
        return logits