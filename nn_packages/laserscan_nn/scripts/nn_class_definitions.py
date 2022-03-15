from torch import nn
import torch.nn.functional as F
import torch


class intersection_detector_v1(nn.Module):
    """Detecta entre recta e intersección a 4"""
    def __init__(self):
        super(intersection_detector_v1, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(723, 723*2),
            nn.ReLU(),
            nn.Linear(723*2, 723*2),
            nn.Sigmoid(),
            nn.Linear(723*2, 2),
            nn.Softmax(1)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class intersection_detector_v2(nn.Module): 
    """Detecta entre 5 categorías:
    - Intersección T,
    - Intersección a 4
    - Curva
    - Recta
    - Fin de camino
    
    NO FUNCIONA NI PIDIENDOLO POR FAVOR"""
    def __init__(self):
        super(intersection_detector_v2, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(723, 723*2),
            nn.ReLU(),
            nn.Linear(723*2, 723*2),
            nn.Sigmoid(),
            nn.Linear(723*2, 5),
            nn.Softmax(1)
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class intersection_detector_v3(nn.Module):
    """Detecta entre 5 categorías:
    - Intersección T,
    - Intersección a 4
    - Curva
    - Recta
    - Fin de camino
    """
    def __init__(self):
        super(intersection_detector_v3, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(723, 723*2),
            nn.ReLU(),
            nn.Linear(723*2, 723*2),
            nn.ReLU(),
            nn.Linear(723*2, 723),
            nn.Sigmoid(),
            nn.Linear(723, 5),
        )
    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class conv_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv1d(1, 4, 20),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Conv1d(4, 8, 20),
            nn.ReLU(),
            nn.MaxPool1d(5),
            nn.Flatten(),
            nn.Linear(192, 192*2),
            nn.ReLU(),
            nn.Linear(192*2, 192),
            nn.ReLU(),
            nn.Linear(192, 5)
        )

    @classmethod
    def is_2d(cls):
        return False

    def forward(self, x):
        old_shape = x.shape
        new_shape = (old_shape[0],1,old_shape[1])
        x = torch.reshape(x, new_shape)
        logits = self.linear_relu_stack(x)
        logits = torch.reshape(logits, [logits.shape[0],5])
        return logits

class conv_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(1, 16, [5,5]),
            nn.ReLU(),
            nn.MaxPool2d([3,3],[2,2]),
            nn.Conv2d(16, 32, [5,5]),
            nn.ReLU(),
            nn.MaxPool2d([3,3],[2,2]),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, int(512/2)),
            nn.ReLU(),
            nn.Linear(int(512/2), int(512/4)),
            nn.ReLU(),
            nn.Linear(int(512/4), 5)
        )
    @classmethod
    def is_2d(cls):
        return True

    def forward(self, x):
        x = torch.reshape(x,[x.shape[0],1,x.shape[1],x.shape[2]]) 
        logits = self.linear_relu_stack(x)
        logits = torch.reshape(logits, [logits.shape[0],5])
        return logits