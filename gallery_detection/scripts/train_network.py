#!/bin/python3
import argparse
from syslog import LOG_DEBUG
from training_utils.ImageDataset import ImageDataset
from training_utils.training_utils import load_class

##############################################################
#	Configuration of the parser
##############################################################
parser = argparse.ArgumentParser(description='Tests a gallery detection Neural Network.')
parser.add_argument("nn_type", type=str)
parser.add_argument("nn_name", type=str)
parser.add_argument("path_to_dataset", type=str)
parser.add_argument("--fraction_to_test", type=int)
args = parser.parse_args()

##############################################################
#	Import the nn
##############################################################
NET = load_class(f"{args.nn_type}.nn_definitions.{args.nn_name}")
print("Training net of type : {}".format(NET.__name__))

##############################################################
#	Load Dataset
##############################################################
dataset = ImageDataset(args.path_to_dataset)
exit()
import torch
import torch.utils.data as data_utils
from torchvision import transforms
import numpy as np
from torchsummary import summary
from torch import nn
import matplotlib.pyplot as plt
from laserscan_image_nn import *
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
DATASET_FOLDER = "/home/lorenzo/Datasets/gallery_detection/laserscan_image_polished"
PATH_TO_MODEL = "/home/lorenzo/catkin_ws/data/trained_nets/gallery_detection_nets/laserscan_image_based/gallery_detector_v3_loss_MSELoasdfss_lr_0.0005_N_16__"
UPDATE_MODEL = False

net = NET().to(device)
summary(net,(1,16,360))

DATASET_FOLDER = "/home/lorenzo/Datasets/gallery_detection/laserscan_image_polished"
dataset = ImageDataset(path_to_dataset=DATASET_FOLDER,do_augment=False)
i = 0


train_len = int(dataset.__len__() * 0.9)
test_len = int(dataset.__len__() - train_len)

train_dataset, test_dataset = data_utils.random_split(dataset,[train_len, test_len])

net = NET()
torch.cuda.empty_cache()
LR = [0.0002, 0.0001, 0.00005, 0.00001]
BATCH_SIZE = 512
N_EPOCHS = 32

train_dataloader = data_utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)    
test_dataloader = data_utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=5)    


criterion = nn.MSELoss()
if UPDATE_MODEL:
    net.load_state_dict(torch.load(PATH_TO_MODEL))
net = net.to(device).float()
for lr in LR:
    torch.cuda.empty_cache()

    print(
        "type: {}, loss: {}, lr: {}".format(
            NET.__name__, criterion.__class__.__name__, lr
        )
    ) 
    
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=lr,
    )
    loss_hist = basic_train(
        net, train_dataloader,test_dataloader, criterion, optimizer, N_EPOCHS
    )
    NN_PATH = "/home/lorenzo/catkin_ws/data/trained_nets/gallery_detection_nets/laserscan_image_based/{}_losss_{}_lr_{}_N_{}_small".format(
        NET.__name__, criterion.__class__.__name__, lr, N_EPOCHS
    )
    print(NN_PATH)
    torch.save(net.state_dict(), NN_PATH)