#!/bin/python3
import argparse
import pickle
import torch
from laserscan_image_nn.nn_definitions2 import *
from training_utils.ImageDataset import ImageDataset
from torchsummary import summary
import torch.utils.data as data_utils
from training_utils.training_utils import load_class, basic_train
import torch.nn as nn
import os
import glob
cuda = torch.device('cuda')

##############################################################
#	Configuration of the parser
##############################################################
NET = gallery_detector_v4_1_small
dataset_name = "transformed_old_dataset"
dataset_type = "2d_gallery_detection"
model_save_folder = "/home/lorenzo/catkin_data/models/gallery_detection_nn"
n_epochs = 16
batch_size = 1024
base_lr = 0.001
LR = []#[0.002,0.001,0.0009,0.0007,0.0005,0.0003,0.0001]
multiplicators = [1,0.8,0.6,0.4,0.2]
for m in multiplicators:
    LR.append(base_lr*m)


# CLEAR THE TENSORBOARD
files = glob.glob("/home/lorenzo/tensor_board/*")
for f in files:
    os.remove(f)

##############################################################
#	Summary
##############################################################
print("Training net of type : {}".format(NET.__name__))
print(summary(NET().to(cuda), (1, 16, 360)))

##############################################################
# Other arguments
##############################################################
dataset_root_folder = "/home/lorenzo/catkin_data/datasets"
path_to_dataset = os.path.join(
    dataset_root_folder, dataset_type, dataset_name)

##############################################################
#	Load Dataset
##############################################################
dataset = ImageDataset(path_to_dataset,do_augment=False)

train_len = int(dataset.__len__() * 0.9)
test_len = int(dataset.__len__() - train_len)

train_dataset, test_dataset = data_utils.random_split(
    dataset, [train_len, test_len])



torch.cuda.empty_cache()

train_dataloader = data_utils.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
test_dataloader = data_utils.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)
    
model = NET()
model = model.to(cuda).float()
for lr in LR:
    
    model_file = f"{model.__class__.__name__}_lr{lr}_bs{batch_size}_ne{n_epochs}"
    model_save_file = os.path.join(model_save_folder, model_file+".pickle")
    if not os.path.isdir(model_save_folder):
        os.mkdir(model_save_folder)

    criterion = nn.MSELoss()

    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )
    loss_hist = basic_train(
        model, train_dataloader, criterion, optimizer, n_epochs, cuda,lr
    )
    print(f"\n Saving model in: {model_save_file}")
    with open(model_save_file, "wb+") as f:
        pickle.dump(model, f)
