#!/usr/bin/python3
import threading
import pickle

from matplotlib import pyplot as plt
import numpy as np
from training_utils.ImageDataset import ImageDataset
from torch import device
cpu = device("cpu")

##############################################################
#	INPUT ARGS
##############################################################
path_to_nn = "/home/lorenzo/catkin_data/models/gallery_detection_nn/gallery_detector_v3_lr5e-05_bs512_ne512.pickle"
path_to_dataset= "/home/lorenzo/catkin_data/datasets/2d_gallery_detection/test_dataset"

##############################################################
#	Import the network
##############################################################
with open(path_to_nn, "rb") as f:
    model = pickle.load(f, fix_imports=True)

##############################################################
#	Import the dataset
##############################################################
dataset = ImageDataset(path_to_dataset, do_augment=False)
fig = plt.figure(figsize=(10,10))

model = model.to(cpu)
def thread_target():
        
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312) 
    ax3 = plt.subplot(313)
    for image, label in dataset:
        image = image.to(cpu)[None, :]
        label = label.to(cpu)

        prediction = model(image)
        plt.sca(ax1)
        plt.gca().clear()
        plt.imshow(np.reshape(image.detach().numpy(), (16,720)))
        plt.sca(ax2)
        plt.gca().clear()
        np_label = label.detach().numpy().flatten()
        np_label = np.roll(np.flip(np_label),180)
        plt.plot(np_label)
        plt.sca(ax3)
        plt.gca().clear()
        np_pred = prediction.detach().numpy().flatten()
        np_pred = np.roll(np.flip(np_pred),180)
        plt.plot(np_pred)
        plt.draw()
        input()

th = threading.Thread(target = thread_target)

th.start()
plt.show()
th.join()