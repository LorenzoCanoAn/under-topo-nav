DATASET_PATH = "/home/lorenzo/catkin_data/datasets/2d_gallery_detection/r_1654610759595649467_s1.0"


from threading import Thread
import pickle

import os
from time import time_ns as ns
import matplotlib.pyplot as plt
import numpy as np

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def plot_label(label):
    label = np.roll(np.flip(label),180)
    plt.plot(label)
    plt.xlim((0,360))

def plotting_thread_target(dataset):
        
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    for i, (image, label) in enumerate(dataset):    
        np_image = image.numpy()
        np_label = label.numpy()
        plt.sca(ax1)
        plt.gca().clear()
        if there_is_info:
            pose, yaw = info[i].split("]-[")
            pose = pose.replace("[","")
            yaw = yaw.replace("]","")
            quaternion = euler_to_quaternion(0,0,float(yaw))
            title = f"Pose: {pose}, Orientation: {quaternion}"
            print(title)
            plt.title(title)
        plt.imshow(np_image)
        plt.sca(ax2)
        plt.gca().clear()
        plot_label(np_label)
        plt.draw()
        input()
        
n = ns()
files = os.listdir(DATASET_PATH)
n_elements = len(files)
dataset = [None for _ in range(n_elements)]
for i, file in enumerate(files):
    if ".pickle" in file:
        with open(os.path.join(DATASET_PATH, file),"rb") as f:
            dataset[i] = pickle.load(f)
there_is_info = False
info_file_path = os.path.join(DATASET_PATH, "info.txt")
if os.path.isfile(info_file_path):
    there_is_info = True
    with open(info_file_path, "r") as info_file:
        info = info_file.read().split("\n")
secs = (ns()-n)/1e9
print(f"{secs} sec for {n_elements} elements")


plt.figure(figsize=(10,5))
plotting_thread = Thread(target=plotting_thread_target, args=[dataset])
plotting_thread.start()


plt.show()
plotting_thread.join()