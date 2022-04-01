DATASET_PATH = "/home/lorenzo/catkin_data/datasets/2d_gallery_detection/test_dataset"


from threading import Thread
import pickle
import os
from time import time_ns as ns
import matplotlib.pyplot as plt
import numpy as np

def plot_label(label):
    label = np.roll(np.flip(label),180)
    plt.plot(label)
    plt.xlim((0,360))

def plotting_thread_target(dataset):
        
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    for image, label in dataset:
        np_image = image.numpy()
        np_label = label.numpy()
        plt.sca(ax1)
        plt.gca().clear()
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
    with open(os.path.join(DATASET_PATH, file),"rb") as f:
        dataset[i] = pickle.load(f)
secs = (ns()-n)/1e9
print(f"{secs} sec for {n_elements} elements")
sec_per_element = secs / n_elements
print(f"{sec_per_element} sec per elements elements")
expected_size = 90000
print(f"{sec_per_element * expected_size} sec for {expected_size} elements")


plt.figure(figsize=(10,5))
plotting_thread = Thread(target=plotting_thread_target, args=[dataset])
plotting_thread.start()


plt.show()
plotting_thread.join()