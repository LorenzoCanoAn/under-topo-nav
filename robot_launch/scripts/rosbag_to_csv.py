import rosbag
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

PATH_TO_FILE = "/home/lorenzo/intersection_vector_2024-06-27-18-09-25.bag"

bag = rosbag.Bag(PATH_TO_FILE)


stacked_vectors = np.zeros((0, 360))
for topic, msg, stamp in bag.read_messages():
    if topic == "/gallery_detection_vector":
        vector = np.array(msg.data).reshape((1, -1))
        stacked_vectors = np.vstack((stacked_vectors, vector))
with open("/home/lorenzo/experiment.csv", "w") as f:
    np.savetxt(f, stacked_vectors[100:])


fig = plt.figure()
ax1 = fig.add_subplot(111)
(nn_output_lines,) = ax1.plot([], lw=6, c="b")
ax1.set_xlim(0, 360)
ax1.set_ylim(0, 1.2)
fig.canvas.draw()
ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
plt.show(block=False)
angles = np.linspace(0, 360, 360)
for a in tqdm(stacked_vectors[100:], total=len(stacked_vectors[100:])):
    fig.canvas.restore_region(ax1background)
    nn_output_lines.set_data(angles, a)
    ax1.draw_artist(nn_output_lines)
    fig.canvas.blit(ax1.bbox)
    fig.canvas.flush_events()
    time.sleep(0.1)
