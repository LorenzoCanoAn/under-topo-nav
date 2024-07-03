#!/bin/python3

from std_msgs.msg import Float32, Float32MultiArray
import rospy
import matplotlib.pyplot as plt
import numpy as np
import math
import threading
import time

PI = math.pi
import io
import cv_bridge
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image


class Plotter:
    def __init__(self):
        self._updated_detected_galleries = False
        self._updated_nn_output = False
        self._updated_tracked_galleries = False
        self._updated_back_gallery = False
        self._updated_followed_gallery = False
        self._updated_corrected_angle = False
        rospy.Subscriber(
            "/gallery_detection_vector",
            Float32MultiArray,
            callback=self.gallery_detection_vector_callback,
        )
        self.publisher = rospy.Publisher("/histogram_plot", ImageMsg, queue_size=1)
        self.draw_loop()

    def gallery_detection_vector_callback(self, msg):
        gallery_detection_vector = np.array(msg.data)
        angles = np.linspace(0, 2 * PI, len(gallery_detection_vector))
        self._gallery_detection_vector = gallery_detection_vector
        self._angles = angles
        self._updated_nn_output = True

    def draw_loop(self):
        # INIT THE PYPLOT VARIABLES
        fig = plt.figure(figsize=(5, 5))
        self._ax1 = fig.add_subplot(111)
        while True:
            if self._updated_nn_output:
                self._ax1.hist([self._gallery_detection_vector], bins=30)
                break
        self._ax1.tick_params(labelsize=20)
        # self._ax1.set_ylim([0, 1])
        fig.canvas.draw()
        self._ax1background = fig.canvas.copy_from_bbox(self._ax1.bbox)
        self.bridge = cv_bridge.CvBridge()
        while not rospy.is_shutdown():
            # set new drawings
            # NN OUTPUT
            if not self._updated_nn_output:
                time.sleep(0.05)
                continue
            self._ax1.cla()
            fig.canvas.restore_region(self._ax1background)
            # redraw the datapoints
            self._ax1.hist([self._gallery_detection_vector], bins=30, range=(0, 1))
            # fill the axes rectangle
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            pil_img = Image.open(buf)
            pil_img.convert("RGB")
            opencv_img = np.array(pil_img)[:, :, :3][:, :, ::-1]
            self.publisher.publish(self.bridge.cv2_to_imgmsg(opencv_img))


def main():
    rospy.init_node("plotter")
    rospy_thread = threading.Thread(target=rospy.spin).start()
    plotter = Plotter()
    rospy_thread.join()


if __name__ == "__main__":
    main()
