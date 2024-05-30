#!/bin/python3

import time
import matplotlib
from matplotlib.axis import Axis
import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import rospy
import matplotlib.pyplot as plt
import numpy as np
import math
import threading
import io
import cv_bridge
from sensor_msgs.msg import Image as ImageMsg
from PIL import Image


PI = math.pi

SINGLE_ANGLE_PLOT_DIST = 10


class Plotter:
    def __init__(self):
        self._angles_updated = False
        self._desired_angle_updated = False
        self._corrected_angle_updated = False
        self._scanner_updated = False
        self._final_weight_updated = False
        self._desired_angle_weight_updated = False
        self._laser_scan_weight_updated = False
        self._plot_on_window = rospy.get_param("~plot_on_window", False)
        self._plot_on_rviz = rospy.get_param("~plot_on_rviz", True)

        rospy.Subscriber("/oa_angles", std_msgs.Float32MultiArray, callback=self.angles_callback)

        while not self._angles_updated:
            time.sleep(0.5)

        rospy.Subscriber(
            "/predicted_bearing",
            std_msgs.Float32,
            callback=self.desired_angle_callback,
        )
        rospy.Subscriber(
            "/corrected_bearing",
            std_msgs.Float32,
            callback=self.corrected_angle_callback,
        )
        rospy.Subscriber("/scan", sensor_msgs.LaserScan, callback=self.scanner_callback)
        rospy.Subscriber(
            "/oa_final_weight",
            std_msgs.Float32MultiArray,
            callback=self.final_weight_callback,
        )
        rospy.Subscriber(
            "/oa_desired_angle_weight",
            std_msgs.Float32MultiArray,
            callback=self.desired_angle_weight_callback,
        )
        rospy.Subscriber(
            "/oa_laser_scan_weight",
            std_msgs.Float32MultiArray,
            callback=self.laser_scan_weight_callback,
        )
        if self._plot_on_rviz:
            self.publisher = rospy.Publisher("/oa_plot", ImageMsg, queue_size=1)
        self.draw_loop()

    def angles_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32MultiArray)
        self._angles_plotting_data = msg.data

        self._angles_updated = True

    def desired_angle_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32)
        x = [msg.data + i for i in [-0.01, 0, +0.01]]
        y = [0, SINGLE_ANGLE_PLOT_DIST, 0]
        self._desired_angle_plotting_data = np.reshape(np.array([x, y]), [2, -1])

        self._desired_angle_updated = True

    def corrected_angle_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32)
        x = [msg.data + i for i in [-0.01, 0, +0.01]]
        y = [0, SINGLE_ANGLE_PLOT_DIST, 0]
        self._corrected_angle_plotting_data = np.reshape(np.array([x, y]), [2, -1])

        self._corrected_angle_updated = True

    def scanner_callback(self, msg):
        assert isinstance(msg, sensor_msgs.LaserScan)
        self._scanner_plotting_data = np.reshape(
            np.array([self._angles_plotting_data, msg.ranges]), [2, -1]
        ).T

        self._scanner_updated = True

    def final_weight_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32MultiArray)
        self._final_weight_plotting_data = np.reshape(
            np.array([self._angles_plotting_data, msg.data]), [2, -1]
        ).T

        self._final_weight_updated = True

    def desired_angle_weight_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32MultiArray)
        self._desired_angle_weight_plotting_data = np.reshape(
            np.array([self._angles_plotting_data, msg.data]), [2, -1]
        ).T

        self._desired_angle_weight_updated = True

    def laser_scan_weight_callback(self, msg):
        assert isinstance(msg, std_msgs.Float32MultiArray)
        self._laser_scan_weight_plotting_data = np.reshape(
            np.array([self._angles_plotting_data, msg.data]), [2, -1]
        ).T

        self._laser_scan_weight_updated = True

    def draw_loop(self):
        # INIT THE PYPLOT VARIABLES
        fig = plt.figure(figsize=(5, 5))
        self._ax1 = fig.add_subplot(111, polar=True)

        (self._desired_angle_artist,) = self._ax1.plot([], lw=6, c="k")
        self._laserscan_artist = self._ax1.scatter([], [], c="k", s=40)
        self._desired_angle_weight_artist = self._ax1.scatter([], [], c="r", s=40)
        self._laserscan_weight_artist = self._ax1.scatter([], [], c="b", s=40)
        self._final_weight_artist = self._ax1.scatter([], [], c="g", s=40)
        (self._corrected_angle_artist,) = self._ax1.plot([], lw=6, c="g")

        self._ax1.tick_params(labelsize=20)
        self._ax1.set_ylim([0, 5])
        self._ax1.set_theta_zero_location("N")
        fig.canvas.draw()
        self._ax1background = fig.canvas.copy_from_bbox(self._ax1.bbox)
        if self._plot_on_window:
            plt.show(block=False)
        if self._plot_on_rviz:
            self.bridge = cv_bridge.CvBridge()

        while not rospy.is_shutdown():
            # set new drawings
            # NN OUTPUT
            if self._desired_angle_updated:
                self._desired_angle_artist.set_data(
                    self._desired_angle_plotting_data[0],
                    self._desired_angle_plotting_data[1],
                )
            if self._scanner_updated:
                self._laserscan_artist.set_offsets(self._scanner_plotting_data)
            if self._desired_angle_weight_updated:
                self._desired_angle_weight_artist.set_offsets(
                    self._desired_angle_weight_plotting_data
                )
            if self._laser_scan_weight_updated:
                self._laserscan_weight_artist.set_offsets(self._laser_scan_weight_plotting_data)
            if self._final_weight_updated:
                self._final_weight_artist.set_offsets(self._final_weight_plotting_data)
            if self._corrected_angle_updated:
                self._corrected_angle_artist.set_data(
                    self._corrected_angle_plotting_data[0],
                    self._corrected_angle_plotting_data[1],
                )

            # restore background
            fig.canvas.restore_region(self._ax1background)
            # redraw the datapoints
            self._ax1.draw_artist(self._desired_angle_artist)
            self._ax1.draw_artist(self._laserscan_artist)
            self._ax1.draw_artist(self._desired_angle_weight_artist)
            self._ax1.draw_artist(self._laserscan_weight_artist)
            self._ax1.draw_artist(self._final_weight_artist)
            self._ax1.draw_artist(self._corrected_angle_artist)
            # fill the axes rectangle
            if self._plot_on_window:
                fig.canvas.blit(self._ax1.bbox)
                fig.canvas.flush_events()
            if self._plot_on_rviz:
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
