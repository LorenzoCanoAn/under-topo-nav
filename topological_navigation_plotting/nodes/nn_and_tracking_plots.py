#!/bin/python3

from std_msgs.msg import Float32, Float32MultiArray
import rospy
import matplotlib.pyplot as plt
import numpy as np
import math
import threading

PI = math.pi


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
        rospy.Subscriber(
            "/currently_detected_galleries",
            Float32MultiArray,
            callback=self.detected_galleries_callback,
        )
        rospy.Subscriber(
            "/tracked_galleries",
            Float32MultiArray,
            callback=self.tracked_galleries_callback,
        )
        rospy.Subscriber("/back_gallery", Float32, callback=self.back_gallery_callback)
        rospy.Subscriber(
            "/followed_gallery", Float32, callback=self.followed_gallery_callback
        )
        rospy.Subscriber(
            "/corrected_angle", Float32, callback=self.corrected_angle_callback
        )
        self.draw_loop()

    def gallery_detection_vector_callback(self, msg):
        raw_vector = np.array(msg.data)
        # inverted = np.flip(raw_vector)
        # gallery_detection_vector = np.roll(inverted, 180)
        angles = np.linspace(0, 2 * PI, len(raw_vector))
        self._gallery_detection_vector = raw_vector
        self._angles = angles
        self._updated_nn_output = True

    def detected_galleries_callback(self, msg):
        data = np.array(msg.data)
        assert len(data) % 2 == 0
        self._detected_galleries = np.reshape(data, [2, -1]).T
        self._updated_detected_galleries = True

    def tracked_galleries_callback(self, msg):
        data = np.array(msg.data)
        assert len(data) % 2 == 0
        self._tracked_galleries = np.reshape(data, [2, -1]).T
        self._tracked_galleries[:, 1] /= 20
        self._updated_tracked_galleries = True

    def back_gallery_callback(self, msg):
        self._back_gallery = np.array([msg.data, 0.9])
        self._updated_back_gallery = True

    def followed_gallery_callback(self, msg):
        self._followed_gallery = np.array([msg.data, 0.9])
        self._updated_followed_gallery = True

    def corrected_angle_callback(self, msg):
        self._corrected_angle = np.array([msg.data, 0.8])
        self._updated_corrected_angle = True

    def draw_loop(self):
        # INIT THE PYPLOT VARIABLES
        fig = plt.figure(figsize=(5, 5))
        self._ax1 = fig.add_subplot(111, polar=True)
        (self._nn_output_lines,) = self._ax1.plot([], lw=6, c="b")
        self._currently_detected_scatter = self._ax1.scatter([], [], c="b", s=300)
        self._tracked_galleries_scatter = self._ax1.scatter([], [], c="r", s=300)
        self._back_gallery_scatter = self._ax1.scatter(
            [], [], color="k", s=500, marker="P"
        )
        self._followed_gallery_scatter = self._ax1.scatter(
            [], [], color="c", s=500, marker="P"
        )
        self._corrected_angle_scatter = self._ax1.scatter(
            [], [], color="g", s=500, marker="P"
        )
        self._ax1.tick_params(labelsize=20)
        self._ax1.set_ylim([0, 1])
        self._ax1.set_theta_zero_location("N")
        fig.canvas.draw()
        self._ax1background = fig.canvas.copy_from_bbox(self._ax1.bbox)
        plt.show(block=False)

        while not rospy.is_shutdown():
            # set new drawings
            # NN OUTPUT
            if self._updated_nn_output:
                self._nn_output_lines.set_data(
                    self._angles, self._gallery_detection_vector
                )
            if False:  # self._updated_detected_galleries:
                self._currently_detected_scatter.set_offsets(self._detected_galleries)
            if self._updated_tracked_galleries:
                self._tracked_galleries_scatter.set_offsets(self._tracked_galleries)
            if self._updated_back_gallery:
                self._back_gallery_scatter.set_offsets(self._back_gallery)
            if self._updated_followed_gallery:
                self._followed_gallery_scatter.set_offsets(self._followed_gallery)
            if self._updated_corrected_angle:
                self._corrected_angle_scatter.set_offsets(self._corrected_angle)
            # restore background
            fig.canvas.restore_region(self._ax1background)
            # redraw the datapoints
            self._ax1.draw_artist(self._nn_output_lines)
            self._ax1.draw_artist(self._currently_detected_scatter)
            self._ax1.draw_artist(self._tracked_galleries_scatter)
            self._ax1.draw_artist(self._back_gallery_scatter)
            self._ax1.draw_artist(self._followed_gallery_scatter)
            self._ax1.draw_artist(self._corrected_angle_scatter)
            # fill the axes rectangle
            fig.canvas.blit(self._ax1.bbox)
            fig.canvas.flush_events()


def main():
    rospy.init_node("plotter")
    rospy_thread = threading.Thread(target=rospy.spin).start()
    plotter = Plotter()
    rospy_thread.join()


if __name__ == "__main__":
    main()
