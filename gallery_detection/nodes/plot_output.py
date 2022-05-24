#!/bin/python3

from std_msgs.msg import Float32MultiArray, Float32
import rospy
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        self.angles = (np.arange(0, 360, 1) / 360 * 2 * np.math.pi).flatten()
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.followed_gallery = None
        self.back_gallery = None
        rospy.Subscriber(
            "/tracked_galleries",
            Float32MultiArray,
            callback=self.tracked_callback,
        )
        rospy.Subscriber(
            "/gallery_detection_vector",
            Float32MultiArray,
            callback=self.callback,
        )
        rospy.Subscriber(
            "/followed_gallery",
            Float32,
            callback=self.gallery_to_follow_callback,
        )
        rospy.Subscriber(
            "/back_gallery",
            Float32,
            callback=self.back_gallery_callback,
        )

    def callback(self, msg):
        assert isinstance(msg, Float32MultiArray)
        plt.gca().clear()
        raw_vector = np.array(msg.data)
        inverted = np.flip(raw_vector)
        shifted = np.roll(inverted, 180)
        plt.plot(self.angles, shifted.flatten())
        plt.scatter(self.x, self.y)
        if not self.followed_gallery is None:
            plt.scatter(self.followed_gallery, 0.5, c="r")
        if not self.back_gallery is None:
            plt.scatter(self.back_gallery, 0.5, c="k")
        plt.gca().set_theta_zero_location("N")
        plt.gca().set_rlim([0, 1])
        plt.draw()

    def tracked_callback(self, msg):
        assert isinstance(msg, Float32MultiArray)
        self.x = np.array(msg.data)
        self.y = np.ones(len(self.x))

    def gallery_to_follow_callback(self, msg):
        assert isinstance(msg, Float32)
        self.followed_gallery = msg.data

    def back_gallery_callback(self, msg):
        assert isinstance(msg, Float32)
        self.back_gallery = msg.data


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    rospy.init_node("plotter")
    plotter = Plotter()
    plt.show()


if __name__ == "__main__":
    main()
