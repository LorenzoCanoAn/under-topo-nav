#!/bin/python3

from std_msgs.msg import Float32MultiArray
import rospy
import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self):
        self.angles = (np.arange(0, 360, 1) / 360 * 2 * np.math.pi).flatten()
        self.x = np.zeros(0)
        self.y = np.zeros(0)
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

    def callback(self, msg):
        assert isinstance(msg, Float32MultiArray)
        plt.gca().clear()
        raw_vector = np.array(msg.data)
        inverted = np.flip(raw_vector)
        shifted = np.roll(inverted, 180)
        plt.plot(self.angles, shifted.flatten())
        plt.scatter(self.x, self.y)
        plt.gca().set_theta_zero_location("N")
        plt.gca().set_rlim([0, 1])
        plt.draw()

    def tracked_callback(self, msg):
        assert isinstance(msg, Float32MultiArray)
        self.x = np.array(msg.data)
        self.y = np.ones(len(self.x))


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    rospy.init_node("plotter")
    plotter = Plotter()
    plt.show()


if __name__ == "__main__":
    main()
