#!/bin/python3

from std_msgs.msg import Float32MultiArray
import rospy
import matplotlib.pyplot as plt
import numpy as np


def tracked_callback(msg):
    assert isinstance(msg, Float32MultiArray)
    plt.gca().clear()
    x = np.array(msg.data)
    y = np.ones(len(x))
    plt.scatter(x, y)
    plt.gca().set_theta_zero_location("N")
    plt.gca().set_rlim([0, 1])
    plt.draw()


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    rospy.init_node("plotter")

    rospy.Subscriber(
        "/tracked_galleries",
        Float32MultiArray,
        callback=callback,
    )
    plt.show()


if __name__ == "__main__":
    main()
