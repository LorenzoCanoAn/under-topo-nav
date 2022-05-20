#!/bin/python3

# This node takes the output of the neural network as an imput, and outputs a list of angles at which a gallery could be present
import std_msgs.msg as std_msg
import rospy
import numpy as np


def filter_vector(msg):
    print(msg)


def main():
    rospy.init_node("dd")

    subscriber = rospy.Subscriber(
        "/currently_detected_galleries",
        std_msg.Float32MultiArray,
        callback=filter_vector,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
