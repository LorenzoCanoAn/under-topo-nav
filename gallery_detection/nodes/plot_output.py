#!/bin/python3

from std_msgs.msg import Float32MultiArray
import rospy
import matplotlib.pyplot as plt
def callback(msg):
    assert isinstance(msg, Float32MultiArray)
    plt.gca().clear()
    plt.plot(msg.data)
    plt.draw()
    


def main():
    plt.figure()
    rospy.init_node("plotter")
    rospy.Subscriber("/gallery_detection_vector",Float32MultiArray,callback=callback)
    plt.show()

if __name__ == "__main__":
    main()