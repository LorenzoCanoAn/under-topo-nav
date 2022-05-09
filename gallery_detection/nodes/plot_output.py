#!/bin/python3

from std_msgs.msg import Float32MultiArray
import rospy
import matplotlib.pyplot as plt
import numpy as np
def callback(msg, angles):
    assert isinstance(msg, Float32MultiArray)
    plt.gca().clear()
    raw_vector = np.array(msg.data)
    inverted = np.flip(raw_vector)
    shifted = np.roll(inverted,180)
    plt.plot(angles[0].flatten(), shifted.flatten())
    plt.gca().set_theta_zero_location("N")
    plt.gca().set_rlim([0,1])
    plt.draw()
    
def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    angles = np.arange(0,360,1)/360 * 2 * np.math.pi
    rospy.init_node("plotter")

    rospy.Subscriber("/gallery_detection_vector",Float32MultiArray,callback=callback,callback_args=[angles])
    plt.show()

if __name__ == "__main__":
    main()