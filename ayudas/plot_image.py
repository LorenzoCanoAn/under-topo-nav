import rospy

import matplotlib.pyplot as plt
import std_msgs.msg as ros_std_msg
import numpy as np
from image_tools import ImageTools
from sensor_msgs.msg import Image


class every_other_plot:

    def __init__(self):
        self.image_tools = ImageTools()
        self.switch = 0


    def image_callback(self, msg: Image):
        """ This function should take the input from the neural network, and 
            translate it to quadrants"""
        self.switch = (self.switch + 1) % 2
        if not self.switch:
            plt.clf()

            image = self.image_tools.convert_ros_msg_to_cv2(msg,image_encoding="mono8")
            plt.imshow(image)
            plt.draw()


def main():
    rospy.init_node("ahhh")
    fig = plt.figure()


    plot_handler = every_other_plot()

    gallery_subscriber = rospy.Subscriber(
        "/laserscan_image", Image, plot_handler.image_callback)
    plt.show()
    rospy.spin()


if __name__ == "__main__":
    main()
