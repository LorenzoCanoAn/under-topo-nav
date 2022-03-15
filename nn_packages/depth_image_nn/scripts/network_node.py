#!/bin/python3

import torch
import numpy as np
from image_tools import ImageTools

import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
from matplotlib import pyplot as plt
from nn_class_definitions import gallery_detector_v2
# insert at 1, 0 is the script path (or '' in REPL)
PLOT = False
class scan_handler:
    def __init__(self,plot=True):
        self.PATH_TO_MODEL = "/home/lorenzo/catkin_ws/src/lorenzo/pointcloud_nn/trained_nets_v4/gallery_detector_v2_loss_MSELoss_lr_9e-05_N_128__"
        self.image_tools = ImageTools()
        self.plot = plot
        self.init_network()
        if self.plot:
            self.figure = plt.figure()
        self.image_subscriber = rospy.Subscriber(
        "/laser_depth_image", sensor_msg.Image, self.image_callback
        )
        self.detection_publisher = rospy.Publisher("/gallery_detection_vector",std_msg.Float32MultiArray,queue_size=10)

    def init_network(self):
        self.net = gallery_detector_v2()
        self.net.load_state_dict(torch.load(self.PATH_TO_MODEL))
        self.net.eval()
        self.net = self.net.float()


    def image_callback(self, msg):
        depth_image = self.image_tools.convert_ros_msg_to_cv2(
                            msg, image_encoding="mono8"
                        )
        depth_image_tensor = torch.reshape(torch.tensor(depth_image).float(),[1,1,100,360])
        output = self.net(depth_image_tensor)
        output = output.cpu().detach().numpy()
        output = output[0,:]

        dimensions = std_msg.MultiArrayDimension()
        dimensions.label = "0"
        dimensions.size = output.__len__()
        dimensions.stride = 1

        layout = std_msg.MultiArrayLayout()
        layout.dim.append(dimensions)
        layout.data_offset = 0

        output_message = std_msg.Float32MultiArray()
        output_message.data = output.astype("float32")
        output_message.layout = layout
        self.detection_publisher.publish(output_message)

        if self.plot:
            plt.cla()
            plt.plot(np.linspace(-180,180,360),output)
            plt.draw()
            

def main():
    rospy.init_node("gallery_network")
    plot = PLOT
    my_scan_handler = scan_handler(plot=plot)
    if plot:
        plt.show()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()