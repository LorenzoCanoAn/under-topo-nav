#!/usr/bin/python3
import pointcloud_to_image.conversion_functions as conversors
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
import cv2
import numpy as np
from enum import Enum

# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------


class ConversorTypes(Enum):
    ptcl_to_height_depth_image = 1
    ptcl_to_angle_depth_image = 2
    laserscan_to_cenital_img = 3


class ConversionNode:
    def __init__(self):
        rospy.init_node("ptcl_to_img_node")
        self.get_params()
        self.setup_conversor()
        self.seq = 0
        self.setup_sub_pub()

    def setup_sub_pub(self):
        self._publisher = rospy.Publisher(self.output_topic, Image, queue_size=1)
        if self.normalize:
            self._norm_publisher = rospy.Publisher(
                self.output_topic + "_norm", Image, queue_size=1
            )
        self._bridge = CvBridge()
        self._subscriber = rospy.Subscriber(
            self.input_topic, PointCloud2, callback=self.callback
        )

    def get_params(self):
        self.conversor_type = rospy.get_param("~/conversor_type", default=1)
        self.input_topic = rospy.get_param(
            "~/input_topic", default="/velodyne_points", type=str
        )
        self.output_topic = rospy.get_param(
            "~/output_topic", default="/lidar_image", type=str
        )
        self.normalize = rospy.get_param("~/normalize", default=True, type=bool)
        self.output_width = rospy.get_param("~/output_width", default=None, type=int)
        self.output_height = rospy.get_param("~/output_height", default=None, type=int)

    def setup_conversor(self):
        if self.conversor_type == ConversorTypes.ptcl_to_height_depth_image:
            self.input_topic_type = PointCloud2
            self.conversor = conversors.PtclToHeightImageConversor()
        elif self.conversor_type == ConversorTypes.ptcl_to_angle_depth_image:
            self.input_topic_type = PointCloud2
            self.conversor = conversors.PtclToAngleImageConversor()
        elif self.conversor_type == ConversorTypes.laserscan_to_cenital_img:
            self.input_topic_type = LaserScan
            self.conversor = conversors.ScanToCenithImage()

    def callback(self, msg):
        # Create image
        image = self.conversor(msg)
        self.seq += 1
        stamp = rospy.Time.now()
        if image.shape[0] != self.height or image.shape[1] != self.widht:
            image = cv2.resize(
                image,
                dsize=(self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        img_msg = self._bridge.cv2_to_imgmsg(image, "32FC1")
        img_msg.header.stamp = stamp
        img_msg.header.seq = self.seq
        self._publisher.publish(img_msg)
        if self.normalize:
            norm_img = image / np.max(image)
            norm_img_msg = self._bridge.cv2_to_imgmsg(norm_img, "32FC1")
            norm_img_msg.header.stamp = stamp
            norm_img_msg.header.seq = self.seq
            # Send image message
            self._norm_publisher.publish(norm_img_msg)

    def run(self):
        rospy.spin()


def main():
    node = ConversionNode()
    node.run()


if __name__ == "__main__":
    main()
