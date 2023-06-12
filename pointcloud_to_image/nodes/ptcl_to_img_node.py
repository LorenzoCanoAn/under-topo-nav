#!/usr/bin/python3
import pointcloud_to_image.conversion_functions as conversors
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, LaserScan
import cv2
import numpy as np
from enum import IntEnum

# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------


class ConversorTypes(IntEnum):
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

    def get_params(self):
        self.conversor_type = int(rospy.get_param("~conversor_type"))
        self.input_topic = rospy.get_param("~input_topic", default="/velodyne_points")
        self.output_topic = rospy.get_param("~output_topic", default="/lidar_image")
        self.normalize = bool(int(rospy.get_param("~normalize", default=True)))
        self.output_width = rospy.get_param("~output_width", default=None)
        self.output_height = rospy.get_param("~output_height", default=None)

    def setup_conversor(self):
        if self.conversor_type == ConversorTypes.ptcl_to_height_depth_image:
            self.input_topic_type = PointCloud2
            z_res = int(rospy.get_param("~conversor/z_res", default=100))
            z_max = float(rospy.get_param("~conversor/z_max", default=5))
            if self.output_height is None:
                self.output_height = z_res
            if self.output_width is None:
                self.output_width = 360
            self.conversor = conversors.PtclToHeightImageConversor(
                Z_RES=z_res, Z_MAX=z_max
            )
        elif self.conversor_type == ConversorTypes.ptcl_to_angle_depth_image:
            self.input_topic_type = PointCloud2
            n_rays = int(rospy.get_param("~conversor/n_rays", default=16))
            cutoff_distance = int(
                rospy.get_param("~conversor/cutoff_distance", default=100)
            )
            void_value = float(
                rospy.get_param("~conversor/void_value", default=cutoff_distance)
            )
            if self.output_height is None:
                self.output_height = n_rays
            if self.output_width is None:
                self.output_width = 360
            self.conversor = conversors.PtclToAngleImageConversor(
                n_rays=n_rays, cutoff_distance=cutoff_distance, void_value=void_value
            )
        elif self.conversor_type == ConversorTypes.laserscan_to_cenital_img:
            self.input_topic_type = LaserScan
            void_value = float(rospy.get_param("~conversor/void_value", default=0))
            max_coord_val = float(
                rospy.get_param("~conversor/max_coord_val", default=30)
            )
            img_size = int(rospy.get_param("~conversor/img_size", default=30))
            if self.output_height is None:
                self.output_height = img_size
            if self.output_width is None:
                self.output_width = img_size
            self.conversor = conversors.ScanToCenithImage(
                max_coord_val, img_size, void_value
            )

    def setup_sub_pub(self):
        self._publisher = rospy.Publisher(self.output_topic, Image, queue_size=1)
        if self.normalize:
            self._norm_publisher = rospy.Publisher(
                self.output_topic + "_norm", Image, queue_size=1
            )
        self._bridge = CvBridge()
        self._subscriber = rospy.Subscriber(
            self.input_topic, self.input_topic_type, callback=self.callback
        )

    def callback(self, msg):
        # Create image
        image = self.conversor(msg)
        self.seq += 1
        stamp = rospy.Time.now()
        if image.shape[0] != self.output_height or image.shape[1] != self.output_width:
            image = cv2.resize(
                image,
                dsize=(self.output_width, self.output_height),
                interpolation=cv2.INTER_NEAREST,
            )
        img_msg = self._bridge.cv2_to_imgmsg(image.astype(np.float32), "32FC1")
        img_msg.header.stamp = stamp
        img_msg.header.seq = self.seq
        self._publisher.publish(img_msg)
        if self.normalize:
            norm_img = image / np.max(image)
            norm_img_msg = self._bridge.cv2_to_imgmsg(
                norm_img.astype(np.float32), "32FC1"
            )
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
