#!/usr/bin/python3
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Empty
import cv2
import numpy as np


# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------
class ScanToCenithImage:
    def __init__(self, max_coord_val, img_size, void_value):
        self.max_coord_val = max_coord_val
        self.img_size = img_size
        self.void_value = void_value
        self.scale = self.img_size / (self.max_coord_val * 2)

    def __call__(self, msg: LaserScan):
        ranges = np.reshape(np.array(msg.ranges), -1)
        n_ranges = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n_ranges)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        to_keep = np.logical_and(np.abs(x) < self.max_coord_val, np.abs(y) < self.max_coord_val)
        xy = np.vstack([x[to_keep], y[to_keep]]).T * self.scale
        i = (-xy[:, 0]).astype(int) + int(self.img_size / 2)
        j = (-xy[:, 1]).astype(int) + int(self.img_size / 2)
        img = np.ones((self.img_size, self.img_size)) * self.void_value
        img[i, j] = 1
        return img


class ScanToBEVImageNode:
    def __init__(self):
        rospy.init_node("scan_to_bev_image")
        self.get_params()
        self.setup_conversor()
        self.seq = 0
        self.setup_sub_pub()

    def get_params(self):
        self.input_topic = rospy.get_param("~input_topic", default="/velodyne_points")
        self.output_topic = rospy.get_param("~output_topic", default="/lidar_image")
        self.normalize = bool(int(rospy.get_param("~normalize", default=True)))

    def setup_conversor(self):
        void_value = float(rospy.get_param("~conversor/void_value", default=0))
        max_coord_val = float(rospy.get_param("~conversor/max_coord_val", default=30))
        img_size = int(rospy.get_param("~conversor/img_size", default=30))
        self.conversor = ScanToCenithImage(max_coord_val, img_size, void_value)

    def setup_sub_pub(self):
        self._publisher = rospy.Publisher(self.output_topic, Image, queue_size=1)
        if self.normalize:
            self._norm_publisher = rospy.Publisher(self.output_topic + "_norm", Image, queue_size=1)
        self._bridge = CvBridge()
        self._main_subscriber = rospy.Subscriber(self.input_topic, LaserScan, callback=self.callback)
        self._reconfig_conversor_sub = rospy.Subscriber("~reconfigure_conversor", Empty, callback=self.setup_conversor, queue_size=1)

    def callback(self, msg):
        # Create image
        image = self.conversor(msg)
        self.seq += 1
        stamp = rospy.Time.now()
        img_msg = self._bridge.cv2_to_imgmsg(image.astype(np.float32), "32FC1")
        img_msg.header.stamp = stamp
        img_msg.header.seq = self.seq
        self._publisher.publish(img_msg)
        if self.normalize:
            norm_img = image / np.max(image)
            norm_img_msg = self._bridge.cv2_to_imgmsg(norm_img.astype(np.float32), "32FC1")
            norm_img_msg.header.stamp = stamp
            norm_img_msg.header.seq = self.seq
            self._norm_publisher.publish(norm_img_msg)

    def run(self):
        rospy.spin()


def main():
    node = ScanToBEVImageNode()
    node.run()


if __name__ == "__main__":
    main()
