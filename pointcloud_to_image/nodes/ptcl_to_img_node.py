#!/usr/bin/python3
from pointcloud_to_image import get_conversor_by_str
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import cv2
import numpy as np

# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------


class ConversionNode:
    def __init__(self):
        rospy.init_node("ptcl_to_img_node")
        self.seq = 0
        self.setup_conversor()
        self.normalize = rospy.get_param("~normalize", default=True)
        self._norm_publisher = rospy.Publisher("/lidar_image_norm", Image, queue_size=5)
        self._publisher = rospy.Publisher("/lidar_image", Image, queue_size=5)
        self._bridge = CvBridge()
        self._subscriber = rospy.Subscriber(
            "/velodyne_points", PointCloud2, callback=self.callback
        )

    def setup_conversor(self):
        """Modifies the parameters of the conversor if they are set in the ROS parameters"""
        self.conversor = get_conversor_by_str(
            rospy.get_param("image_type", default="angle")
        )()
        for key in self.conversor.__dict__.keys():
            setattr(
                self.conversor,
                key,
                rospy.get_param(key, default=self.conversor.__dict__[key]),
            )

    def callback(self, msg):
        # Create image
        c_nmp = f"{rospy.get_namespace()}/{rospy.get_name()}"
        self.image_width = rospy.get_param(f"ptcl_to_img/image_width", default=360)
        self.image_height = rospy.get_param(f"ptcl_to_img/image_height", default=16)
        image = self.conversor(msg)
        self.seq += 1
        stamp = rospy.Time.now()
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
