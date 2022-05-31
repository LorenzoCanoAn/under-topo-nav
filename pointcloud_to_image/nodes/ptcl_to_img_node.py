#!/usr/bin/python3
from pointcloud_to_image import get_conversor_by_str
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
import cv2

# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------


class ConversionNode:
    def __init__(self):
        rospy.init_node("ptcl_to_img_node")
        self.seq = 0
        self.setup_conversor()
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
        image = cv2.resize(
            image,
            dsize=(self.image_width, self.image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        # Create image message
        image_msg = self._bridge.cv2_to_imgmsg(image, "32FC1")
        image_msg.header.seq = self.seq
        self.seq += 1
        image_msg.header.stamp = rospy.Time.now()

        # Send image message
        self._publisher.publish(image_msg)

    def run(self):
        rospy.spin()


def main():
    node = ConversionNode()
    node.run()


if __name__ == "__main__":
    main()
