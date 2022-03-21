#!/bin/python3
import torch
from cv_bridge import CvBridge
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy


# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------

class NetworkNode:
    def __init__(self):
        rospy.init_node("gallery_network")

        self.init_network()

        self._cv_bridge = CvBridge()

        self.image_subscriber = rospy.Subscriber(
            "/laserscan_image", sensor_msg.Image, self.image_callback
        )
        self.detection_publisher = rospy.Publisher(
            "/gallery_detection_vector", std_msg.Float32MultiArray, queue_size=10)

    def init_network(self):
        try:
            nn_file = rospy.get_param("saved_nn_path")
        except KeyError:
            rospy.logerr("'saved_nn_path' parameter must be set")
            exit()

        self.model = torch.jit.load(nn_file).float()
        self.model.eval()

    def image_callback(self, msg):
        depth_image = self._cv_bridge.imgmsg_to_cv2(msg, "mono8")

        depth_image_tensor = torch.tensor(depth_image).float()
        depth_image_tensor /= torch.max(depth_image_tensor)
        depth_image_tensor = torch.reshape(depth_image_tensor, [1, 1, -1, 360])

        output = self.model(depth_image_tensor)
        output = output.cpu().detach().numpy()
        output = output[0, :]

        dim = std_msg.MultiArrayDimension("0",output.__len__(), 1)

        layout = std_msg.MultiArrayLayout(dim, 0)

        output_message = std_msg.Float32MultiArray(
            output.astype("float32"), layout)

        self.detection_publisher.publish(output_message)


def main():
    network_node = NetworkNode()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()
