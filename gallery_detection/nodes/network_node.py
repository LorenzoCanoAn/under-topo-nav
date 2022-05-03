#!/bin/python3
import torch
from cv_bridge import CvBridge
from laserscan_image_nn.nn_definitions2 import gallery_detector_v4, gallery_detector_v4_1_small,gallery_detector_v3
import sensor_msgs.msg as sensor_msg
import std_msgs.msg as std_msg
import rospy
import cv2

# --------------------------------------------------------------
#       SIMPLE CLASS TO HANDLE ROS COMMUNICATIONS
# --------------------------------------------------------------

class NetworkNode:
    def __init__(self):
        rospy.init_node("gallery_network")

        self.init_network()

        self._cv_bridge = CvBridge()

        self.image_subscriber = rospy.Subscriber(
            "/lidar_image", sensor_msg.Image, self.image_callback
        )
        self.detection_publisher = rospy.Publisher(
            "/gallery_detection_vector", std_msg.Float32MultiArray, queue_size=10)

    def init_network(self):
        file = rospy.get_param("saved_nn_path",default="/home/lorenzo/catkin_data/models/gallery_detection_nn_well_saved/gallery_detector_v3_loss_MSELoss_lr_0.0001_N_8_refined_state_dict")
        self.model = gallery_detector_v3()
        self.model.load_state_dict(torch.load(file))
        self.model.eval()



    def image_callback(self, msg):
        depth_image = self._cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        depth_image = cv2.resize(depth_image, dsize=(16, 360), interpolation=cv2.INTER_CUBIC)
        depth_image_tensor = torch.tensor(depth_image).float().to(torch.device("cpu"))
        depth_image_tensor /= torch.max(depth_image_tensor)
        depth_image_tensor = torch.reshape(depth_image_tensor, [1, 1, -1, 720])*255

        data = self.model(depth_image_tensor)
        data = data.cpu().detach().numpy()
        data = data[0, :]

        dim = (std_msg.MultiArrayDimension("0",data.__len__(), 1),)

        layout = std_msg.MultiArrayLayout(dim, 0)

        output_message = std_msg.Float32MultiArray(
            layout,data.astype("float32"))
        print("publishing")
        self.detection_publisher.publish(output_message)
        print("published")


def main():
    network_node = NetworkNode()
    rospy.loginfo("Gallery network beguinning to spin")
    rospy.spin()


if __name__ == "__main__":
    main()
