import matplotlib
import rospy
import math
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32


class DetectionMetricsNode:
    def __init__(self):
        rospy.init_node(str(self.__class__.__name__))
        detection_vector_topic = rospy.get_param(
            "~detection_vector_topic", default="/gallery_detection_vector"
        )
        rospy.Subscriber(detection_vector_topic, Float32MultiArray, callback=self.callback)
        self.std_publisher = rospy.Publisher("/detection_vector_metrics/std", Float32, queue_size=1)
        self.avg_publisher = rospy.Publisher("/detection_vector_metrics/avg", Float32, queue_size=1)
        self.avg_o_std_publisher = rospy.Publisher(
            "/detection_vector_metrics/avg_o_std", Float32, queue_size=1
        )
        self.std_o_avg_publisher = rospy.Publisher(
            "/detection_vector_metrics/std_o_avg", Float32, queue_size=1
        )

    def callback(self, msg: Float32MultiArray):
        data = np.array(msg.data).reshape(-1)
        avg = np.average(data)
        std = np.std(data)
        self.std_publisher.publish(std)
        self.avg_publisher.publish(avg)
        self.std_o_avg_publisher.publish(std / avg)
        self.avg_o_std_publisher.publish(avg / std)

    def run(self):
        rospy.spin()


def main():
    node = DetectionMetricsNode()
    node.run()


if __name__ == "__main__":
    main()
