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
        detected_galleries_topic = rospy.get_param(
            "~detected_galleries_topic", default="/currently_detected_galleries"
        )
        self.diff_buffer_length = rospy.get_param("~diff_buffer_length", 10)
        rospy.Subscriber(
            detection_vector_topic, Float32MultiArray, callback=self.detection_vector_callback
        )
        rospy.Subscriber(
            detected_galleries_topic, Float32MultiArray, callback=self.detected_galleries_callback
        )
        self.output_publisher = rospy.Publisher("/avg_over_n_gal", Float32, queue_size=10)
        self.output_publisher_1 = rospy.Publisher("/n_gal", Float32, queue_size=10)
        self.output_publisher_2 = rospy.Publisher("/diff_of_avg_over_n_gal", Float32, queue_size=10)
        self.output_publisher_3 = rospy.Publisher("/avg_of_diff", Float32, queue_size=10)
        self.detection_vector = None
        self.detected_galleries = None
        self.prev_detection_vector = None
        self.diff_buffer = []

    def detection_vector_callback(self, msg: Float32MultiArray):
        self.prev_detection_vector = self.detection_vector
        dv = np.array(msg.data).reshape(-1)
        self.detection_vector = dv / np.max(dv)

    def detected_galleries_callback(self, msg: Float32MultiArray):
        if not self.detected_galleries is None:
            self.prev_detected_galleries = self.detected_galleries
        self.detected_galleries = np.array(msg.data).reshape(-1)
        if hasattr(self, "metric"):
            self.prev_metric = self.metric
        self.metric = np.sum(np.sqrt(self.detection_vector)) / len(self.detected_galleries)
        self.output_publisher.publish(self.metric)
        self.output_publisher_1.publish(len(self.detected_galleries))
        if hasattr(self, "prev_metric"):
            diff = abs(self.metric - self.prev_metric)
            self.diff_buffer.append(diff)
            if len(self.diff_buffer) > self.diff_buffer_length:
                self.diff_buffer.pop(0)
            self.output_publisher_2.publish(diff)
            self.output_publisher_3.publish(np.max(np.array(self.diff_buffer)))

    def run(self):
        rospy.spin()


def main():
    node = DetectionMetricsNode()
    node.run()


if __name__ == "__main__":
    main()
