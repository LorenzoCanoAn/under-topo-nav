#!/bin/python3
import rospy
import numpy as np
import sensor_msgs.msg as sensor_msgs
from functions import min_distance, filter_and_inflate_ranges, filter_vector
import std_msgs.msg as std_msgs


class LaserData:
    ranges = None
    angles = None
    filtered = None


class ObstacleAvoidanceNode:
    def __init__(self):
        self.passthrough = rospy.get_param("~passthrough", default=False)
        self.block = False
        self.laser_data = LaserData()
        self.angle_publisher = rospy.Publisher(
            "output_corrected_angle", std_msgs.Float32, queue_size=1
        )

        # PLOTTING TOPICS
        self._final_angle_value_publisher = rospy.Publisher(
            "output_final_weights", std_msgs.Float32MultiArray, queue_size=1
        )
        self._desired_angle_weight_publisher = rospy.Publisher(
            "output_desired_angle_weights", std_msgs.Float32MultiArray, queue_size=1
        )
        self._scanner_weight_publisher = rospy.Publisher(
            "output_laser_scan_weights", std_msgs.Float32MultiArray, queue_size=1
        )
        self._angles_publisher = rospy.Publisher(
            "output_angles", std_msgs.Float32MultiArray, queue_size=1
        )
        self._final_angle_value_message = std_msgs.Float32MultiArray()
        self._desired_angle_weight_message = std_msgs.Float32MultiArray()
        self._scanner_weight_message = std_msgs.Float32MultiArray()
        self._angles_message = std_msgs.Float32MultiArray()
        rospy.Subscriber(
            "input_desired_angle",
            std_msgs.Float32,
            callback=self.angle_callback,
            queue_size=1,
        )
        rospy.Subscriber("input_scan", sensor_msgs.LaserScan, callback=self.scanner_callback)

    def angle_callback(self, msg):
        if self.passthrough:
            self.angle_publisher.publish(msg)
            return
        if self.laser_data.angles is None:
            return
        objective_angle = msg.data
        desired_angle_weight = np.zeros(self.laser_data.angles.__len__())
        for n, i in enumerate(self.laser_data.angles):
            desired_angle_weight[n] = np.math.pi - min_distance(i, objective_angle)
        desired_angle_weight /= np.max(desired_angle_weight)
        desired_angle_weight = desired_angle_weight
        final_angle_value_vector = np.multiply(
            desired_angle_weight, self.laser_data.filtered
        )
        max_idx = np.argmax(final_angle_value_vector)
        final_angle = self.laser_data.angles[max_idx]
        self.angle_publisher.publish(final_angle)

        self._final_angle_value_message.data = final_angle_value_vector
        self._desired_angle_weight_message.data = desired_angle_weight
        self._scanner_weight_message.data = self.laser_data.filtered
        self._angles_message.data = self.laser_data.angles

        self._final_angle_value_publisher.publish(self._final_angle_value_message)
        self._desired_angle_weight_publisher.publish(self._desired_angle_weight_message)
        self._scanner_weight_publisher.publish(self._scanner_weight_message)
        self._angles_publisher.publish(self._angles_message)

    def scanner_callback(self, msg):
        scan_ranges = np.array(msg.ranges).flatten()
        scan_angles = np.linspace(msg.angle_min, msg.angle_max, len(scan_ranges))
        self.laser_data.angles = scan_angles
        self.laser_data.ranges = scan_ranges
        self.laser_data.filtered = filter_and_inflate_ranges(
            scan_ranges, msg.angle_increment
        )


def main():
    rospy.init_node("obstacle_avoidance")
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
