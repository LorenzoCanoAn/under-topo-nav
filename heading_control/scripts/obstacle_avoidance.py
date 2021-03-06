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
        rospy.Subscriber(
            "/followed_gallery", std_msgs.Float32, callback=self.angle_callback
        )
        self.block = False
        rospy.Subscriber("/scan", sensor_msgs.LaserScan,
                         callback=self.scanner_callback)
        self.laser_data = LaserData()

        self.angle_publisher = rospy.Publisher(
            "/corrected_angle", std_msgs.Float32, queue_size=10
        )

    def angle_callback(self, msg):
        if self.laser_data.angles is None:
            return

        objective_angle = msg.data
        angle_value_vector = np.zeros(self.laser_data.angles.__len__())
        for n, i in enumerate(self.laser_data.angles):
            angle_value_vector[n] = np.math.pi - \
                min_distance(i, objective_angle)
        angle_value_vector /= np.max(angle_value_vector)
        angle_value_vector = angle_value_vector
        total_value_vector = np.multiply(
            angle_value_vector, self.laser_data.filtered)
        max_idx = np.argmax(total_value_vector)
        final_angle = self.laser_data.angles[max_idx]
        self.angle_publisher.publish(final_angle)

    def scanner_callback(self, msg):
        scan_ranges = np.array(msg.ranges).flatten()
        scan_angles = np.linspace(
            msg.angle_min, msg.angle_max, len(scan_ranges)
        )
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
