#!/bin/python3
import rospy
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import math

MAX_ANG_VEL = 0.4
MAX_VEL = 1
MIN_DIST = 1


class AngleToVelNode:
    def __init__(self) -> None:
        rospy.init_node("angle_to_twist")
        self.publisher = rospy.Publisher("/cmd_vel", geometry_msgs.Twist, queue_size=10)
        self.subscriber = rospy.Subscriber(
            "/estimated_relative_yaw",
            std_msgs.Float32,
            callback=self.angle_callback,
        )
        self.obstacle_sub = rospy.Subscriber(
            "obstacle_detected", std_msgs.Bool, callback=self.obstacle_detected_callback
        )
        self.obstacle_detected = False

    def obstacle_detected_callback(self, msg: std_msgs.Bool):
        self.obstacle_detected = msg.data

    def angle_to_speed(self, angle):
        if angle > math.pi:
            angle -= 2 * math.pi
        w = -angle
        if abs(w) > MAX_ANG_VEL:
            w = w / abs(w) * MAX_ANG_VEL
        v = (MAX_VEL * (min(2, 2) / 2)) - abs(w) / MAX_ANG_VEL * MAX_VEL * 0.2
        v = max((v, 0))
        print((v, w))
        return v, w

    def angle_callback(self, msg):
        if self.obstacle_detected:
            self.publisher.publish(geometry_msgs.Twist())
        else:
            angle = msg.data
            v, w = self.angle_to_speed(angle)
            out_msg = geometry_msgs.Twist()
            out_msg.angular.z = w
            out_msg.linear.x = v
            self.publisher.publish(out_msg)


def main():
    node = AngleToVelNode()
    rospy.spin()


if __name__ == "__main__":
    main()
