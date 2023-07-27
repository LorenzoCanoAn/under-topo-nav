#!/bin/python3
import rospy
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import math

MAX_ANG_VEL = 2
MAX_VEL = 5


class AngleToVelNode:
    def __init__(self) -> None:
        rospy.init_node("angle_to_twist")
        self.max_ang_vel = MAX_ANG_VEL
        self.max_vel = MAX_VEL
        self.publisher = rospy.Publisher("/cmd_vel", geometry_msgs.Twist, queue_size=1)
        self.input_topic = rospy.get_param("~input_topic")
        self.subscriber = rospy.Subscriber(
            self.input_topic,
            std_msgs.Float32,
            callback=self.angle_callback,
            queue_size=1,
        )
        self.change_vel_sub = rospy.Subscriber(
            "/tunnel_traversal/new_max_vel",
            std_msgs.Float32,
            callback=self.change_max_vel_callback,
            queue_size=1,
        )
        self.change_ang_vel_sub = rospy.Subscriber(
            "/tunnel_traversal/new_max_ang_vel",
            std_msgs.Float32,
            callback=self.change_max_ang_vel_callback,
            queue_size=1,
        )
        self.obstacle_sub = rospy.Subscriber(
            "/obstacle_detected",
            std_msgs.Bool,
            callback=self.obstacle_detected_callback,
            queue_size=1,
        )
        self.obstacle_detected = False

    def change_max_ang_vel_callback(self, msg: std_msgs.Float32):
        self.max_ang_vel = msg.data
        rospy.loginfo(f"Changed max ang vel to: {self.max_ang_vel}")

    def change_max_vel_callback(self, msg: std_msgs.Float32):
        self.max_vel = msg.data
        rospy.loginfo(f"Changed max vel to: {self.max_vel}")

    def obstacle_detected_callback(self, msg: std_msgs.Bool):
        self.obstacle_detected = msg.data

    def angle_to_speed(self, angle):
        if angle > math.pi:
            angle -= 2 * math.pi
        w = angle * 3
        if abs(w) > self.max_ang_vel:
            w = w / abs(w) * self.max_ang_vel
        v = (self.max_vel * (min(2, 2) / 2)) - abs(
            w
        ) / self.max_ang_vel * self.max_vel * 0.2
        v = max((v, 0))
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
