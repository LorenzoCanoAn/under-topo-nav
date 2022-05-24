#!/bin/python3
import rospy
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import math

MAX_ANG_VEL = 0.2
MAX_VEL = 0.4
MIN_DIST = 1


def angle_to_speed(angle):
    if angle > math.pi:
        angle -= 2 * math.pi
    w = angle
    if abs(w) > MAX_ANG_VEL:
        w = w / abs(w) * MAX_ANG_VEL
    v = (MAX_VEL * (min(MIN_DIST, 2) / 2)) - abs(w) / MAX_ANG_VEL * MAX_VEL
    v = max((v, 0))
    print((v, w))
    return v, w


def angle_callback(msg, publisher):
    publisher = publisher[0]
    angle = msg.data
    v, w = angle_to_speed(angle)
    out_msg = geometry_msgs.Twist()
    out_msg.angular.z = w
    out_msg.linear.x = v
    publisher.publish(out_msg)


def main():
    rospy.init_node("angle_to_twist")
    publisher = rospy.Publisher("/cmd_vel", geometry_msgs.Twist, queue_size=10)
    subscriber = rospy.Subscriber(
        "/corrected_angle",
        std_msgs.Float32,
        callback=angle_callback,
        callback_args=[publisher],
    )
    rospy.spin()


if __name__ == "__main__":
    main()
