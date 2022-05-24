#!/bin/python3
import rospy
import std_msgs.msg as std_msg
from time import time_ns as ns


def secs():
    return ns() / 1e9


class HeadingControlNode:
    def __init__(self):
        rospy.init_node("heading_control_node")
        self.state = "first_callback"
        self.galleries = []
        self.galleries_subscriber = rospy.Subscriber(
            "/tracked_galleries",
            std_msg.Float32MultiArray,
            callback=self.tracked_galleries_callback,
        )

    def tracked_galleries_callback(self, msg):
        galleries = list(msg.data)
        galleries.sort(reverse=True)

        if self.state == "first_callback":
            self.timer = ns()
            self.state = "setting_up"
        elif self.state == "setting_up":
            print("setting up")
            self.galleries = galleries
            if secs() - self.timer > 3:
                self.instructions = rospy.get_param(
                    "/topological_instructions", default=None
                )
                if not self.instructions is None:
                    self.state == "in transition"
                    self.transition_timer = secs()
        elif self.state == "transition":
            pass


def main():
    heading_control_node = HeadingControlNode()
    rospy.spin()


if __name__ == "__main__":
    main()
