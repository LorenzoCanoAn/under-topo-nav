#!/bin/python3
import threading
import rospy
import std_msgs.msg as std_msg
import numpy as np
import math
from datetime import datetime
import time


def min_dis(angles, angle):
    distances = np.abs((angles - angle))
    distances[distances > math.pi] = 2 * \
        math.pi - distances[distances > math.pi]
    return distances


class HeadingControlNode:
    def __init__(self):
        rospy.init_node("heading_control_node")
        self.state = "Init"
        self.change_state("waiting_for_instructions")
        self.galleries = None
        self.prev_galleries = None
        self.back_gallery = None
        self.followed_gallery = None
        self.changed_gallery_number = False
        self.galleries_subscriber = rospy.Subscriber(
            "/tracked_galleries",
            std_msg.Float32MultiArray,
            callback=self.tracked_galleries_callback,
        )

        self.followed_galery_publisher = rospy.Publisher(
            "/followed_gallery", std_msg.Float32, queue_size=5
        )
        self.back_gallery_publisher = rospy.Publisher(
            "/back_gallery", std_msg.Float32, queue_size=5
        )

        self.thread = threading.Thread(target=self.state_machine_thread)
        self.thread.start()
        self.thread.join()

    def tracked_galleries_callback(self, msg):
        angles, confidences = np.split(np.array(msg.data), 2)
        new_galleries = list(angles[confidences > 20*0.8])

        new_galleries.sort()
        if self.galleries is None:
            self.galleries = new_galleries
            self.prev_galleries = new_galleries
        else:
            self.prev_galleries = self.galleries
            self.galleries = new_galleries
        if len(self.galleries) != len(self.prev_galleries):
            self.changed_gallery_number = True
        self.update_back_gallery()
        self.update_followed_gallery()

    def state_machine_thread(self):
        start = datetime.now()
        while not rospy.is_shutdown():
            diff = (datetime.now() - start).seconds
            if diff < 0.1:
                time.sleep(0.1 - diff)
            start = datetime.now()

            if self.state == "waiting_for_instructions":
                self.followed_gallery = None
                self.instructions = rospy.get_param(
                    "/topological_instructions", default=None
                )
                if not self.instructions is None:
                    try:
                        if len(self.instructions) > 0:
                            self.change_state("instructions_recieved")
                            self.curr_inst = 0
                    except:
                        print(
                            "The topological instructions parameter must be a list!")

            elif self.state == "instructions_recieved":
                self.changed_gallery_number = False
                self.back_gallery = None
                self.choose_gallery_to_follow()
                self.transition_counter = 0

            elif self.state == "in_transition":
                if self.changed_gallery_number:
                    self.transition_counter = 0
                    self.changed_gallery_number = False
                self.transition_counter += 1
                if self.transition_counter == 20:
                    self.curr_inst += 1
                    if self.curr_inst >= len(self.instructions):
                        self.change_state("finished")
                    else:
                        self.choose_gallery_to_follow()

            elif self.state == "following_gallery":
                if self.changed_gallery_number:
                    self.change_state("in_transition")
            elif self.state == "finished":
                rospy.set_param("/topological_instructions", [])
                self.change_state("waiting_for_instructions")
            if not self.followed_gallery is None:
                self.followed_galery_publisher.publish(self.followed_gallery)

            if not self.back_gallery_publisher is None:
                self.back_gallery_publisher.publish(self.back_gallery)

    def choose_gallery_to_follow(self):
        if self.update_back_gallery():
            self.followed_gallery = self.galleries[
                (self.back_gallery_idx() + self.instructions[self.curr_inst])
                % len(self.galleries)
            ]
            self.change_state("following_gallery")
        else:
            return False

    def change_state(self, new_state):
        print(f"Changing state from {self.state} to {new_state}")
        self.previous_state = self.state
        self.state = new_state

    def update_back_gallery(self):
        if len(self.galleries) == 0:
            return False
        galleries = np.array(self.galleries)
        if self.back_gallery is None:
            distances = min_dis(galleries, math.pi)
            new_back = galleries[np.argmin(distances)]
            self.back_gallery = new_back
        else:
            distances = min_dis(galleries, self.back_gallery)
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] > (20 / 180 * math.pi):
                self.back_gallery = None
                self.update_back_gallery()
            else:
                self.back_gallery = galleries[min_dist_idx]
        return True

    def update_followed_gallery(self):
        if self.followed_gallery is None:
            return
        galleries = np.array(self.galleries)
        distances = min_dis(galleries, self.followed_gallery)
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] > (15 / 180 * math.pi):
            self.followed_gallery = (
                self.galleries[self.back_gallery_idx()] + math.pi + math.pi * 2
            ) % (2 * math.pi)
        else:
            self.followed_gallery = galleries[min_dist_idx]

    def back_gallery_idx(self):
        return np.argmin(min_dis(self.galleries, self.back_gallery))


def main():
    rospy.sleep(5)
    heading_control_node = HeadingControlNode()
    rospy.spin()


if __name__ == "__main__":
    main()
