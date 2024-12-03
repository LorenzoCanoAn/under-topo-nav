#!/bin/python3
import rospy
import nav_msgs.msg as nav_msg
from gallery_tracking.msg import TrackedGalleries
from gallery_detection_ros.msg import DetectedGalleries
import math
import numpy as np
from scipy.spatial.transform import Rotation


def euler_from_quaternion(quat):
    return Rotation.from_quat(quat).as_euler("xyz")  # in radians


def get_distances_to_angle(angle: float, angles: np.ndarray):
    distances = (angles - angle) % (2 * np.pi)
    indexer = np.where(distances > np.pi)
    distances[indexer] = 2 * np.pi - distances[indexer]
    return distances


GALLERY_NOT_TRACKED = 0
GALLERY_TRACKED = 1
GALLERY_LOST = 2
GALLERY_TO_REMOVE = 3


class Queue:
    """This is a generic data storage class that keeps a certain number of elements. If the number of elements is 
    equal to it's length, is new element is added, but the oldest element is removed"""
    def __init__(self, length):
        self.data = list()
        self.length = length

    def add_data(self, data):
        if len(self.data) < self.length:
            self.data.append(data)
        else:
            self.data.pop(0)
            self.data.append(data)

    def mean(self):
        return np.mean(np.array(self.data))

    def max(self):
        return np.max(np.array(self.data))


class Gallery:
    seen_to_set_id_threshold: int = 10
    unseen_to_remove_id_threshold: int = 2
    angle_threshold = np.deg2rad(10)
    value_threshold_to_detect = 0.7
    value_threshold_to_remove = 0.4
    id_counter = 0
    history_length = 10

    @classmethod
    def update_angle_threshold(cls, new_angle_threshold):
        cls.angle_threshold = new_angle_threshold

    @classmethod
    def update_counter_threshold(cls, new_counter_threshold):
        cls.seen_to_set_id_threshold = new_counter_threshold

    @classmethod
    def increase_counter(cls):
        cls.id_counter += int(1)

    @classmethod
    def assign_id(cls):
        cls.increase_counter()
        return int(cls.id_counter)

    def __init__(self, angle, value):
        self.id = None
        self.seen_counter = 0  # Keep track of how many times it has been seen
        self.unseen_counter = 0  # Keep track of how many times it has not been seen
        self.state = GALLERY_NOT_TRACKED
        self.angle = angle
        self.value_history = Queue(self.history_length)

    def new_galleries(self, angles: np.ndarray, values: np.ndarray):
        
        assert len(angles) == len(values)
        if len(angles) == 0:
            min_distance = np.pi
        else:
            distances = get_distances_to_angle(self.angle, angles)
            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
        if self.state == GALLERY_NOT_TRACKED:  # If the gallery has not started to be tracked
            if min_distance < self.angle_threshold:  # I has been detected again
                self.angle = angles[min_distance_idx]
                self.value_history.add_data(values[min_distance_idx])
                self.seen_counter += 1
                if self.seen_counter > self.seen_to_set_id_threshold and self.value_history.mean() > self.value_threshold_to_detect:
                    self.state = GALLERY_TRACKED
                    self.id = self.assign_id()
                return min_distance_idx
        elif self.state == GALLERY_TRACKED:
            if min_distance < self.angle_threshold and values[min_distance_idx] > self.value_threshold_to_remove:  # I has been detected again
                self.value_history.add_data(values[min_distance_idx])
                self.angle = angles[min_distance_idx]
                return min_distance_idx
            else:
                self.unseen_counter += 1
                if self.unseen_counter >= self.unseen_to_remove_id_threshold:
                    print(f"Gallery {self.id} set to lost")
                    self.state = GALLERY_LOST
        elif self.state == GALLERY_LOST:
            if min_distance < self.angle_threshold:  # I has been detected again
                self.angle = angles[min_distance_idx]
                self.state = GALLERY_TRACKED
                self.unseen_counter = 0
                return min_distance_idx
            else:
                self.unseen_counter += 1
                if self.unseen_counter >= self.unseen_to_remove_id_threshold * 2:
                    self.state = GALLERY_LOST

    def new_odometry(self, odometry_turn: float):
        self.angle = (self.angle - odometry_turn) % (np.pi * 2)


class GalleryTracker:
    def __init__(self):
        self.galleries: list[Gallery] = []
        self.back_gallery_id = -1

    def key(self, element: Gallery):
        return element.id

    @property
    def tracked_galleries(self):
        return [gallery for gallery in self.galleries if gallery.state == GALLERY_TRACKED]

    @property
    def lost_galleries(self):
        return [gallery for gallery in self.galleries if gallery.state == GALLERY_LOST]

    @property
    def non_tracked_galleries(self):
        return [gallery for gallery in self.galleries if gallery.state == GALLERY_NOT_TRACKED]

    @property
    def sorted_galleries(self):
        temp = self.tracked_galleries
        temp.sort(key=self.key, reverse=False)
        return temp + self.lost_galleries + self.non_tracked_galleries

    def new_galleries(self, angles, values):
        new_angles = angles
        new_values = values
        for gallery in self.sorted_galleries:
            sel_ang_id = gallery.new_galleries(new_angles, new_values)
            if sel_ang_id is False:
                pass
            elif not sel_ang_id is None:
                new_angles = np.delete(new_angles, sel_ang_id)
                new_values = np.delete(new_values, sel_ang_id)
            else:
                if gallery.state == GALLERY_TO_REMOVE:
                    self.galleries.remove(gallery)

        for new_angle, new_value in zip(new_angles, new_values):
            self.galleries.append(Gallery(new_angle, new_value))
        # Get back gallery
        if len(self.tracked_galleries) > 0:
            ids = [g.id for g in self.tracked_galleries]
            dsts = np.array([g.angle for g in self.tracked_galleries])
            bgidx = np.argmin(get_distances_to_angle(np.pi, dsts))
            self.back_gallery_id = ids[bgidx]

    def angles(self):
        return [gallery.angle for gallery in self.tracked_galleries]

    def ids(self):
        return [gallery.id for gallery in self.tracked_galleries]

    def new_odom(self, turn):
        for gal in self.galleries:
            gal.new_odometry(turn)


class TrackingNode:
    def __init__(self):
        rospy.init_node("gallery_tracking_node")
        Gallery.angle_threshold = np.deg2rad(rospy.get_param("~threshold_deg", 50))
        Gallery.seen_to_set_id_threshold = rospy.get_param("~counter_threshold", 10) # Number of times a gallery has to be seen to be considered as Tracked
        Gallery.value_threshold_to_detect = rospy.get_param("~value_threshold_to_detect", 0.5) # Value the peak of the gallery detection must reach
        Gallery.value_threshold_to_remove = rospy.get_param("~value_threshold_to_remove", 0.4)
        print(np.rad2deg(Gallery.angle_threshold))
        self.prev_z = 0
        self.tracker = GalleryTracker()
        self.back_gallery = None
        self.gallery_subscriber = rospy.Subscriber(
            "input_galleries_topic",
            DetectedGalleries,
            callback=self.currently_detected_callback,
        )
        self.odometry_subscriber = rospy.Subscriber(
            "input_odometry_topic",
            nav_msg.Odometry,
            callback=self.odometry_callback,
        )
        self.tracked_galleries_publisher = rospy.Publisher("output_tracked_galleries_topic", TrackedGalleries, queue_size=10)

    def currently_detected_callback(self, msg: DetectedGalleries):
        angles = np.array(msg.angles)
        values = np.array(msg.values)
        self.tracker.new_galleries(angles, values)
        output_message = TrackedGalleries()
        output_message.header = msg.header
        output_message.angles = self.tracker.angles()
        output_message.ids = self.tracker.ids()
        output_message.back_gallery = self.tracker.back_gallery_id
        self.tracked_galleries_publisher.publish(output_message)

    def odometry_callback(self, msg: nav_msg.Odometry):
        q = msg.pose.pose.orientation
        x, y, z = euler_from_quaternion(np.array((q.x, q.y, q.z, q.w)))
        if hasattr(self, "prev_z"):
            turn = z - self.prev_z
            self.tracker.new_odom(turn)
        self.prev_z = z


def main():
    node = TrackingNode()
    rospy.spin()


if __name__ == "__main__":
    main()
