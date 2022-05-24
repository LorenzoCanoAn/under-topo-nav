#!/bin/python3
import rospy
import std_msgs.msg as std_msg
import nav_msgs.msg as nav_msg
import math
import numpy as np


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class GalleryTracker:
    def __init__(self):
        self.galleries = np.zeros(0)
        self.confidences = np.zeros(0)
        self.block_odom = True
        self.block_gallery = False
        self.max_confidence = 20

    def new_odom(self, turn):
        if self.block_odom:
            return
        self.galleries = (self.galleries - turn + 2 * math.pi) % (2 * math.pi)

    def new_galleries(self, angles, values):
        if self.block_gallery:
            return
        self.block_gallery = True
        self.block_odom = True
        base_confidences = values / max(values)
        angles = np.delete(angles, np.where(base_confidences < 0.2))
        base_confidences = np.delete(base_confidences, np.where(base_confidences < 0.2))
        distance_matrix = np.zeros((len(self.galleries), len(angles)))

        for i, gallery in enumerate(self.galleries):
            distances = (gallery - angles + 2 * math.pi) % (2 * math.pi)
            distances[distances > math.pi] = (
                2 * math.pi - distances[distances > math.pi]
            )
            distance_matrix[i, :] = distances
        unasigned_angles = list(angles)
        galleries_to_delete = []

        for i in range(len(self.galleries)):
            distances = distance_matrix[i, :]
            j = np.argmin(distances)
            min_distance = distances[j]
            gallery_re_observed = False

            if i == np.argmin(distance_matrix[:, j]):
                if min_distance < 10 / 180 * math.pi:
                    self.galleries[i] = angles[j]
                    self.confidences[i] = min(
                        base_confidences[j] + self.confidences[i], self.max_confidence
                    )
                    unasigned_angles.remove(angles[j])
                    gallery_re_observed = True

            if not gallery_re_observed:
                self.confidences[i] -= 2
                if self.confidences[i] < 0:
                    galleries_to_delete.append(i)

        galleries_to_delete.sort(reverse=True)
        for i in galleries_to_delete:
            self.galleries = np.delete(self.galleries, i)
            self.confidences = np.delete(self.confidences, i)

        for a in unasigned_angles:
            self.galleries = np.append(self.galleries, a)
            self.confidences = np.append(
                self.confidences, base_confidences[list(angles).index(a)]
            )

        # Delete galleries too close to each other
        gallery_was_deleted = True
        while gallery_was_deleted:
            gallery_was_deleted = False
            for gallery in self.galleries:
                distances = (self.galleries - gallery + 2 * math.pi) % (2 * math.pi)
                distances[distances > math.pi] = (
                    2 * math.pi - distances[distances > math.pi]
                )
                close_to_gallery = np.array(
                    np.where(distances < 20 / 180 * 2 * math.pi)
                ).flatten()
                if len(close_to_gallery) > 1:
                    dominant_gallery = np.argmax(self.confidences[close_to_gallery])
                    close_to_gallery = np.delete(close_to_gallery, dominant_gallery)
                    close_to_gallery = list(close_to_gallery)
                    close_to_gallery.sort(reverse=True)
                    for gal_to_del in close_to_gallery:
                        self.galleries = np.delete(self.galleries, gal_to_del)
                        self.confidences = np.delete(self.confidences, gal_to_del)
                    gallery_was_deleted = True

        self.block_odom = False
        self.block_gallery = False
        return self.galleries[self.confidences > self.max_confidence * 0.8]


class TrackingNode:
    def __init__(self):
        rospy.init_node("gallery_tracking_node")
        self.tracker = GalleryTracker()
        self.prev_z = 0
        self.gallery_subscriber = rospy.Subscriber(
            "/currently_detected_galleries",
            std_msg.Float32MultiArray,
            callback=self.currently_detected_callback,
        )
        self.odometry_subscriber = rospy.Subscriber(
            "/odometry/filtered",
            nav_msg.Odometry,
            callback=self.odometry_callback,
        )
        self.tracked_galleries_publisher = rospy.Publisher(
            "/tracked_galleries", std_msg.Float32MultiArray, queue_size=10
        )

    def currently_detected_callback(self, msg):
        assert msg.data.__len__() % 2 == 0
        reshaped = np.reshape(msg.data, (2, -1))
        galleries = self.tracker.new_galleries(reshaped[0, :], reshaped[1, :])
        if galleries is None:
            return
        data = galleries
        dim = (std_msg.MultiArrayDimension("0", data.__len__(), 2),)
        layout = std_msg.MultiArrayLayout(dim, 0)
        output_message = std_msg.Float32MultiArray(layout, data)
        self.tracked_galleries_publisher.publish(output_message)

    def odometry_callback(self, msg):

        q = msg.pose.pose.orientation
        x, y, z = euler_from_quaternion(q.x, q.y, q.z, q.w)
        turn = z - self.prev_z
        self.tracker.new_odom(turn)
        self.prev_z = z


def main():
    node = TrackingNode()
    rospy.spin()


if __name__ == "__main__":
    main()
