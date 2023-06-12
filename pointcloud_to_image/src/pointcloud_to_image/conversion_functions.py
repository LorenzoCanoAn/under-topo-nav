#!/bin/python3
import rospy
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
import numpy as np


class PtclToHeightImageConversor:
    def __init__(self, Z_RES=100, Z_MAX=5):
        self.Z_RES = Z_RES
        self.Z_MAX = Z_MAX

    def __call__(self, msg: PointCloud2):
        point_array = (
            np.array(
                list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            )
            * 1
        )
        # Delete points with |z| > self.Z_MAX
        indices_to_delete = np.array(
            np.abs(point_array[:, 2]) >= self.Z_MAX, dtype=bool
        )
        point_array = np.delete(point_array, indices_to_delete, 0)
        # Calculate theta angles of each point, and correspoinding index in image
        theta_angles = 359 - (
            (np.arctan2(point_array[:, 1], point_array[:, 0]) * 180 / np.pi + 360 + 180)
            % 360
        ).astype(int)
        z = (
            self.Z_RES
            - (point_array[:, 2] + self.Z_MAX) / (2 * self.Z_MAX) * self.Z_RES
        ).astype(int)
        z = z % 99
        distances = np.sqrt((np.sum(np.square(point_array), 1)))
        distances /= np.max(distances)
        distances *= 255
        distances = distances.astype("uint8")
        image = np.zeros([self.Z_RES, 360]).astype("uint8")
        image[z, theta_angles] = distances
        return image


class PtclToAngleImageConversor:
    def __init__(self, n_rays=16, cutoff_distance=100, void_value=100):
        self.n_rays = n_rays
        self.cutoff_distance = cutoff_distance
        self.void_value = void_value

    def __call__(self, msg: PointCloud2):
        point_array = (
            np.array(
                list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            )
            * 1
        )
        theta_angles = np.arctan2(point_array[:, 1], point_array[:, 0])
        theta_angles_deg = np.rad2deg(theta_angles)
        # theta_angles_shifted = 359 - ((theta_angles_deg + 360 + 180) % 360)
        theta_angles_shifted = (theta_angles_deg + 360) % 360
        theta_angles_double = (theta_angles_shifted * 2).astype(int)
        distance_to_z_axis = np.linalg.norm(point_array[:, 0:2], axis=1)
        tangent = np.reshape(np.divide(point_array[:, 2], distance_to_z_axis), (-1, 1))
        delta_angles_deg = (
            (15 - (np.round(np.arctan(tangent) / np.math.pi * 180) + 15) / 2)
            .astype(int)
            .flatten()
        )
        distances = np.linalg.norm(point_array, axis=1).astype("float32")
        image = np.ones((self.n_rays, 360 * 2)).astype("float32") * self.void_value
        image[delta_angles_deg, theta_angles_double] = distances
        if self.cutoff_distance > 0:
            image = np.where(image > self.cutoff_distance, self.cutoff_distance, image)
        return image


class ScanToCenithImage:
    def __init__(self, max_coord_val, img_size, void_value):
        self.max_coord_val = max_coord_val
        self.img_size = img_size
        self.void_value = void_value
        self.scale = self.img_size / (self.max_coord_val * 2)

    def __call__(self, msg: LaserScan):
        ranges = np.reshape(np.array(msg.ranges), -1)
        n_ranges = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n_ranges)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        to_keep = np.logical_and(
            np.abs(x) < self.max_coord_val, np.abs(y) < self.max_coord_val
        )
        xy = np.vstack([x[to_keep], y[to_keep]]).T * self.scale
        i = (-xy[:, 0]).astype(int) + int(self.img_size / 2)
        j = (-xy[:, 1]).astype(int) + int(self.img_size / 2)
        img = np.ones((self.img_size, self.img_size)) * self.void_value
        img[i, j] = 1
        return img
