import numpy as np
import os
import argparse
import rospy
from scipy.stats import norm
import trimesh
import matplotlib.pyplot as plt
from gazebo_msgs.srv import (
    SpawnModel,
    SpawnModelRequest,
    SpawnModelResponse,
    SetModelState,
    SetModelStateRequest,
    SetModelStateResponse,
    DeleteModel,
    DeleteModelRequest,
    DeleteModelResponse,
)
import geometry_msgs
import sensor_msgs
from cv_bridge import CvBridge
import time
import pyvista as pv

gaussian_witdth = np.arange(-3, 3, 0.1)
GAUSSIAN = norm.pdf(gaussian_witdth, 0, 1)
GAUSSIAN = GAUSSIAN / max(GAUSSIAN)


def get_transformation_matrix(x, y, z, roll, pitch, yaw):
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    R_x = np.array([[1, 0, 0], [0, c_roll, -s_roll], [0, s_roll, c_roll]])
    R_y = np.array([[c_pitch, 0, s_pitch], [0, 1, 0], [-s_pitch, 0, c_pitch]])
    R_z = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])

    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    translation_vector = np.array([x, y, z])

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qx, qy, qz, qw


class AxisPointManager:
    def __init__(self, axis_points, axis_vectors, voxel_size=5):
        self.axis_points = axis_points
        self.axis_vectors = axis_vectors
        self.voxel_size = voxel_size
        self.grid = dict()
        max_x = max(axis_points[:, 0])
        min_x = min(axis_points[:, 0])
        max_y = max(axis_points[:, 1])
        min_y = min(axis_points[:, 1])
        max_z = max(axis_points[:, 2])
        min_z = min(axis_points[:, 2])
        max_i = int(np.ceil(max_x / voxel_size)) + 3
        min_i = int(np.floor(min_x / voxel_size)) - 3
        max_j = int(np.ceil(max_y / voxel_size)) + 3
        min_j = int(np.floor(min_y / voxel_size)) - 3
        max_k = int(np.ceil(max_z / voxel_size)) + 3
        min_k = int(np.floor(min_z / voxel_size)) - 3
        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                for k in range(min_k, max_k):
                    self.grid[(i, j, k)] = np.zeros([0, 6])
        ijks = np.floor(self.axis_points / voxel_size).astype(int)
        for ijk, ap, av in zip(ijks, self.axis_points, self.axis_vectors):
            i, j, k = ijk
            ap = np.reshape(ap, (-1, 3))
            av = np.reshape(av, (-1, 3))
            self.grid[(i, j, k)] = np.concatenate(
                [self.grid[(i, j, k)], np.concatenate((ap, av), axis=1)], axis=0
            )

    def get_relevant_points(self, xyz):
        _i, _j, _k = np.floor(xyz / self.voxel_size).astype(int)
        relevant_points = np.zeros((0, 6))
        for i in (_i - 1, _i, _i + 1):
            for j in (_j - 1, _j, _j + 1):
                for k in (_k - 1, _k, _k + 1):
                    relevant_points = np.concatenate(
                        [relevant_points, self.grid[(i, j, k)]], axis=0
                    )
        return relevant_points


class YawDetectionPoseAndLabelGenerator:
    def __init__(
        self,
        axis_points,
        axis_vectors,
        max_hor_disp,
        max_vert_disp,
        min_vert_disp,
        max_inclination_rad,
        label_range_rad,
    ):
        self.axis_points = axis_points
        self.axis_vectors = axis_vectors
        self.max_hor_disp = max_hor_disp
        self.max_vert_disp = max_vert_disp
        self.min_vert_disp = min_vert_disp
        self.max_inclination_rad = max_inclination_rad
        self.label_range_rad = label_range_rad

    def gen_one_sample(self):
        base_point_idx = np.random.randint(0, len(self.axis_points))
        ap = np.reshape(self.axis_points[base_point_idx, :], (1, 3))
        av = np.reshape(self.axis_vectors[base_point_idx, :], (1, 3))
        dist_to_axis = self.max_hor_disp * np.random.uniform(-1, 1)
        axis_theta = np.arctan2(av[0, 1], av[0, 0])
        perp_axis_theta = axis_theta + np.pi / 2
        x_disp = dist_to_axis * np.cos(perp_axis_theta)
        y_disp = dist_to_axis * np.sin(perp_axis_theta)
        z_disp = np.random.uniform(self.min_vert_disp, self.max_vert_disp)
        label = np.random.uniform(-self.label_range_rad, self.label_range_rad)
        pose = ap + np.reshape(np.array([x_disp, y_disp, z_disp]), (1, 3))
        pose = np.reshape(pose, -1)
        roll = np.random.uniform(-self.max_inclination_rad, self.max_inclination_rad)
        pitch = np.random.uniform(-self.max_inclination_rad, self.max_inclination_rad)
        yaw = axis_theta + label
        return pose, roll, pitch, yaw, label

    def gen_n_samples(self, n_samples):
        poses = np.zeros((n_samples, 3))
        rolls = np.zeros((n_samples, 1))
        pitches = np.zeros((n_samples, 1))
        yaws = np.zeros((n_samples, 1))
        labels = np.zeros((n_samples, 1))
        j = 0
        for i in range(n_samples):
            pose, roll, pitch, yaw, label = self.gen_one_sample()
            poses[i, :] = pose
            rolls[i, :] = roll
            pitches[i, :] = pitch
            yaws[i, :] = yaw
            labels[i, :] = label
        return poses, rolls, pitches, yaws, labels


def get_args():
    parser = argparse.ArgumentParser(
        prog="Dataset collector", description="Given a dataset folder structure"
    )
    parser.add_argument("-F", "--dataset_folder", required=True, type=str)
    parser.add_argument("-N", "--number_of_samples_per_env", required=True, type=int)
    parser.add_argument("-O", "--overwrite", required=False, type=bool, default=False)
    parser.add_argument("-YR", "--label_range_deg", required=True, type=float)
    parser.add_argument(
        "-MH", "--max_horizontal_displacement", required=True, type=float
    )
    parser.add_argument("-MV", "--max_vertical_displacement", default=0, type=float)
    parser.add_argument("-_MV", "--min_vertical_displacement", default=0, type=float)
    parser.add_argument("-MI", "--max_inclination_deg", default=10, type=float)
    return parser.parse_args()


class ImageStorage:
    def __init__(self, image_topic):
        self._sub = rospy.Subscriber(
            image_topic, sensor_msgs.msg.Image, callback=self.callback
        )
        self._switch = True
        self._brdg = CvBridge()

    def callback(self, msg):
        self.image = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height,
            msg.width,
        )
        self._switch = False

    def block(self):
        self._switch = True
        while self._switch:
            time.sleep(0.2)


class DatasetRecorder:
    def __init__(self):
        self.counter = 0
        rospy.init_node("dataset_collector")
        self.robot_name = rospy.get_param("~robot_name", default="/")
        self.image_topic = rospy.get_param("~data_topic", default="/cenital_image")
        self.image_storage = ImageStorage(self.image_topic)
        self.set_pose_srv_proxy = rospy.ServiceProxy(
            "/gazebo/set_model_state", SetModelState, persistent=True
        )
        self.spawn_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/spawn_sdf_model", SpawnModel, persistent=True
        )
        self.delete_model_srv_proxy = rospy.ServiceProxy(
            "/gazebo/delete_model", DeleteModel
        )

    def send_position(self, x, y, z, roll, pitch, yaw):
        position = geometry_msgs.msg.Point(x, y, z)
        qx, qy, qz, qw = get_quaternion_from_euler(roll, pitch, yaw)
        orientation = geometry_msgs.msg.Quaternion(qx, qy, qz, qw)
        pose = geometry_msgs.msg.Pose(position, orientation)
        twist = geometry_msgs.msg.Twist(
            geometry_msgs.msg.Vector3(0, 0, 0), geometry_msgs.msg.Vector3(0, 0, 0)
        )
        request = SetModelStateRequest()
        request.model_state.model_name = "/"
        request.model_state.pose = pose
        request.model_state.twist = twist
        request.model_state.reference_frame = ""
        response = self.set_pose_srv_proxy(request)

    def record_dataset(self, folder, poses, rolls, pitches, yaws, labels):
        for i, (pose, roll, pitch, yaw, label) in enumerate(
            zip(poses, rolls, pitches, yaws, labels)
        ):
            print(f"{i:05d}", end="\r", flush=True)
            x, y, z = pose
            self.send_position(x, y, z, roll, pitch, yaw)
            self.image_storage.block()
            self.image_storage.block()
            image = self.image_storage.image
            np.savez(os.path.join(folder, f"{i}.npz"), image=image, label=label)

    def change_environment(self, file_to_model):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = DeleteModelRequest()
        delete_request.model_name = "cave"
        self.delete_model_srv_proxy.call(delete_request)
        time.sleep(2)
        spawn_request = SpawnModelRequest()
        spawn_request.model_name = "cave"
        spawn_request.model_xml = model_text
        spawn_request.reference_frame = ""
        self.spawn_model_srv_proxy.call(spawn_request)


def main():
    args = get_args()
    dataset_folder = args.dataset_folder
    n_samples = args.number_of_samples_per_env
    max_horizontal_displacement = args.max_horizontal_displacement
    max_vertical_displacement = args.max_vertical_displacement
    min_vertical_displacement = args.min_vertical_displacement
    max_inclination_rad = np.deg2rad(args.max_inclination_deg)
    label_range_rad = np.deg2rad(args.label_range_deg)
    dataset_recorder = DatasetRecorder()
    env_folders = os.listdir(dataset_folder)
    env_folders.sort()
    for env_folder in env_folders:
        print(env_folder)
        abs_env_folder = os.path.join(dataset_folder, env_folder)
        axis_file = os.path.join(abs_env_folder, "axis.txt")
        model_file = os.path.join(abs_env_folder, "model.sdf")
        fta_dist_file = os.path.join(abs_env_folder, "fta_dist.txt")
        dataset_recorder.change_environment(model_file)
        axis_data = np.loadtxt(axis_file)
        axis_points = axis_data[:, :3]
        axis_vectors = axis_data[:, 3:6]
        fta_dist = np.loadtxt(fta_dist_file).item(0)
        pose_and_label_generator = YawDetectionPoseAndLabelGenerator(
            axis_points=axis_points,
            axis_vectors=axis_vectors,
            max_hor_disp=max_horizontal_displacement,
            max_vert_disp=max_vertical_displacement + fta_dist,
            min_vert_disp=min_vertical_displacement + fta_dist,
            max_inclination_rad=max_inclination_rad,
            label_range_rad=label_range_rad,
        )
        poses, rolls, pitches, yaws, labels = pose_and_label_generator.gen_n_samples(
            n_samples
        )
        save_folder = os.path.join(abs_env_folder, "data")
        if os.path.exists(save_folder):
            if args.overwrite:
                pass
            else:
                raise Exception("The Dataset is going to be overwriten")
        else:
            os.mkdir(save_folder)
        dataset_recorder.record_dataset(
            save_folder, poses, rolls, pitches, yaws, labels
        )


if __name__ == "__main__":
    main()
