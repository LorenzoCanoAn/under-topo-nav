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

gaussian_witdth = np.arange(-3, 3, 0.1)
GAUSSIAN = norm.pdf(gaussian_witdth, 0, 1)
GAUSSIAN = GAUSSIAN / max(GAUSSIAN)


def plot_label(label):
    assert len(label) == 360
    np.roll(label, 180)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    angles = np.deg2rad(np.linspace(-180, 179, len(label)))
    plt.plot(angles, label)
    plt.show()


def put_gaussian_in_place(i_array, idx):
    for i in range(60):
        idx_ = (idx - 30 + i) % 360
        i_array[idx_] = max(GAUSSIAN[i], i_array[idx_])
    return i_array


def label_from_angles(angles_deg):
    label = np.zeros((360))
    for angle in angles_deg:
        label = put_gaussian_in_place(label, angle)
    return label


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


class PoseAndLabelGenerator:
    def __init__(
        self,
        axis_points,
        axis_vectors,
        axis_radi,
        mesh: trimesh.Trimesh,
        dist_between_aps,
        max_hor_disp,
        max_vert_disp,
        min_vert_disp,
        max_inclination_rad,
    ):
        self.ap_manager = AxisPointManager(
            axis_points, axis_vectors, voxel_size=max(axis_radi)
        )
        self.mesh = mesh
        self.axis_points = axis_points
        self.axis_vectors = axis_vectors
        self.axis_radi = axis_radi
        self.dist_between_aps = dist_between_aps
        self.max_hor_disp = max_hor_disp
        self.max_vert_disp = max_vert_disp
        self.min_vert_disp = min_vert_disp
        self.max_inclination_rad = max_inclination_rad

    def gen_one_sample(self, place="other"):
        while True:
            base_point_idx = np.random.randint(0, len(self.ap_manager.axis_points))
            base_point = np.reshape(self.axis_points[base_point_idx, :], (1, 3))
            base_vector = np.reshape(self.axis_vectors[base_point_idx, :], (1, 3))
            local_radius = self.axis_radi[base_point_idx]
            if place == "other":
                if local_radius < 10:
                    break
            else:
                if local_radius >= 10:
                    break
        dist_to_axis = self.max_hor_disp * np.random.random()
        if dist_to_axis > local_radius:
            dist_to_axis = local_radius * 0.7
        theta = np.arctan2(base_vector[0, 1], base_vector[0, 0])
        if np.random.random() > 0.5:
            theta += np.pi
        x_disp = dist_to_axis * np.sin(theta)
        y_disp = dist_to_axis * np.cos(theta)
        z_disp = np.random.uniform(self.min_vert_disp, self.max_vert_disp)
        pose = base_point + np.reshape(np.array([x_disp, y_disp, z_disp]), (1, 3))
        pose = np.reshape(pose, -1)
        points_for_label = self.get_label_points(pose, local_radius)
        roll = np.random.uniform(-self.max_inclination_rad, self.max_inclination_rad)
        pitch = np.random.uniform(-self.max_inclination_rad, self.max_inclination_rad)
        yaw = np.random.uniform(0, 2 * np.pi)
        tf = get_transformation_matrix(pose[0], pose[1], pose[2], roll, pitch, yaw)
        points_for_label = np.concatenate(
            (points_for_label, np.ones((len(points_for_label), 1))), axis=1
        )
        points_for_label = np.matmul(np.linalg.inv(tf), points_for_label.T).T
        yaw_of_points_deg = np.rad2deg(
            np.arctan2(points_for_label[:, 1], points_for_label[:, 0])
        )
        label = label_from_angles(yaw_of_points_deg.astype(int))
        return pose, roll, pitch, yaw, label

    def gen_n_samples(self, n_samples, percentage_of_diaph_intersections):
        poses = np.zeros((n_samples, 3))
        rolls = np.zeros((n_samples, 1))
        pitches = np.zeros((n_samples, 1))
        yaws = np.zeros((n_samples, 1))
        labels = np.zeros((n_samples, 360))
        n_diaph = int(n_samples * percentage_of_diaph_intersections)
        j = 0
        for i in range(n_samples):
            print(f"{i:04d}", end="\r", flush=True)
            if j < n_diaph:
                pose, roll, pitch, yaw, label = self.gen_one_sample(place="diaph_inter")
                j += 1
            else:
                pose, roll, pitch, yaw, label = self.gen_one_sample(place="other")
            poses[i, :] = pose
            rolls[i, :] = roll
            pitches[i, :] = pitch
            yaws[i, :] = yaw
            labels[i, :] = label
        return poses, rolls, pitches, yaws, labels

    def get_label_points(self, point, label_dist):
        axis_data = self.ap_manager.get_relevant_points(np.reshape(point, -1))
        points = axis_data[:, :3]
        vectors = axis_data[:, 3:6]
        dists = np.linalg.norm(points - point, axis=1)
        idxs_of_label_points = np.abs(dists - label_dist) < self.dist_between_aps / 2
        ap_at_distance = points[idxs_of_label_points, :]
        av = vectors[idxs_of_label_points, :]
        p_to_ap_v = ap_at_distance - point
        p_to_ap_v /= np.linalg.norm(p_to_ap_v, axis=1)
        product = np.einsum("ij,ij->i", av, p_to_ap_v)
        angles = np.arcos(np.abs(product))
        ap_at_distance = ap_at_distance[np.where(angles < np.deg2rad(70))]
        # Check if ray from point intersects the mesh
        point = np.reshape(point, (1, 3))
        idxs_to_delete = []
        for idx, ap in enumerate(ap_at_distance):
            ap = np.reshape(ap, (1, 3))
            vector = ap - point
            d_p_ap = np.linalg.norm(vector)
            vector /= d_p_ap
            intersections, _, _ = self.mesh.ray.intersects_location(point, vector)
            d_to_inter = np.linalg.norm(intersections - point, axis=1)
            if np.min(d_to_inter) < d_p_ap:
                idxs_to_delete.append(idx)
        final_points = np.delete(ap_at_distance, idxs_to_delete, axis=0)
        final_points = np.reshape(final_points, (-1, 3))
        return final_points


def get_args():
    parser = argparse.ArgumentParser(
        prog="Dataset collector", description="Given a dataset folder structure"
    )
    parser.add_argument("-F", "--dataset_folder", required=True, type=str)
    parser.add_argument("-N", "--number_of_samples_per_env", required=True, type=int)
    parser.add_argument("-O", "--overwrite", required=False, type=bool, default=False)
    parser.add_argument(
        "-MH", "--max_horizontal_displacement", required=True, type=float
    )
    parser.add_argument("-MV", "--max_vertical_displacement", default=0, type=float)
    parser.add_argument("-_MV", "--min_vertical_displacement", default=0, type=float)
    parser.add_argument("-MI", "--max_inclination_deg", default=10, type=float)
    parser.add_argument("--fraction_of_diaph_intersections", default=0.3, type=float)
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
        self.image_topic = rospy.get_param("~data_topic", default="/lidar_image")
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
            # plt.figure()
            # ax1 = plt.subplot(2, 1, 1)
            # plt.imshow(image)
            # ax2 = plt.subplot(2, 1, 2)
            # plt.plot(label)
            # plt.show()

    def change_environment(self, file_to_model):
        with open(file_to_model, "r") as f:
            model_text = f.read()
        delete_request = DeleteModelRequest()
        delete_request.model_name = "cave"
        self.delete_model_srv_proxy.call(delete_request)
        time.sleep(5)
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
    fraction_of_diaph_intersection = args.fraction_of_diaph_intersections
    max_inclination_rad = np.deg2rad(args.max_inclination_deg)
    dataset_recorder = DatasetRecorder()
    env_folders = os.listdir(dataset_folder)
    env_folders.sort()
    for env_folder in env_folders:
        print(env_folder)
        abs_env_folder = os.path.join(dataset_folder, env_folder)
        axis_file = os.path.join(abs_env_folder, "axis.txt")
        model_file = os.path.join(abs_env_folder, "model.sdf")
        fta_dist_file = os.path.join(abs_env_folder, "fta_dist.txt")
        mesh_file = os.path.join(abs_env_folder, "mesh.obj")
        dataset_recorder.change_environment(model_file)
        axis_data = np.loadtxt(axis_file)
        axis_points = axis_data[:, :3]
        axis_vectors = axis_data[:, 3:6]
        axis_radi = axis_data[:, 6]
        fta_dist = np.loadtxt(fta_dist_file).item(0)
        pose_and_label_generator = PoseAndLabelGenerator(
            axis_points,
            axis_vectors,
            axis_radi,
            mesh=trimesh.load_mesh(mesh_file),
            dist_between_aps=0.6,
            max_hor_disp=max_horizontal_displacement,
            max_vert_disp=max_vertical_displacement - fta_dist,
            min_vert_disp=min_vertical_displacement - fta_dist,
            max_inclination_rad=max_inclination_rad,
        )
        poses, rolls, pitches, yaws, labels = pose_and_label_generator.gen_n_samples(
            n_samples, fraction_of_diaph_intersection
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
