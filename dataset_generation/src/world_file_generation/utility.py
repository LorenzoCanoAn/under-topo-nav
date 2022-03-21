from scipy.spatial.transform import Rotation


def euler_to_quaternion(yaw, pitch, roll, degrees=True):

    rotation_object = Rotation.from_euler(
        "xyz", [yaw, pitch, roll], degrees=degrees)
    return rotation_object.as_quat()


def quaternion_to_euler(x, y, z, w, degrees=True):

    rotation_object = Rotation.from_quat([x, y, z, w])
    return rotation_object.as_euler(seq="xyz", degrees=degrees)
