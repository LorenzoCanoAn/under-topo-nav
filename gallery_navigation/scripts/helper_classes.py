import rospy
import move_base_msgs.msg as ros_mb_msg
import geometry_msgs.msg as ros_geom_msg
import std_msgs.msg as ros_std_msg
import tf
import actionlib
import numpy as np
from collections import Counter
 

##############################################################
#	HELPER FUNCTIONS
##############################################################
def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def euler_to_quaternion(yaw, pitch, roll):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

##############################################################
#	HELPER CLASSES
##############################################################
class NNOutputHandler:
    def __init__(self):
        self.first_callback = False
        self.nn_subscriber = rospy.Subscriber(
            "/gallery_detection_vector", ros_std_msg.Float32MultiArray, self.neural_network_callback)

    def neural_network_callback(self, msg: ros_std_msg.Float32MultiArray):
        self.vector = msg.data
        self.filtered = self.filter_vector(msg.data)
        self.gallery_angles = self.filtered_to_gallery_angles(self.filtered)
        self.situation = self.determine_situation(self.gallery_angles)
        self.quadrants = self.get_quadrants_from_angles(self.gallery_angles)
        self.valid_directions = self.get_valid_directions_from_quadrants(
            self.quadrants)
        self.first_callback = True

    def get_valid_directions_from_quadrants(self, quadrants):
        valid_directions = []
        for key in quadrants.keys():
            if type(None) != type(quadrants[key]):
                valid_directions.append(key)
        return valid_directions

    def has_first_callback_happened(self):
        return self.first_callback

    def get_quadrants(self):
        return self.quadrants

    def change_nn_callback(self, new_function):
        self.nn_subscriber = rospy.Subscriber(
            "/gallery_detection_vector", ros_std_msg.Float32MultiArray, new_function)

    def filter_vector(self, vector):
        filtered = np.zeros(360)
        for i in range(360):
            to_check = vector[i]
            filtered[i] = to_check
            a = 40
            for j in range(a):
                index_inside_subsection = ((-int(a/2) + j) + i) % 356
                if vector[index_inside_subsection] > to_check:
                    filtered[i] = 0
        return filtered

    def array_position_to_angle(self, array_position):
        return 180 - array_position

    def filtered_to_gallery_angles(self, filtered):
        max_peak = np.max(filtered)
        ratio = 0.3
        galleries_indices = np.nonzero(self.filtered > max_peak * ratio)[0]
        galleries_angles = []
        for index in galleries_indices:
            galleries_angles.append(
                self.array_position_to_angle(index)/180.0 * np.math.pi)
        true_gallery_angles = []
        for a1 in galleries_angles:
            passes = True
            for a2 in true_gallery_angles:
                if self.min_distance(a1, a2) < 0.17:  # 10 degrees
                    passes = False
            if passes:
                true_gallery_angles.append(a1)
        return true_gallery_angles

    def determine_situation(self, gallery_angles):
        n = gallery_angles.__len__()
        if n == 1:
            return "in_end_of_gallery"
        elif n == 2:
            return "in_rect"
        elif n > 2:
            return "in_node"

    def min_distance(self, angle, obj):
        distance = (angle - obj) % (np.math.pi*2)
        if distance < -np.math.pi:
            distance += np.math.pi * 2
        elif distance > np.math.pi:
            distance -= np.math.pi * 2
        distance = abs(distance)
        return distance

    def get_closest_angle_with_tolerance(self, angles, obj, tolerance=50):
        min_distance = 4
        for angle in angles:
            distance = self.min_distance(angle, obj)
            if distance < min_distance:
                min_distance = distance
                candidate = angle

        if min_distance < tolerance:
            return candidate
        else:
            return None

    def get_angle_to_front(self, angles):
        return self.get_closest_angle_with_tolerance(angles, 0)

    def get_angle_to_right(self, angles):
        return self.get_closest_angle_with_tolerance(angles, -np.math.pi / 4)

    def get_angle_to_left(self, angles):
        return self.get_closest_angle_with_tolerance(angles, np.math.pi / 4)

    def get_angle_to_back(self, angles):
        return self.get_closest_angle_with_tolerance(angles, np.math.pi)

    def get_quadrants_from_angles(self, angles):
        quadrants = {}
        quadrants["front"] = self.get_angle_to_front(angles)
        quadrants["back"] = self.get_angle_to_back(angles)
        quadrants["left"] = self.get_angle_to_left(angles)
        quadrants["right"] = self.get_angle_to_right(angles)
        return quadrants


class MoveBaseHandler:
    def __init__(self):
        self.first_callback = False
        self.move_base_active = False
        self.seq = 0
        self.listener = tf.TransformListener()
        self.tf_transformer = tf.TransformerROS()
        self.move_base_client = actionlib.SimpleActionClient(
            "/move_base", ros_mb_msg.MoveBaseAction)
        if self.move_base_client.wait_for_server(timeout=rospy.Duration.from_sec(5)):
            rospy.loginfo("MOVE BASE RECIEVED")
        else:
            rospy.logerr("MOVE BASE NOT ACTIVE")

    def angle_to_point(self, angle, d):
        quaternion = euler_to_quaternion(angle, 0, 0)
        point = [d * np.math.cos(angle), d * np.math.sin(angle), 0]
        return point, quaternion

    def get_seq(self):
        self.seq += 1
        return self.seq - 1

    def point_to_geom_msg(self, point, quaternion):
        header = ros_std_msg.Header(
            self.get_seq(), rospy.Time.now(), "base_link")
        position = ros_geom_msg.Point(point[0], point[1], point[2])
        orientation = ros_geom_msg.Quaternion(
            quaternion[0], quaternion[1], quaternion[2], quaternion[3])
        pose = ros_geom_msg.Pose(position, orientation)
        geom_msg = ros_geom_msg.PoseStamped(header, pose)
        return geom_msg

    def send_goal_from_angle(self, angle, distance=2):
        point, quaternion = self.angle_to_point(angle, distance)
        goal_geom_message = self.point_to_geom_msg(point, quaternion)
        # Transform the goal to the map frame

        t = self.listener.getLatestCommonTime("odom", "base_link")
        goal_geom_message.header.stamp = t
        self.tf_transformer._buffer = self.listener._buffer
        goal_geom_message = self.tf_transformer.transformPose(
            "odom", goal_geom_message)

        goal_msg = ros_mb_msg.MoveBaseGoal(goal_geom_message)
        self.current_goal = goal_msg
        self.move_base_client.send_goal(
            goal_msg, done_cb=self.done_cb, active_cb=self.active_cb, feedback_cb=self.feedback_cb)

    def done_cb(self, msg: ros_mb_msg.MoveBaseResult, hola):
        self.move_base_active = False

    def active_cb(self):
        self.move_base_active = True

    def feedback_cb(self, msg: ros_mb_msg.MoveBaseFeedback):
        current_position = msg.base_position.pose.position
        x_diff = current_position.x - self.current_goal.target_pose.pose.position.x
        y_diff = current_position.y - self.current_goal.target_pose.pose.position.y
        distance = np.math.sqrt(x_diff**2 + y_diff**2)
        if distance < 5:
            self.move_base_active = False
