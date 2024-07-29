import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from gallery_tracking.msg import TrackedGalleries
from gallery_detection_ros.msg import DetectionVectorStability
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from enum import auto, Enum
import numpy as np
import pyvista as pv
from actionlib.action_client import ActionClient
from topological_navigation.msg import topo_navigateAction, topo_navigateFeedback, topo_navigateGoal, topo_navigateResult, topo_navigateActionResult
from scipy.spatial.transform import Rotation
import math
import networkx
from topological_map import TopologicalMap, Node, NodeType
from topological_map.plotting import plot_map


def xyz_of_odom(odom: Odometry) -> np.ndarray:
    p = odom.pose.pose.position
    return np.array((p.x, p.y, p.z))


def rpy_of_odom(odometry: Odometry):
    r = odometry.pose.pose.orientation
    quat = np.array((r.x, r.y, r.z, r.w))
    R = Rotation.from_quat(quat)
    return R.as_euler("xyz")


def abs_dir_to_node(odometry: Odometry, galangle: float):
    r, p, y = rpy_of_odom(odometry)
    abs_galangle = r + galangle
    x = math.cos(abs_galangle)
    y = math.sin(abs_galangle)
    return np.array((x, y, 0))


def assert_map_is_empty(map: TopologicalMap):
    assert len(map.nodes) == 0
    assert len(map.edges) == 0


def xyz_of_virt_node(odom, galang):
    base_xyz = xyz_of_odom(odom)
    diff_xyz = abs_dir_to_node(odom, galang)
    return base_xyz + diff_xyz * 10


def wrap_angle_2pi(angle):
    return angle % (np.pi * 2)


def wrap_angle_pi(angle):
    angle = wrap_angle_2pi(angle)
    if angle > np.pi:
        angle = angle - np.pi * 2
    return angle


def roll_list(i_list: list, n):
    for _ in range(n):
        i_list.append(i_list[0])
        i_list.pop(0)
    return i_list


def angular_magnitude(angle):
    # distance of angle to 0
    return abs(wrap_angle_pi(angle))


def tracked_gals_to_list_of_tuples(gals: TrackedGalleries):
    """Takes a TrackedGalleries msg and outputs a list of tuples. Each tuple has two elements, the first one
    is the id, the second one the angle."""
    return [(id, ang) for id, ang in zip(gals.ids, gals.angles)]


def magnitude_key(ang_id_tuple):
    """Ordering key for the list.sort method. The list that uses it must be a list of tuples, where
    the first element of each tuple is the id of a gallery, and the second the angle. Returns the distance of the gallery
    to the angle 0."""
    return angular_magnitude(ang_id_tuple[1])


def angle_key(ang_id_tuple):
    """Ordering key for the list.sort method. The list that uses it must be a list of tuples, where
    the first element of each tuple is the id of a gallery, and the second the angle. Returns the angle wraped from 0 to 2pi"""
    return wrap_angle_2pi(ang_id_tuple[1])


def order_gals_by_angle_magnitude(gals: TrackedGalleries):
    """Takes a TrackedGalleries msg and outputs a list of tuples. Each tuple has two elements, the first one
    is the id, the second one the angle. The tuples are ordered inside of the list so that the first one has the
    closest angle to 0 (either from left or right), the second the second smallest etc..."""
    list_of_tuples = tracked_gals_to_list_of_tuples(gals)
    list_of_tuples.sort(key=magnitude_key)
    return list_of_tuples


def order_gals_by_angle(gals: TrackedGalleries):
    """Takes a TrackedGalleries msg and outputs a list of tuples. Each tuple has two elements, the first one
    is the id, the second one the angle. The tuples are ordered in with increasing angles from 0 to 2*pi"""
    list_of_tuples = tracked_gals_to_list_of_tuples(gals)
    list_of_tuples.sort(key=angle_key)
    return list_of_tuples


def back_gal_id(gals: TrackedGalleries):
    """Returns the back gallery id from a TrackedGalleries msg"""
    return order_gals_by_angle_magnitude(gals)[-1][0]


def order_gals_by_angle_starting_at_back(gals: TrackedGalleries) -> list[tuple]:
    """This function takes a TrackedGalleries msg, and outputs a list of tuples. Each tuple
    has two elements, the first being the id and second the angle. The tuples are ordered inside of the list
    so that the first one is the back galleriy, the second is the next one to the right of the back gallery etc..."""
    ordered_gals = order_gals_by_angle(gals)
    bgid = back_gal_id(gals)
    for n, (galid, galang) in enumerate(ordered_gals):
        if galid == bgid:
            bgidx = n
    reordered_gals = roll_list(ordered_gals, bgidx)
    return reordered_gals


def init_map_from_end_of_gal(map: TopologicalMap, odom: Odometry, gals: TrackedGalleries) -> Node:
    # Set current node as visited and add edge to next node
    assert len(gals.angles) == 1
    assert_map_is_empty(map)
    n1 = Node(NodeType.REAL, xyz_of_odom(odom))
    n2 = Node(NodeType.VIRTUAL, xyz_of_virt_node(odom, gals.angles[0]))
    map.add_edge(n1, n2)
    return n1, [n2], n2


def init_map_from_gallery(map: TopologicalMap, odom: Odometry, gals: TrackedGalleries):
    # Add two virtual nodes and set the closest one to the back as previous
    assert len(gals.angles) == 2
    assert_map_is_empty(map)
    ordered_gals = order_gals_by_angle_magnitude(gals)
    nf = Node(NodeType.VIRTUAL, xyz_of_virt_node(odom, ordered_gals[0][1]))
    nb = Node(NodeType.VIRTUAL, xyz_of_virt_node(odom, ordered_gals[1][1]))
    map.add_edge(nb, nf)
    return nb, [nf, nb], nf


def init_map_from_intersection(map: TopologicalMap, odom: Odometry, gals: TrackedGalleries):
    assert len(gals.angles) > 2
    assert_map_is_empty(map)
    nc = Node(NodeType.REAL, xyz_of_odom(odom))
    ordered_gals = order_gals_by_angle_starting_at_back(gals)
    nodes = []
    for n, (galid, galang) in enumerate(ordered_gals):
        n = Node(NodeType.VIRTUAL, xyz_of_virt_node(odom, galang))
        nodes.append(n)
        map.add_edge(nc, n)
    return nodes[0], nodes, nodes[1]


def update_map_when_arrived_at_node(map: TopologicalMap, odom: Odometry, gals: TrackedGalleries, prev_node: Node, current_node: Node):
    ordered_gals = order_gals_by_angle_starting_at_back(gals)
    ordered_gals.pop(0)
    map.update_node_position(current_node, xyz_of_odom(odom))
    current_node.make_real()
    new_nodes = []
    for id, ang in ordered_gals:
        node = Node(NodeType.VIRTUAL, xyz_of_virt_node(odom, ang))
        new_nodes.append(node)
        map.add_edge(current_node, node)
    return new_nodes


#############################################################################################
#############################################################################################
# Topological mapping node
#############################################################################################
#############################################################################################


class States(Enum):
    START = auto()
    NO_OBJ = auto()
    WITH_OBJ = auto()


class TopologicalMappingNode:
    def __init__(self):
        rospy.init_node(self.__class__.__name__)
        # Set variables
        self.map = TopologicalMap()
        self.map_update_counter = 0
        self.current_galleries: TrackedGalleries = None
        self.current_stability: DetectionVectorStability = None
        self.current_odom: Odometry = None
        self._bridge = CvBridge()
        self.nodes_to_explore: list[Node] = []
        self.change_state(States.START)
        # Get params
        topo_map_img_topic = rospy.get_param("~topo_map_topic", "/topological_map_img")
        # Set publishers
        self.map_img_publisher = rospy.Publisher(topo_map_img_topic, Image, queue_size=1, latch=True)
        self.nav_client = ActionClient("/topo_navigate", topo_navigateAction)
        rospy.Subscriber("/topo_navigate/result", topo_navigateResult, callback=self.action_server_result_cb)
        # Set subscribers
        sub1 = Subscriber("/tracked_galleries", TrackedGalleries)
        sub2 = Subscriber("/is_detection_stable", DetectionVectorStability)
        sync = TimeSynchronizer([sub1, sub2], queue_size=2)
        sync.registerCallback(self.main_callback)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg

    def set_objective(self, objective):
        print(f"Objetive set to: {objective}")
        self.objective_node: Node = objective

    def main_callback(self, traked_galleries: TrackedGalleries, stability: DetectionVectorStability):
        self.current_galleries = traked_galleries
        self.current_stability = stability.is_stable
        self.state_machine_iteration()
        if self.map_update_counter % 4 == 0:
            img = plot_map(self.map)
            self.map_img_publisher.publish(self._bridge.cv2_to_imgmsg(img))
            self.map.updated = False
        self.map_update_counter += 1

    def change_state(self, newstate: States):
        print(f"Changing state to {newstate}")
        self.state = newstate

    def action_server_result_cb(self, msg: topo_navigateActionResult):
        if msg.result.success:
            if self.objective_node.nodetype == NodeType.VIRTUAL:
                self.nodes_to_explore += update_map_when_arrived_at_node(
                    self.map, self.current_odom, self.current_galleries, self.current_node, self.objective_node
                )
            self.previous_node = self.current_node
            self.current_node = self.objective_node
            self.change_state(States.NO_OBJ)
        else:
            rospy.logerr("Did not get to node")

    def feedback_cb(self, msg: topo_navigateFeedback):
        pass

    def state_machine_iteration(self):
        if self.state == States.START:
            # This state is the initial one, and it is needed to decide to which node to go
            goal = topo_navigateGoal()
            if len(self.current_galleries.ids) == 1:
                prev_node, nodes_to_visit, obj_node = init_map_from_end_of_gal(self.map, self.current_odom, self.current_galleries)
                rospy.loginfo("Inited node from end of gallery")
                goal.topological_instructions = ["advance_until_node"]
            elif len(self.current_galleries.ids) == 2:
                prev_node, nodes_to_visit, obj_node = init_map_from_gallery(self.map, self.current_odom, self.current_galleries)
                rospy.loginfo("Inited node from gallery")
                goal.topological_instructions = ["advance_until_node"]
            elif len(self.current_galleries.ids) > 2:
                prev_node, nodes_to_visit, obj_node = init_map_from_intersection(self.map, self.current_odom, self.current_galleries)
                rospy.loginfo("Inited node from intersection")
                goal.topological_instructions = ["take 1", "advance_until_node"]
            else:
                print(self)
            rospy.loginfo(f"Sending:\n {goal.topological_instructions}")
            self.set_objective(obj_node)
            self.current_node = prev_node
            self.nodes_to_explore += nodes_to_visit
            self.send_goal(goal)

        if self.state == States.WITH_OBJ:
            pass

        elif self.state == States.NO_OBJ:
            next_theo_node = self.nodes_to_explore.pop(-1)
            goal = topo_navigateGoal()
            instructions, sequence = self.map.get_instructions_to_get_to_node(next_theo_node, self.current_node, self.previous_node)
            self.objective_node = next_theo_node
            self.current_node = sequence[-2]
            goal.topological_instructions = instructions
            self.send_goal(goal)

    def send_goal(self, goal):
        if self.state != States.WITH_OBJ:
            self.nav_client.send_goal(goal, transition_cb=self.action_server_result_cb, feedback_cb=self.feedback_cb)
            self.state = States.WITH_OBJ
        else:
            rospy.logerr("Sent a goal but the system already had a goal.")

    def run(self):
        rospy.spin()


def main():
    node = TopologicalMappingNode()
    node.run()


if __name__ == "__main__":
    main()
