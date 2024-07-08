import rospy
from topological_exploration.msg import TopologicalMap as TopoMapMsg
from gallery_tracking.msg import TrackedGalleries
from gallery_detection_ros.msg import DetectionVectorStability
from nav_msgs.msg import Odometry
from message_filters import Subscriber, TimeSynchronizer
from enum import auto, Enum
import numpy as np
import pyvista as pv


class Pose:
    def __init__(self, *arg):
        args = list(arg)
        self.x = arg[0]
        self.y = arg[1]
        self.z = arg[2]
        self.asarr = np.array(arg).reshape(-1)

    def __diff__(self, other):
        if isinstance(other, Pose):
            return self.asarr - other.asarr
        else:
            raise ValueError(f"Not supperted for {type(other)}")

    def __sum__(self, other):
        if isinstance(other, Pose):
            return self.asarr + other.asarr
        else:
            raise ValueError(f"Not supperted for {type(other)}")


class NodeType(Enum):
    THEORETICAL = auto
    REAL = auto


class Node:
    id = 0

    @classmethod
    def get_new_id(cls):
        id_to_give = cls.id
        cls.id += 1
        return id_to_give

    def __init__(self, pose, nodetype, edges):
        self.pose: Pose = pose
        self.edges: list[Edge] = edges
        self.type = nodetype
        self.id = self.get_new_id()

    def __eq__(self, other):
        assert isinstance(other, Node)
        return self.id == other.id


class Edge:
    def __init__(self, node_1: Node, node_2: Node):
        if node_1.id < node_2.id:
            self.nodes = (node_1, node_2)
        elif node_1.id > node_2.id:
            self.nodes = (node_2, node_1)
        elif node_1.id == node_2.id:
            raise ValueError("The two nodes have the same id")

    def __eq__(self, other):
        assert isinstance(other, Edge)
        for node in self.nodes:
            if not node in other.nodes:
                return False
        return False

    def __hash__(self) -> int:
        return hash(f"{self.nodes[0].id}_{self.nodes[1].id}")

    @property
    def direction(self):
        return self.nodes[1] - self.nodes[0]

    @property
    def distance(self):
        return np.linalg.norm(self.direction, 2)

    @property
    def center(self):
        return (self.nodes[0] + self.nodes[1]) / 2


class States(Enum):
    START = auto()
    IN_EDGE = auto()
    IN_NODE = auto()
    IN_TRANSIT = auto()


class TopologicalMap:
    seq = 0

    @classmethod
    def get_seq(cls):
        to_return = cls.seq
        cls.seq += 1
        return to_return

    def __init__(self):
        self.nodes: set[Node] = set()
        self.edges: set[Edge] = set()

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        for node in edge.nodes:
            self.nodes.add(node)

    def remove_edge(self, edge: Edge):
        if edge in self.edges:
            self.edges.remove(edge)
            for node in edge.nodes:
                self.nodes.remove(node)


class TopologicalMappingNode:
    def __init__(self):
        rospy.init_node(self.__class__.__name__)
        # Set variables
        self.map = TopologicalMap()
        # Get params
        topo_map_topic = rospy.get_param("~topo_map_topic", "/topological_map")
        # Set publishers
        self.map_publisher = rospy.Publisher(topo_map_topic, TopoMapMsg, queue_size=1, latch=True)
        # Set subscribers
        sub1 = Subscriber("/tracked_galleries", TrackedGalleries)
        sub2 = Subscriber("/is_detection_stable", DetectionVectorStability)
        sync = TimeSynchronizer([sub1, sub2], queue_size=1)
        sync.registerCallback(self.callback)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)

    def odom_callback(self, msg: Odometry):
        self.current_pose = msg.pose.pose.position

    def callback(self, traked_galleries: TrackedGalleries, stability: DetectionVectorStability):
        print("Callback")


######################################################################################################
######################################################################################################
# Plotting Utils
######################################################################################################
######################################################################################################

NODE_TYPE_TO_COLOR = {
    NodeType.REAL: "blue",
    NodeType.THEORETICAL: "green",
}


def plot_node(plotter: pv.Plotter, node: Node):
    color = NODE_TYPE_TO_COLOR[node.type]
    plotter.add_mesh(pv.Sphere(2, center=node.pose), color=color)


def plot_edge(plotter: pv.Plotter, edge: Edge):
    plotter.add_mesh(pv.Cylinder(center=edge.center, direction=edge.direction, radius=1, height=edge.distance))


def plot_nodes(plotter: pv.Plotter, nodes: dict[int:Node]):
    for node_id in nodes.keys():
        plot_node(plotter, nodes[node_id])


def plot_edges(plotter: pv.Plotter, edges: list[Edge]):
    for edge in edges:
        plot_edge(plotter, edge)


def plot_map(map: TopologicalMap):
    plotter = pv.Plotter(off_screen=True)
    plot_nodes(plotter, map.nodes)
    plot_edges(plotter, map.edges)
    return plotter.show(screenshot=True)
