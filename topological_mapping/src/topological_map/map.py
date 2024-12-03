from scipy.spatial.transform import Rotation
import math
import numpy as np
from enum import Enum, auto
import networkx
import itertools

class NodeType(Enum):
    VIRTUAL = auto()
    REAL = auto()


class Node:
    id_iter = itertools.count()

    def __init__(self, nodetype: NodeType, pose: np.ndarray = np.zeros((1,3))):
        self._pose: np.ndarray = pose
        self._neighbors: list[Node] = []
        self.nodetype: NodeType = nodetype
        self.__id: int = next(self.__class__.id_iter)

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def n_neighbors(self):
        return len(self._neighbors)

    def make_real(self):
        self.nodetype = NodeType.REAL

    def add_neighbor(self, neigh):
        assert isinstance(neigh, Node)
        assert not neigh in self._neighbors
        self._neighbors.append(neigh)
    
    def del_neighbor(self, neigh):
        assert isinstance(neigh, Node)
        assert neigh in self._neighbors
        self._neighbors.append(neigh)

    def is_node_neighbor(self, other_node):
        return other_node in self._neighbors

    def __eq__(self, other):
        assert isinstance(other, Node)
        return self.id == other.id

    def __hash__(self):
        return hash("node" + str(self.id))

    def update_pose(self, newpose: np.ndarray):
        self._pose = newpose

    @property
    def id(self):
        return self.__id


class EdgeType(Enum):
    COMPLETE = auto()
    SEMICOMP = auto()
    IMAGINARY = auto()


class Edge:

    def __init__(self, node_1: Node, node_2: Node, connect_nodes=True):
        nodes = [node_1, node_2]
        nodes.sort(key=lambda a: a.id)
        self.nodes = tuple(nodes)
        if connect_nodes:
            node_1.add_neighbor(node_2)
            node_2.add_neighbor(node_1)

    def __eq__(self, other):
        assert isinstance(other, Edge)
        for node in self.nodes:
            if not node in other.nodes:
                return False
        return False

    def __hash__(self) -> int:
        return hash(f"{self.nodes[0].id}_{self.nodes[1].id}")

    @property
    def type(self) -> EdgeType:
        if self.nodes[0].nodetype == self.nodes[1].nodetype == NodeType.REAL:
            return EdgeType.COMPLETE
        elif self.nodes[0].nodetype != self.nodes[2].nodetype:
            return EdgeType.SEMICOMP
        else:
            return EdgeType.IMAGINARY

    @property
    def direction(self):
        return self.nodes[1]._pose - self.nodes[0]._pose

    @property
    def distance(self):
        return np.linalg.norm(self.direction, 2)

    @property
    def center(self):
        return (self.nodes[0]._pose + self.nodes[1]._pose) / 2


class TopologicalMap:
    seq = 0

    def __init__(self):
        self.updated = True
        self.nodes: set[Node] = set()
        self.edges: set[Edge] = set()

    def add_node(self, n: Node):
        self.nodes.add(n)

    def add_edge(self, n1: Node, n2: Node):
        self.add_node(n1)
        self.add_node(n2)
        edge = Edge(n1, n2)
        self.edges.add(edge)
        self.updated = True

    @property
    def to_networkx(self):
        graph = networkx.Graph()
        for edge in self.edges:
            graph.add_edge(edge.nodes[0], edge.nodes[1])
        return graph

    def are_nodes_neigh(self, node1: Node, node2: Node):
        c1 = node1.is_node_neighbor(node2)
        c2 = node2.is_node_neighbor(node1)
        assert c1 == c2  # Check for errors.
        return c1

    def instruction(self, prev_node: Node, curr_node: Node, next_node: Node):
        assert self.are_nodes_neigh(prev_node, curr_node)
        assert self.are_nodes_neigh(curr_node, next_node)
        back_edge_idx = curr_node._neighbors.index(prev_node)
        next_edge_idx = curr_node._neighbors.index(next_node)
        return next_edge_idx - back_edge_idx

    def get_instructions_to_get_to_node(self, obj_node: Node, current_node: Node, prev_node: Node):
        assert self.are_nodes_neigh(current_node, prev_node)
        nxgraph = self.to_networkx
        sequence = networkx.shortest_path(nxgraph, current_node, obj_node)
        if current_node.n_neighbors == 1:
            instructions = ["advance_until_node"]
        else:
            sequence = [prev_node] + sequence
            instructions = []
        for i in range(len(sequence) - 2):
            instructions.append(f"take {self.instruction(sequence[i], sequence[i+1], sequence[i+2])}")
        instructions.append("advance_until_node")
        return instructions, sequence

    @property
    def theoretical_nodes(self):
        return [n for n in self.nodes if n.nodetype == NodeType.VIRTUAL]

    def update_node_position(self, node: Node, position: np.ndarray):
        node.update_pose(position)

