import pyvista as pv
from .map import Edge, Node, TopologicalMap, NodeType
import numpy as np

NODE_TYPE_TO_COLOR = {
    NodeType.REAL: "blue",
    NodeType.VIRTUAL: "green",
}


def plot_node(plotter: pv.Plotter, node: Node):
    color = NODE_TYPE_TO_COLOR[node.nodetype]
    plotter.add_mesh(pv.Sphere(2, center=node._pose), color=color)


def plot_edge(plotter: pv.Plotter, edge: Edge):
    plotter.add_mesh(pv.Cylinder(center=edge.center, direction=edge.direction, radius=1, height=edge.distance))


def plot_nodes(plotter: pv.Plotter, nodes: Node):

    for node in nodes:
        plot_node(plotter, node)


def plot_edges(plotter: pv.Plotter, edges: list[Edge]):
    for edge in edges:
        plot_edge(plotter, edge)


def plot_map(map: TopologicalMap) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True)
    plot_nodes(plotter, list(map.nodes))
    plot_edges(plotter, list(map.edges))
    plotter.view_xy()
    return plotter.show(screenshot=True)
