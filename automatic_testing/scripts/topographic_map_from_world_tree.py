import pickle
from world_file_generation.tile import Tile, TileTree
from mine_topological_map.graph import Graph
from mine_topological_map.drawing import GraphDrawing
from world_file_generation.automatic_world_file_generator import AutoMapGenerator



def topological_graph_from_world_tree(tree):
    graph = Graph()
    for tile in tree.tiles:
        graph.add_node((tile.pose[0], tile.pose[1]),i_id=tile.get_id())
    
    for tile in tree.tiles:
        for connection in tile.connections:
            graph.add_connection(tile.id,connection)

    while graph.refine_multinode_galleries():
        pass

    while graph.refine_double_intersections():
        pass

    return graph

    

if __name__ == "__main__":
    with open("/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/automatic_testing/tests_data/test_7_2_10/trees/1646157837488404942.pickle", "rb") as f:
        tree = pickle.load(f)
    graph = topological_graph_from_world_tree(tree)
    graph_gui = GraphDrawing()
    graph_gui.radius = 10
    
    graph_gui.set_graph(graph)
    graph_gui.run_gui()
