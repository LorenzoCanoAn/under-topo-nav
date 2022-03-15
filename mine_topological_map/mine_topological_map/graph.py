from dis import Instruction
import math
import networkx
import numpy as np
import cv2 as cv

class Node:
    def __init__(self, graph, i_id,  center):
        self.graph = graph
        self.id = i_id
        self.center = center
        self.connections = []

    def is_point_inside(self, point):
        p1 = self.center
        p2 = point
        d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        if d < self.radius:
            return True
        else:
            return False

    def add_connection(self, conn):
        self.connections.append(conn)

    def delete_connection(self, conn):
        self.connections.pop(self.connections.index(conn))

    def order_connections_by_angle(self):
        def key(conn):
            x1,y1 = self.center
            for idx in conn:
                if idx != self.id:
                    x2,y2 = self.graph.get_node_by_id(idx).center
           
            x = x2-x1
            y = y2-y1
            angle = (math.atan2(y,x)+2*np.math.pi)%(2*np.math.pi)
            print(angle)
            return angle
        self.connections.sort(key=key)

    def give_instruction(self, origin_node, destination_node):
        self.order_connections_by_angle()
        ocn, dcn = None, None
        
        for n, conn in enumerate(self.connections):
            if origin_node in conn:
                ocn = n
            if destination_node in conn:
                dcn = n
        if ocn != dcn:
            print("origin: {} destination{}".format(ocn, dcn))
            preliminar_instruction = dcn-ocn
            return preliminar_instruction
        else:
            print("ERROR: origin node and destination node give the same connection")

    def get_connected_nodes(self):
        connected_nodes = []
        for conn in self.connections:
            for idx in conn:
                if idx != self.id:
                    connected_nodes.append(idx)
        return connected_nodes



class Graph:
    def __init__(self):
        self.nodes = []
        self.connections = []
        self.id_counter = 0

    def generate_path(self, origin, end):
        solver = networkx.Graph()        
        for edge in self.connections:
            solver.add_edge(edge[0],edge[1],weight=1)

        return networkx.shortest_path(solver, origin, end, weight='weight')

    def generate_instructions_from_path(self, path):
        instructions = []
        for n, node_id in enumerate(path):
            if n >= 1:
                if n < path.__len__()-1:
                    instructions.append(self.get_node_by_id(node_id).give_instruction(path[n-1], path[n+1]))
        return instructions

    def get_id_from_click(self, position):
        for node in self.nodes:
            if node.is_point_inside(position):
                return node.id

    def get_node_by_id(self, i_id):
        for node in self.nodes:
            if node.id == i_id:
                return node

    def delete_node(self, i_id):
        for n, node in enumerate(self.nodes):
            if node.id == i_id:
                self.delete_connections_of_node(i_id)
                self.nodes.pop(n)
                return True
        return False

    def delete_connection(self, connection):
        for node in self.nodes:
            assert(isinstance(node, Node))
            if node.id in connection:
                node.connections.remove(connection)
        self.connections.remove(connection)

    def delete_connections_of_node(self, i_id):
        idxs = []
        for n, conn in enumerate(self.connections):
            if i_id in conn:
                self.get_node_by_id(conn[0]).delete_connection(conn)
                self.get_node_by_id(conn[1]).delete_connection(conn)
                idxs.append(n)
        idxs.sort(reverse=True)
        for idx in idxs:

            self.connections.pop(idx)

    def add_node(self, position, i_id=None):
        if i_id == None:
            i_id = self.generate_new_id()
        self.nodes.append(Node(self, i_id, position))
        self.id_counter += 1
        return i_id
    def generate_new_id(self):
        max_id = 0
        for node in self.nodes:
            if node.id > max_id:
                max_id = node.id
        return max_id + 1
    def add_connection(self, id1, id2):
        for connection in self.connections:
            if id1 in connection and id2 in connection:
                print("Connection already present")
                return
        self.connections.append((id1, id2))
        for node in self.nodes:
            if node.id in (id1, id2):
                node.add_connection((id1,id2))

    def refine_multinode_galleries(self):
        deleted_a_node = False
        for node in self.nodes:
            assert(isinstance(node, Node))
            if node.connections.__len__() == 2:
                for connection in node.get_connected_nodes():
                    neighbor_node = self.get_node_by_id(connection)
                    assert(isinstance(neighbor_node, Node))
                    if neighbor_node.connections.__len__() == 2:
                        self.delete_gallery_node_keep_connection(neighbor_node)
                        deleted_a_node = True
                        break
            if deleted_a_node:
                break
        return deleted_a_node

    def refine_double_intersections(self):
        added_node = False
        for node in self.nodes:
            assert(isinstance(node, Node))
            if node.connections.__len__() > 2:
                for connection in node.get_connected_nodes():
                    neighbor_node = self.get_node_by_id(connection)
                    assert(isinstance(neighbor_node, Node))
                    if neighbor_node.connections.__len__() > 2:
                        self.add_node_between_connected_nodes(node.id, neighbor_node.id)
                        added_node = True
                        break
            if added_node:
                break
        return added_node

    def get_connection(self, id1, id2):
        for connection in self.connections:
            if id1 in connection and id2 in connection:
                return connection
        return None

    def delete_gallery_node_keep_connection(self, node):
        assert(isinstance(node, Node))
        nodes_to_connect = node.get_connected_nodes()
        self.delete_node(node.id)
        assert(nodes_to_connect.__len__() == 2)
        self.add_connection(nodes_to_connect[0], nodes_to_connect[1])

    def add_node_between_connected_nodes(self, id1, id2):
        connection = self.get_connection(id1, id2)
        if connection != None:
            self.delete_connection(connection)
            p1 = self.get_node_by_id(id1).center
            p2 = self.get_node_by_id(id2).center
            xmid = (p1[0] + p2[0])/2
            ymid = (p1[1] + p2[1])/2
            new_node_id = self.add_node((xmid, ymid))
            self.add_connection(new_node_id, id1)
            self.add_connection(new_node_id, id2)

    def auto_gen_instructions(self):
        # get initial node
        min_dist = 20
        for node in self.nodes:
            if node.connections.__len__() == 1:
                d = math.sqrt(node.center[0]**2 + (node.center[1]-13)**2)
                if d < min_dist:
                    min_dist = d
                    initial_node_id = node.id

        max_path_len = 0
        for node in self.nodes:
            if node.id == initial_node_id:
                continue
            else:
                path = self.generate_path(initial_node_id, node.id)
                if path.__len__() > max_path_len:
                    final_path = path
                    final_node_coords = node.center
                    max_path_len = path.__len__()
            

        instructions = self.generate_instructions_from_path(final_path)
        return instructions, final_node_coords, final_path