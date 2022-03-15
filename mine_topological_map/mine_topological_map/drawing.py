import pickle
import cv2 as cv
from mine_topological_map.graph import Graph, Node
import numpy as np
import networkx
import math

from torch import save
import os
# mouse callback function


cv.destroyAllWindows()
COLORS = {"none": (100, 85, 82),
          "background": (143, 130, 116),
          "base_node_color": (206, 192, 150),
          "base_path_color": (181, 185, 190),
          "initial_node_color": (86, 91, 194),
          "final_node_color": (235, 246, 254)}


class GraphDrawing:
    def __init__(self):
        self.graph = Graph()
        cv.namedWindow('image')
        cv.setMouseCallback('image', self.mouse_callback)
        self.path = []
        self.width = 1080
        self.height = 1080
        self.x_offset = 0
        self.y_offset = 0
        self.x_scale = 1
        self.y_scale = 1
        self.radius = 20
        self.margin = self.radius * 3
        self.clear_path()

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump(self.graph, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
            self.graph.clear_path()

    def set_graph(self, graph):
        self.graph = graph
        self.set_offsets()

    def set_path(self, path):
        self.path = path

    def set_offsets(self):
        assert(isinstance(self.graph, Graph))
        max_x = 0
        min_x = 0
        max_y = 0
        min_y = 0
        for node in self.graph.nodes:
            assert(isinstance(node, Node))
            x = node.center[0]
            y = node.center[1]
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        print(max_x, max_y)
        print(min_x, min_y)
        self.x_scale = (self.width - 2*self.margin) / (max_x - min_x)
        self.y_scale = (self.height - 2*self.margin) / (max_y-min_y)

        self.x_scale = min(self.y_scale, self.x_scale)
        self.y_scale = min(self.y_scale, self.x_scale)

        self.x_offset = - min_x
        self.y_offset = (self.height - self.y_scale * max_y) / self.y_scale

    def clear_path(self):
        self.path = []
        self.i_rc_node_id = None
        self.f_rc_node_id = None

    def pixels_to_coords(self, center):
        i, j = center
        x = (i - self.margin)/self.x_scale - self.x_offset
        y = -(j - self.height - self.margin)/self.y_scale - self.y_offset
        return (x, y)

    def coord_to_pixels(self, center):
        x, y = center
        i = (x + self.x_offset) * self.x_scale + self.margin
        j = self.height - ((y + self.y_offset) * self.y_scale) + self.margin
        return (int(i), int(j))

    def run_gui(self):
        while(1):
            img = self.draw()
            cv.imshow('image', img)
            if cv.waitKey(20) & 0xFF == 27:
                break
        cv.imwrite('/home/lorenzo/Documents/PAPERS/IROS2022/figures/topological_map.jpg', np.array(img*255,dtype=np.int))

    def mouse_callback(self, event, xp, yp, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.i_lc_node = self.get_node_id_from_click((xp, yp))

        if event == cv.EVENT_LBUTTONUP:
            if self.i_lc_node != None:
                self.f_lc_node = self.get_node_id_from_click((xp, yp))
                if self.i_lc_node != self.f_lc_node and self.f_lc_node != None:
                    self.graph.add_connection(self.i_lc_node, self.f_lc_node)

        if event == cv.EVENT_LBUTTONDBLCLK:
            self.doubleclick((xp, yp))

        if event == cv.EVENT_RBUTTONDBLCLK:
            self.double_right_click((xp, yp))

    def get_node_id_from_click(self, p1):
        for node in self.graph.nodes:
            assert(isinstance(node, Node))
            p2 = self.coord_to_pixels(node.center)
            d = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if d < self.radius:
                return node.id
        else:
            return None

    def double_right_click(self, location):
        if self.path.__len__() != 0:
            self.clear_path()
        node_id = self.get_node_id_from_click(location)
        if node_id != None:
            if self.i_rc_node_id == None:
                self.i_rc_node_id = node_id
                print("Origin_set")
            elif self.f_rc_node_id == None:
                if self.i_rc_node_id != node_id:
                    self.f_rc_node_id = node_id
                    print("End set")
                    self.path = self.graph.generate_path(
                        self.i_rc_node_id, self.f_rc_node_id)
                    print(self.graph.generate_instructions_from_path(self.path))

    def doubleclick(self, position):
        to_delete = self.get_node_id_from_click(position)
        if to_delete == None:
            self.graph.add_node(self.pixels_to_coords(position))
        else:
            self.delete_node(to_delete)

    def draw(self):
        image = np.ones((self.width, self.height, 3))*COLORS["background"]
        for node in self.graph.nodes:
            cv.circle(image, self.coord_to_pixels(
                node.center), self.radius, COLORS["base_node_color"], -1)
        for connection in self.graph.connections:
            cv.line(image, self.coord_to_pixels(self.graph.get_node_by_id(connection[0]).center), self.coord_to_pixels(self.graph.get_node_by_id(
                connection[1]).center), COLORS["base_node_color"], thickness=2)

        prev_node = None
        # Draw path lines
        for i, node_id in enumerate(self.path):
            node = self.graph.get_node_by_id(node_id)
            if prev_node != None:
                cv.line(image, self.coord_to_pixels(node.center), self.coord_to_pixels(prev_node.center),
                        COLORS["base_path_color"], thickness=5)
            prev_node = node
        # Draw path nodes
        for i, node_id in enumerate(self.path):
            if i == 0:
                node_color = COLORS["initial_node_color"]
            elif i == self.path.__len__()-1:
                node_color = COLORS["final_node_color"]
            else:
                node_color = COLORS["base_path_color"]
            node = self.graph.get_node_by_id(node_id)


            cv.circle(image, self.coord_to_pixels(
                node.center), self.radius, node_color, -1)

        return image/255*0.9


if __name__ == "__main__":
    graph_gui = GraphDrawing()
    graph_gui.set_graph
    graph_gui.run_gui()
    # graph_gui.save("/home/lorenzo/catkin_ws/src/mine_topographic_map/saved_maps/saved_graph_.pkl")
