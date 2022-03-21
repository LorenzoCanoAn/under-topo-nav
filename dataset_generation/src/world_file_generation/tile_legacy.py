from logging import error
from math import ceil
import math
import random
from time import time
from typing import Tuple
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import affinity
from scipy.spatial.transform import Rotation
import pickle
import geometry_msgs.msg as ros_geom_msgs
from scipy.stats import norm
from matplotlib import pyplot as plt
import tf.transformations
import os

TILE_DEFINITION_FILE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/text files/tile_definition.txt"
BOILERPLATE_FILE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/text files/boilerplate.txt"
OBSTACLE_DEFINITION_FILE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/scripts/world_file_generation/text files/obstacle.txt"

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A, B, C, D):
    line1 = LineString([A, B])
    line2 = LineString([C, D])
    return line1.intersects(line2)


def iterative_scale(object_to_scale, scale):
    if type(object_to_scale) in [list, tuple]:
        object_to_scale = list(object_to_scale)
        for i, element in enumerate(object_to_scale):
            element = iterative_scale(element, scale)
            object_to_scale[i] = element
        return object_to_scale
    else:
        return object_to_scale * scale


class TileTree:
    TYPE_DICT = {}

    def __init__(self, name, scale=1.0):
        self.tiles: list[Tile] = []
        self.name = name
        self.scale = scale
        self.scale_tiles()

    def scale_tiles(self):
        for key in self.TYPE_DICT.keys():
            self.TYPE_DICT[key].scale(self.scale)

    def add_tile(self, type, parent, p_connection, c_connection, is_root=False):
        self.tiles.append(self.TYPE_DICT[type](self, self.tiles.__len__(
        ), parent, p_connection, c_connection, is_root=is_root))

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump(self, f)

    def load(self, path):
        self.tiles = pickle.load(open(path, "rb")).tiles

    def plot(self):
        for tile in self.tiles:
            for segment in tile.t_plot_lines:
                x = []
                y = []
                for point in segment:
                    x.append(point[0])
                    y.append(point[1])
                plt.plot(x, y, color=tile.color)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def to_world_file_text(self, path, n_obstacles_per_tile = 0):
        complete_text = ""
        with open(TILE_DEFINITION_FILE_PATH, 'r') as f:
            raw_text = f.read()

        for i, t in enumerate(self.tiles):
            # ADD THE TILE
            assert(isinstance(t, Tile))
            text_pose = "{} {} {} {} {} {}".format(
                t.pose[0], t.pose[1], t.pose[2], 0, 0, t.rotation)
            text_filled = raw_text.format(t.name, text_pose, t.model_name)
            complete_text += text_filled

            # ADD THE OBSTACLES
            with open(OBSTACLE_DEFINITION_FILE_PATH, 'r') as f:
                obstacle_raw_test = f.read()
            if i >= 2:
                n_obstacles_per_tile_ = min(n_obstacles_per_tile, t.areas.__len__())
                areas = random.sample(t.areas,n_obstacles_per_tile_)
                for area in areas:
                    pose = area.gen_rand_pose()
                    text_pose = "{} {} {} {} {} {}".format(pose[0], pose[1], t.pose[2] + 0.2, 0, 0, random.uniform(0,math.pi))
                    
                    text_filled = obstacle_raw_test.format(str(time()), text_pose)
                    complete_text += text_filled

        with open(BOILERPLATE_FILE_PATH, 'r') as f:
            final_text = f.read()
        final_text = final_text.format(complete_text)

        with open(path, "w+") as f:
            f.write(final_text)

    def check_collisions(self):
        for tile_ in self.tiles:
            segments_ = tile_.t_plot_lines
            for tile__ in self.tiles:
                if tile_.id == tile__.id:
                    continue
                if tile__.id in tile_.connections:
                    continue
                if tile_.id in tile__.connections:
                    continue
                segments__ = tile__.t_plot_lines
                for segment_ in segments_:
                    for segment__ in segments__:
                        if intersect(segment_[0], segment_[1], segment__[0], segment__[1]):
                            return True
        return False


def cos(x):
    return np.cos(x).item(0)


def sin(x):
    return np.sin(x).item(0)


def rotate_vector(vector, angle):
    if vector.__len__() == 2:
        x = vector[0]
        y = vector[1]

        n_x = x*cos(angle) - y*sin(angle)
        n_y = x*sin(angle) + y*cos(angle)
        return (n_x, n_y)
    elif vector.__len__() == 3:
        x = vector[0]
        y = vector[1]

        n_x = x*cos(angle) - y*sin(angle)
        n_y = x*sin(angle) + y*cos(angle)
        return (n_x, n_y, vector[2])


def rotate_list_of_tuple_points(list_of_points, angle):
    out_list = []
    for point in list_of_points:
        if isinstance(point[0], Tuple):
            rotated = rotate_list_of_tuple_points(point)
            out_list.append(rotated)
        else:
            rotated = affinity.rotate(
                Point(point[0], point[1]), angle, origin=(0, 0))
            out_list.append((rotated.x, rotated.y))

    return tuple(out_list)



class Tile:
    def __init__(self, tree: TileTree, id, parent_id, p_connection, c_connection, is_root=False):
        self.tree = tree
        self.init_parameters()

        self.is_root = is_root
        self.id = id
        self.name = "tile_{}".format(id)
        self.parent_id = parent_id
        self.init_connections()
        if is_root:
            self.pose = (0., 0., 0.)
            self.rotation = float(0)
        else:
            # el hijo conoce a su padre
            self.add_connection(c_connection, parent_id)
            self.tree.tiles[parent_id].add_connection(
                p_connection, id)  # el padre conoce a su hijo
            self.compute_pose()

    def init_connections(self):
        self.connections = [None for _ in range(self.n_connections)]

    def add_connection(self, connection, connected):
        if type(self.connections[connection]) == type(None):
            self.connections[connection] = connected
        else:
            error("Trying to overwrite a connection: self: {} parent: {} conncection: {} connected: {}".format(
                self.id, self.parent_id, connection, connected))
            exit()

    def as_pose_msg(self):
        msg = ros_geom_msgs.Pose()
        msg.position.x = self.pose[0]
        msg.position.y = self.pose[1]
        msg.position.z = self.pose[2]
        quaternion = tf.transformations.quaternion_from_euler(
            0, 0,  self.rotation)
        msg.orientation.x = quaternion[0]
        msg.orientation.y = quaternion[1]
        msg.orientation.z = quaternion[2]
        msg.orientation.w = quaternion[3]
        return msg

    def get_id(self):
        return self.id

    def get_areas(self):
        return self.areas

    def find_connection_index(self, tile_id):
        for i, c in enumerate(self.connections):
            if c == tile_id:
                return i

    def find_connection_vector(self, tile_id) -> tuple:
        return self.connection_possitions[self.find_connection_index(tile_id)]

    def compute_pose(self):
        p_1 = self.tree.tiles[self.parent_id].pose
        theta_1 = self.tree.tiles[self.parent_id].rotation

        t_1 = self.tree.tiles[self.parent_id].find_connection_vector(self.id)
        t_2 = self.find_connection_vector(self.parent_id)

        phi_1 = t_1[3]
        phi_2 = t_2[3]

        t_1 = (t_1[0], t_1[1], t_1[2])
        t_2 = (t_2[0], t_2[1], t_2[2])

        ang_t_1 = (theta_1 + phi_1) % (np.pi*2)
        ang_t_2 = (ang_t_1 + np.pi) % (np.pi*2)

        self.rotation = (ang_t_2 - phi_2 + np.pi*2) % (np.pi*2)

        true_t1 = rotate_vector(t_1, theta_1)
        true_t2 = rotate_vector(t_2, self.rotation)

        x_2 = p_1[0] + true_t1[0] - true_t2[0]
        y_2 = p_1[1] + true_t1[1] - true_t2[1]
        z_2 = p_1[2] + true_t1[2] - true_t2[2]

        self.pose = (x_2, y_2, z_2)

        for area in self.areas:
            area.transform()
        self.transform_plot_lines()

    def gen_rand_pose_msgs_and_vector(self, n=10):
        poses = []
        vectors = []
        n_areas = self.areas.__len__()
        for area in self.areas:
            for _ in range(int(ceil(n/n_areas))):
                pose, vector = area.gen_rand_pose_msg_and_vector()
                poses.append(pose)
                vectors.append(vector)
        return poses, vectors

    def get_numeric_label(self):
        return self.numeric_label

    def init_parameters(self):
        self.model_name = "model://" + str(self.tree.scale) + self.MODEL_NAME
        self.numeric_label = self.__class__.__base__.__subclasses__().index(type(self))
        self.connection_possitions = self.p_connection_possitions
        self.n_connections = self.connection_possitions.__len__()
        self.label = self.__class__.__name__

        n = 0

        self.areas = [Area(self, n, a, i) for n, (a, i) in enumerate(zip(
            self.areas_points, self.areas_exits))]
        self.t_plot_lines = self.plot_segments

    def transform_plot_lines(self):
        self.t_plot_lines = []
        for line in self.plot_segments:
            t_point = []
            for point in line:
                p = Point(point[0], point[1])
                p = affinity.rotate(
                    p, self.rotation, origin=(0, 0), use_radians=True)
                p = affinity.translate(p, xoff=self.pose[0], yoff=self.pose[1])
                t_point.append((p.x, p.y))

            self.t_plot_lines.append(t_point)

    @classmethod
    def plot(self):
        f = plt.figure()
        for e in self.es:
            x = []
            y = []
            for s in e:
                x.append(s[0])
                y.append(s[1])
            plt.plot(x, y, "b")

        for e in self.areas_points:
            x = []
            y = []
            for s in e:
                x.append(s[0])
                y.append(s[1])
            for s in e:
                x.append(s[0])
                y.append(s[1])
                break
            plt.plot(x, y, "g")
        plt.gca().set_aspect("equal")
        plt.gca().set_xlim(-10, 10)
        plt.gca().set_ylim(-10, 10)
        plt.show()

    @classmethod
    def scale(self, scale):
        self.p_connection_possitions= iterative_scale(self.p_connection_possitions, scale)
        for connection in self.p_connection_possitions:
            connection[-1] /=scale
        self.plot_segments= iterative_scale(self.plot_segments, scale)
        self.es= iterative_scale(self.es, scale)
        self.areas_points= iterative_scale(self.areas_points, scale)


class Exit:
    def __init__(self, area, point1: Point, point2: Point):
        self.area = area
        self.o_point1 = point1
        self.o_point2 = point2
        self.t_point1 = point1
        self.t_point2 = point2

    def transform(self):
        rotation = self.area.tile.rotation
        translation_x = self.area.tile.pose[0]
        translation_y = self.area.tile.pose[1]

        p1 = affinity.rotate(self.o_point1, rotation, (0, 0), use_radians=True)
        p1 = affinity.translate(p1, xoff=translation_x, yoff=translation_y)
        p2 = affinity.rotate(self.o_point2, rotation, (0, 0), use_radians=True)
        p2 = affinity.translate(p2, xoff=translation_x, yoff=translation_y)

        self.t_point1 = p1
        self.t_point2 = p2


class Area:
    def __init__(self, tile: Tile, index, perimeter_points, exits):
        self.tile = tile
        self.label = tile.numeric_label
        self.o_polygon = self.as_polygon(perimeter_points)
        self.t_polygon = self.as_polygon(perimeter_points)
        self.index = index
        self.exits_own = [Exit(self, Point(self.tile.es[e][0][0], self.tile.es[e][0][1]),
                               Point(self.tile.es[e][1][0], self.tile.es[e][1][1]))
                          for e in exits]

    def as_polygon(self, points):
        if points.__len__() == 1:
            (x, y) = points[0]
            p = Polygon([(+ x, + y),
                         (+ x, - y),
                         (- x, - y),
                         (- x, + y)])
        else:
            p = Polygon(points)

        return p

    def transform(self):
        rotation = self.tile.rotation
        translation_x = self.tile.pose[0]
        translation_y = self.tile.pose[1]

        p = affinity.rotate(self.o_polygon, rotation, (0, 0), use_radians=True)
        p = affinity.translate(p, xoff=translation_x, yoff=translation_y)
        self.t_polygon = p
        for e in self.exits_own:
            e.transform()

    def gen_normal_distribution(self, n_elements):
        nd = norm(loc=2, scale=0.5)
        n = 0
        array = []
        for i in np.arange(0, 4, 4/n_elements):
            if n == n_elements:
                break
            n += 1
            array.append(nd.pdf(i))

        return array

    def calc_ang_vector_from_robot_pose(self, pose: ros_geom_msgs.Pose):
        o = pose.orientation
        euler_rep = quaternion_to_euler(o.x, o.y, o.z, o.w, degrees=False)
        robot_angle = euler_rep.item(2) * 180 / np.math.pi
        exits_all = []
        for e in self.exits_own:
            exits_all.append(e)

        # Check if this area is a connection area
        if self.index in self.tile.exit_to_areas_relation:
            # find out to which connection this area is related
            connection = self.tile.exit_to_areas_relation.index(self.index)
            # find the tile connected to this area's connection
            connected_tile_id = self.tile.connections[connection]
            connected_tile = self.tile.tree.tiles[connected_tile_id]
            if connected_tile.__class__ == EndOfGallery:
                exits_all.pop(0)
            # find the connection in the connected tile that connects to this tile
            #connected_tile_connection = connected_tile.connections.index(self.tile.id)
            # find the area associated with the tile's connection

            # connected_tile_area_index = connected_tile.exit_to_areas_relation[
            #    connected_tile_connection]
            #connected_area = connected_tile.areas[connected_tile_area_index]
            # for e in connected_area.exits_own:
            #    exits_all.append(e)

        angle_array_o = np.zeros(360)
        for ext in exits_all:
            r_angles = []
            assert(isinstance(ext, Exit))
            for t_point in [ext.t_point1, ext.t_point2]:
                exit_as_numpy = np.array([t_point.x, t_point.y])
                robot_position = np.array([pose.position.x, pose.position.y])
                geometric_vector = exit_as_numpy - robot_position
                angle = (
                    np.math.atan2(geometric_vector.item(1),
                                  geometric_vector.item(0))
                    * 180
                    / np.math.pi
                )  # angle in world coordinates of the vector that goes from the robot to the exit
                r_angles.append(
                    359-int((angle - robot_angle + 360 + 180) % (360)))

            r_angles.sort()

            if r_angles[1] - r_angles[0] < 180:
                center = (r_angles[1] + r_angles[0])/2

            else:
                center = (r_angles[1] + (360-r_angles[1]+r_angles[0])/2) % 359
            aperture = 60
            angle_array = np.zeros(360)
            normal_array = self.gen_normal_distribution(aperture)
            for i in range(aperture):
                index = int(center - aperture/2 + i) % 360

                angle_array[index] = normal_array[i]

            points_to_change = angle_array > angle_array_o
            angle_array_o[points_to_change] = angle_array[points_to_change]

        return angle_array_o

    def gen_rand_pose_msg_and_vector(self):
        min_x, min_y, max_x, max_y = self.t_polygon.bounds
        while 1:
            random_point = Point(
                [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
            )
            if random_point.within(self.t_polygon):
                break

        o = euler_to_quaternion(0, 0, random.uniform(0, 360), degrees=True)
        position = ros_geom_msgs.Point(
            random_point.x, random_point.y, self.tile.pose[2]+0.133)
        orientation = ros_geom_msgs.Quaternion(o[0], o[1], o[2], o[3])
        message = ros_geom_msgs.Pose(position, orientation)
        vector = self.calc_ang_vector_from_robot_pose(message)
        return message, vector

    def gen_rand_pose(self):
        min_x, min_y, max_x, max_y = self.t_polygon.bounds
        while 1:
            random_point = Point(
                [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
            )
            if random_point.within(self.t_polygon):
                break


        return (random_point.x, random_point.y)

class intersection_t(Tile):
    color = "orange"
    MODEL_NAME = "tunnel_intersection_t"
    p_connection_possitions = [(0,     0,      0,      np.pi*3/2),
                               (7.5,   7.5,    0,      0),
                               (-7.5,  7.5,    0,      np.pi)]

    plot_segments = (((2, 0), (2, 7.5)), ((-2, 0), (-2, 7.5)),
                     ((-7.5, 7.5+2), (7.5, 7.5+2)),
                     ((-7.5, 7.5-2), (7.5, 7.5-2)))

    areas_points = []
    areas_exits = []

    es = [((2, 0), (-2, 0)),
          ((7, 9.4), (7, 5.3)),
          ((-7, 9.4), (-7, 5.3)),
          ((2, 4.7), (-2, 4.7)),
          ((2.6, 9.4), (2.6, 5.3)),
          ((-2.6, 9.4), (-2.6, 5.3))]

    # Central area
    areas_points.append([(2.2, 9.4), (2.2, 5.6), (-2.2, 5.6), (-2.2, 9.4)])
    areas_exits = [(3, 4, 5)]

    # asymetric part
    areas_points.append([(1.8, 4), (-1.8, 4), (-1.8, 1), (1.8, 1)])
    areas_exits.append((0, 3))

    # symetric part 1
    areas_points.append([(3, 5.6), (6, 5.6), (6, 9.4), (3, 9.4)])
    areas_exits.append((1, 4))

    # symetric part 2
    areas_points.append([(-3, 5.6), (-6, 5.6), (-6, 9.4), (-3, 9.4)])
    areas_exits.append((2, 5))

    # inter area1
    areas_points.append([(-0.86, 6.35), (0.86, 6.35),
                        (0.86, 4.02), (-0.86, 4.02)])
    areas_exits.append((0, 4, 5))

    # inter area2
    areas_points.append([(1.56, 6.46), (1.56, 8.40), (4, 8.40), (4, 6.46)])
    areas_exits.append((1, 3, 5))

    # inter area3
    areas_points.append(
        [(1.56-5.5, 6.46), (1.56-5.5, 8.40), (4-5.5, 8.40), (4-5.5, 6.46)])
    areas_exits.append((2, 3, 4))

    # In this case, exit 0 goes to area 1, exit 1 goes to area 2 and exit 2 goes
    # to area 3
    exit_to_areas_relation = [1, 2, 3]


class intersection(Tile):
    color = "yellow"
    MODEL_NAME = "tunnel_tile_1"
    p_connection_possitions = [(0,     -10,    0,      np.pi*3/2),
                               (10,    0,      0,      0),
                               (0,     10,     0,      np.pi/2),
                               (-10,   0,      0,      np.pi)
                               ]

    plot_segments = (((0, -10), (0, 10)), ((-10, 0), (10, 0)))

    areas_points = []
    areas_exits = []

    es = [((1.75, -10), (-1.75, -10))]
    es.append(rotate_list_of_tuple_points(es[-1], 90))
    es.append(rotate_list_of_tuple_points(es[-1], 90))
    es.append(rotate_list_of_tuple_points(es[-1], 90))
    es.append(((1.75, -3), (-1.75, -3)))
    es.append(rotate_list_of_tuple_points(es[-1], 90))
    es.append(rotate_list_of_tuple_points(es[-1], 90))
    es.append(rotate_list_of_tuple_points(es[-1], 90))

    # Central Area
    areas_points.append([(1.5, 1.5), (1.5, -1.5), (-1.5, -1.5), (-1.5, 1.5)])
    areas_exits.append((4, 5, 6, 7))

    # New and rotate
    areas_points.append(((1.6, -3.5), (-1.6, -3.5), (-1.6, -7), (1.6, -7)))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_exits.append((0, 4))
    areas_exits.append((1, 5))
    areas_exits.append((2, 6))
    areas_exits.append((3, 7))
    # Transition Areas
    areas_points.append(((-0.7, -1.54), (0.7, -1.54),
                        (0.7, -3.54), (-0.7, -3.54)))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_points.append(rotate_list_of_tuple_points(areas_points[-1], 90))
    areas_exits.append((0, 5, 6, 7))
    areas_exits.append((1, 6, 7, 4))
    areas_exits.append((2, 7, 4, 5))
    areas_exits.append((3, 4, 5, 6))

    exit_to_areas_relation = [1, 2, 3, 4]


class curve(Tile):
    color = "green"
    MODEL_NAME = "tunnel_tile_2"
    p_connection_possitions = [(0,     -10,    0,      np.pi*3/2),
                               (10,    0,      0,      0)]

    plot_segments = (((0, -10), (2.7, -2.7)), ((2.7, -2.7), (10, 0)),)
    areas_points = []
    areas_exits = []

    es = [((2.19, -10), (-2.19, -10))]
    es.append(((10, -2.3), (10, 2.3)))
    es.append(((2.85, -6.17), (-1.14, -4.39)))
    es.append(((5.36, -3.27), (3.15, 0.55)))

    areas_points.append(((1.8, -9.37), (-1.8, -9.37),
                        (-1.18, -5.2), (2.19, -6.56)))
    areas_points.append(((6.08, -2.42), (4.07, 0.72),
                        (8.9, 1.73), (8.9, -1.75)))
    areas_points.append(((2.87, -5.46), (-0.24, -3.77),
                        (2.78, -0.24), (4.57, -3.13)))
    areas_points.append(((3.36, -4.72), (2.11, -7.17),
                        (-1.54, -6.43), (0.00, -3)))
    areas_points.append(((4.44, -3.64), (5.98, -2.35),
                        (4.65, 0.71), (2.60, -0.26)))

    areas_exits.append((0, 2))
    areas_exits.append((1, 3))
    areas_exits.append((2, 3))
    areas_exits.append((0, 3))
    areas_exits.append((1, 2))

    exit_to_areas_relation = [0, 1]







TYPE_DICT = {
    "t":       intersection_t,
    "inter":   intersection,
    "curve":   curve,
    "rect":    rect,
    "block":    EndOfGallery
}

TileTree.TYPE_DICT = TYPE_DICT


if __name__ == "__main__":
    intersection_t.plot()
