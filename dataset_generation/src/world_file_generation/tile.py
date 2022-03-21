from matplotlib import pyplot as plt
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely import affinity
from scipy.spatial.transform import Rotation

##############################################################
#	Loading of the yaml file with the info about the tiles
##############################################################

path_to_yaml_file = "/home/lorenzo/catkin_ws/src/under-topo-nav/dataset_generation/src/world_file_generation/data_files/tile_definitions.yaml"
tile_definitions = {}
with open(path_to_yaml_file, "r") as f:
    raw_yaml = yaml.safe_load_all(f)
    for doc in raw_yaml:
        if type(doc) == dict:
            tile_definitions[doc["model_name"]] = doc

# -------------------------------------------------------------------
#	 definition of the  class
# -------------------------------------------------------------------


class Tile:
    CD = tile_definitions

    def __init__(self, i_type, i_scale=1):
        self.params = self.CD[i_type]
        self.scale(i_scale)
        # Check parameters
        assert len(self.params["exits_of_each_area"]
                   ) == len(self.params["areas"])

        # Initialise all parameters that must change if the tile is moved
        self.T_M = np.matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # Initialize connection points
        self.connection_points = []
        for i in range(len(self.params["connection_points"])):
            self.connection_points.append(ConnectionPoint(self, i))
        # Initialise bounding boxes
        self.bounding_boxes = []
        for i in range(len(self.params["bounding_boxes"])):
            self.bounding_boxes.append(BoundingBox(self, i))
        # Initialise areas
        self.areas = []
        for i in range(len(self.params["areas"])):
            self.areas.append(Area(self, i))

        # Initialise an empty list of connections
        self.connections = [None for _ in range(
            len(self.params["connection_points"]))]

    @classmethod
    def scale(cls, new_scale):
        # The connection points
        for tile_type in cls.CD.keys():
            params = cls.CD[tile_type]
            k = "connection_points"
            for i in range(len(params[k])):
                params[k][i][0] *= new_scale
                params[k][i][1] *= new_scale
                params[k][i][2] *= new_scale
            for k in ["bounding_boxes", "exits", "areas"]:
                params[k] = recursive_scaling(new_scale, params[k])

    def connect(self, parent_tile, connection, parent_connection):
        assert type(parent_tile) == Tile

        # Establish the connections
        self.connections[connection] = parent_tile
        parent_tile.connections[parent_connection] = self

        # Calculate the transformation of the child exit from its current position
        # to its final position. The final position is the parents initial position
        # but rotated pi rad in the Z axis.
        from_world_to_exit = parent_tile.connection_points[parent_connection].op_dir_mat(
        )
        from_exit_to_center = np.linalg.inv(
            self.connection_points[connection].C_T_M)
        from_world_to_center = from_world_to_exit * from_exit_to_center

        # Apply the transformation
        self.move(T=from_world_to_center)

    @property
    def T_M_flatten(self):
        return list(np.array(self.T_M[:3, :3]).flatten()) + list(np.array(self.T_M[:3, -1]).flatten())

    def move(self, params=None, T=None):
        '''Params is a [x,y,z,roll,pitch,yaw] vector.
        T_M is directly the new Transformation Matrix'''
        if params != None:
            self.T = xyzrot_to_TM(params)
        if type(T) != type(None):
            self.T_M = T


class ChildGeometry:
    def __init__(self, parent, idx):
        self.parent = parent
        self.idx = idx

    @property
    def P_T_M(self):
        '''Returns the transformation matrix from the parent'''
        return self.parent.T_M

    def params(self, key):
        return self.parent.params[key][self.idx]


class ConnectionPoint(ChildGeometry):
    key = "connection_points"

    @property
    def C_T_M(self):
        '''Returns the Transformation matrix
         from the tile center to the exit'''
        try:
            return self._C_T_M
        except:
            self._C_T_M = xyzrot_to_TM(self.params(self.key))
            return self._C_T_M

    @property
    def T_M(self):
        return self.P_T_M * self.C_T_M

    def op_dir_mat(self):
        '''Returns the global transformation matrix that 
        an exit connecting to this one must have'''
        return self.T_M * xyzrot_to_TM([0, 0, 0, 0, 0, np.math.pi])

    @property
    def x(self):
        return self.T_M[0, -1]

    @property
    def y(self):
        return self.T_M[1, -1]

    @property
    def z(self):
        return self.T_M[2, -1]


class Area(ChildGeometry):
    area_key = "areas"
    exit_key = "exits"
    connection_areas_key = "connection_exits"
    exit_relation_key = "exits_of_each_area"

    @property
    def raw_exits(self):
        '''Returns the exits for this area before moving the tile
        as a an Nx3 array, N being the number of exits'''
        raw_exts = np.reshape(
            np.array(self.parent.params[self.exit_key]), [-1, 3])
        return raw_exts[self.params(self.exit_relation_key), :]

    @property
    def n_exits(self):
        return len(self.raw_exits)

    @property
    def exits(self):
        '''Returns the exits of the area after moving the tile as 
            an Nx3 array, N being the number of exits'''
        exits = np.zeros([self.n_exits, 3])
        for idx, ext in enumerate(self.raw_exits):
            exits[idx, :] = transform_point(ext, self.P_T_M)
        return exits

    @property
    def raw_area_points(self):
        '''Returns the area points before moving the tile as a 
        Nx3 array, N being the number of points in the area'''
        return np.array(self.params(self.area_key))

    @property
    def n_area_points(self):
        return len(self.raw_area_points)

    @property
    def area_points(self):
        '''Returns the area points after moving the tile as a
        Nx3 array, N being the number of points in the area'''
        points = np.zeros([self.n_area_points, 3])
        for idx, point in enumerate(self.raw_area_points):
            points[idx, :] = transform_point(point, self.P_T_M)
        return points

    def as_polygon(self):
        return Polygon(self.area_points)


class BoundingBox(ChildGeometry):
    perimeter_key = "bounding_boxes"

    @property
    def raw_perimeter_points(self):
        '''Returns the perimeter points before moving the tile as a 
        Nx3 array, N being the number of points in the perimeter'''
        return np.array(self.params(self.perimeter_key))

    @property
    def n_perimeter_points(self):
        return len(self.raw_perimeter_points)

    @property
    def perimeter_points(self):
        '''Returns the perimeter points after moving the tile as a
        Nx3 array, N being the number of points in the perimeter'''
        points = np.zeros([self.n_perimeter_points, 3])
        for idx, point in enumerate(self.raw_perimeter_points):
            points[idx, :] = transform_point(point, self.P_T_M)
        return points

    def as_polygon(self) -> Polygon:
        return Polygon(self.perimeter_points)

##################################################################
##################################################################
#		FUNCTIONS
##################################################################
##################################################################

##############################################################
#	Geometry functions
##############################################################


def xyzrot_to_TM(xyzrot):
    assert len(xyzrot) == 6
    r = np.matrix(Rotation.from_euler("xyz", xyzrot[-3:]).as_dcm())
    p = np.matrix(xyzrot[:3]).T
    return np.vstack([np.hstack([r, p]), np.matrix([0, 0, 0, 1])])


def scale_geom(geom, scale):
    return affinity.scale(geom,
                          xfact=scale,
                          yfact=scale,
                          zfact=scale,
                          origin=(0, 0, 0))


def transform_point(point, T):
    '''Takes a transformation matrix and a point 
    represented as a 3 or 4 element list or array and returns 
    a 3-element array with the transformed point'''
    if len(point) == 3:
        if isinstance(point, np.ndarray):
            point = np.append(point, [1])
        elif isinstance(point, list):
            point.append(1)
    point = np.matrix(point).T    
    transformed_point = (T * point)
    return np.array(transformed_point[:3]).flatten()

##############################################################
#	Data treatment functions
##############################################################


def recursive_scaling(scale, iterable):
    for i, element in enumerate(iterable):
        if type(element) == list:
            element = recursive_scaling(scale, element)
        else:
            iterable[i] *= scale
    return iterable


def close_list_of_points(list_of_points: np.ndarray):
    '''Mainly for plotting purposes, adds the first element to
    the end of the list so a closing segment is plotted with 
    matplotlib.pyplot.plot()'''
    new_line = list_of_points[[0], :]
    return np.vstack([list_of_points, new_line])


##############################################################
#	Plotting Functions
##############################################################

class MinBorders:
    def __init__(self, points: np.ndarray):
        self.min_x = np.min(points[:, 0])
        self.min_y = np.min(points[:, 1])
        self.max_x = np.max(points[:, 0])
        self.max_y = np.max(points[:, 1])

    def update_with_points(self, points: np.ndarray):
        self.min_x = min(self.min_x, np.min(points[:, 0]))
        self.min_y = min(self.min_y, np.min(points[:, 1]))
        self.max_x = max(self.max_x, np.max(points[:, 0]))
        self.max_y = max(self.max_y, np.max(points[:, 1]))

    def update_with_values(self, x, y):
        self.min_x = min(self.min_x, x)
        self.min_y = min(self.min_y, y)
        self.max_x = max(self.max_x, x)
        self.max_y = max(self.max_y, y)

    @property
    def borders(self):
        return self.min_x-1, self.min_y-1, self.max_x+1, self.max_y+1


def plot_tile(tile, bounding_boxes=True, areas=True, exits=True, connections=True):
    assert isinstance(tile, Tile)
    if bounding_boxes:
        for bb in tile.bounding_boxes:
            assert isinstance(bb, BoundingBox)
            points = bb.perimeter_points
            points = close_list_of_points(points)
            plt.plot(points[:, 0], points[:, 1], c="b")
            min_borders = MinBorders(points)
    if areas:
        for area in tile.areas:
            assert isinstance(area, Area)
            points = area.area_points
            points = close_list_of_points(points)
            plt.plot(points[:, 0], points[:, 1], c="r")
            min_borders.update_with_points(points)
            if exits:
                points = area.exits
                points = close_list_of_points(points)
                plt.scatter(points[:, 0], points[:, 1], c="g")
                min_borders.update_with_points(points)
    if connections:
        for cnp in tile.connection_points:
            assert isinstance(cnp, ConnectionPoint)
            plt.scatter(cnp.x, cnp.y, c="k")
            min_borders.update_with_values(cnp.x, cnp.y)

    return min_borders.borders


def plot_seq_of_tiles(seq_of_tiles, bounding_boxes=True, areas=True, exits=True, connections=True):
    for idx, tile in enumerate(seq_of_tiles):
        if idx == 0:
            min_x, min_y, max_x, max_y = plot_tile(
                tile, bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections)
        else:
            minx, miny, maxx, maxy = plot_tile(
                tile, bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections)
            min_x = min(min_x, minx)
            min_y = min(min_y, miny)
            max_x = max(max_x, maxx)
            max_y = max(max_y, maxy)

    size_x = max_x - min_x
    size_y = max_y - min_y

    final_size = max(size_x, size_y)

    plt.gca().set_xlim(min_x, min_x+final_size)
    plt.gca().set_ylim(min_y, min_y+final_size)
