import random
from matplotlib import pyplot as plt
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely import affinity
from scipy.spatial.transform import Rotation
import os.path


ALIAS = {
    "tunnel_block": "tunnel_tile_blocker",
    "tunnel_rect": "tunnel_tile_5",
    "tunnel_t": "tunnel_intersection_t",
    "tunnel_4_way_intersection": "tunnel_tile_1",
    "tunnel_curve": "tunnel_tile_2"
}

############################################################################################################################
#	Loading of the yaml file with the info about the tiles
############################################################################################################################
PATH_TO_YAML = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/tile_definitions.yaml")
tile_definitions = {}

with open(PATH_TO_YAML, "r") as f:
    raw_yaml = yaml.safe_load_all(f)
    for doc in raw_yaml:
        if type(doc) == dict:
            tile_definitions[doc["model_name"]] = doc

####################################################################################################################################
####################################################################################################################################
#		CLASSES DEFINITIONS
####################################################################################################################################
####################################################################################################################################

# --------------------------------------------------------------------------------------------------------------------------
#	 definition of the Tile class
# --------------------------------------------------------------------------------------------------------------------------


class Tile:
    CD = tile_definitions

    def __init__(self, i_type, i_scale=1, tree=None):
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

    def reset_connections(self):
        self.connections = [None for _ in range(
            len(self.params["connection_points"]))]

    @property
    def uri(self):
        return self.params["model_name"]

    @classmethod
    def scale(cls, new_scale):
        if new_scale != 1:
            for tile_type in cls.CD.keys():
                params = cls.CD[tile_type]
                k = "connection_points"
                for i in range(len(params[k])):
                    params[k][i][0] *= new_scale
                    params[k][i][1] *= new_scale
                    params[k][i][2] *= new_scale
                for k in ["bounding_boxes", "exits", "areas"]:
                    params[k] = recursive_scaling(new_scale, params[k])
                k = "model_name"
                params[k] = str(new_scale) + params[k]

    def connect_and_move(self, t2, nc2, nc):
        '''Connects this tile to the parent tile. The parent tile must be a Tile instance.
        After connecting them, it updates the position of this tile so that the connection 
        is possible'''
        # Establish the connections
        self.connect(t2, nc2, nc)
        self.move_to_connection(t2, nc2, nc)

    def connect(self, t2, nc2, nc1):
        '''Connects this tile to another. The other tile tile must be a Tile instance.'''
        self.connections[nc1] = t2
        t2.connections[nc2] = self

    def move_to_connection(self, t2, nc2, nc1):
        '''updates the position of this tile so that the connection 
        is possible'''
        # Calculate the transformation of the child exit from its current position
        # to its final position. The final position is the parents initial position
        # but rotated pi rad in the Z axis.
        from_world_to_exit = t2.connection_points[nc2].op_dir_mat()
        from_exit_to_center = np.linalg.inv(
            self.connection_points[nc1].C_T_M)
        from_world_to_center = from_world_to_exit * from_exit_to_center

        # Apply the transformation
        self.move(T=from_world_to_center)

    def disconnect(self, other_tile):
        self.connections[self.connections.index(other_tile)] = None
        other_tile.connections[self.connections.index(self)] = None

    @property
    def T_M_flatten(self):
        return list(np.array(self.T_M[:3, :3]).flatten()) + list(np.array(self.T_M[:3, -1]).flatten())

    @property
    def xyzrot(self):
        return TM_to_xyzrot(self.T_M)

    @property
    def xyz(self):
        return self.T_M[:3, -1]

    def move(self, params=None, T=None):
        '''Params is a [x,y,z,roll,pitch,yaw] vector.
        T_M is directly the new Transformation Matrix'''
        if params != None:
            self.T = xyzrot_to_TM(params)
        if type(T) != type(None):
            self.T_M = T

    @property
    def empty_connections(self):
        return [nc for nc, c in enumerate(self.connections) if c is None]

    @property
    def neighbors(self):
        return [c for nc, c in enumerate(self.connections) if c is not None]

    def distance(self, other_tile):
        return np.math.sqrt(np.sum(np.square(self.xyz-other_tile.xyz)))
    @property
    def n_connections(self):
        return len(self.connections)
# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the ChildGeometry class
# --------------------------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the ConnectionPoint class
# --------------------------------------------------------------------------------------------------------------------------------------


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

    @property
    def xyz(self):
        return self.T_M[:3, -1]

    def distance(self, other_connection):
        return np.math.sqrt(np.sum(np.square(self.xyz-other_connection.xyz)))

# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the Area class
# --------------------------------------------------------------------------------------------------------------------------------------


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
    


# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the BoundingBox class
# --------------------------------------------------------------------------------------------------------------------------------------


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

############################################################################################################################
############################################################################################################################
#		FUNCTIONS
############################################################################################################################
############################################################################################################################

############################################################################################################################
#	Geometry functions
############################################################################################################################

#	 definition of the xyzrot_to_TM function
# --------------------------------------------------------------------------------------------------------------------------------------


def xyzrot_to_TM(xyzrot):
    assert len(xyzrot) == 6
    r = np.matrix(Rotation.from_euler("xyz", xyzrot[-3:]).as_dcm())
    p = np.matrix(xyzrot[:3]).T
    return np.vstack([np.hstack([r, p]), np.matrix([0, 0, 0, 1])])

#	 definition of the TM_to_xyzrot function
# --------------------------------------------------------------------------------------------------------------------------------------


def TM_to_xyzrot(TM):
    r = list(np.array(Rotation.from_dcm(TM[:3, :3]).as_euler("xyz")).flatten())
    p = list(np.array(TM[:3, -1]).flatten())
    return p + r


#	 definition of the scale_geom function
# --------------------------------------------------------------------------------------------------------------------------------------
def scale_geom(geom, scale):
    return affinity.scale(geom,
                          xfact=scale,
                          yfact=scale,
                          zfact=scale,
                          origin=(0, 0, 0))

#	 definition of the transform_point function
# --------------------------------------------------------------------------------------------------------------------------------------


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

############################################################################################################################
#	Data treatment functions
############################################################################################################################

#	 definition of the transform_point function
# --------------------------------------------------------------------------------------------------------------------------------------


def recursive_scaling(scale, iterable):
    for i, element in enumerate(iterable):
        if type(element) == list:
            element = recursive_scaling(scale, element)
        else:
            iterable[i] *= scale
    return iterable

#	 definition of the transform_point function
# --------------------------------------------------------------------------------------------------------------------------------------


def close_list_of_points(list_of_points: np.ndarray):
    '''Mainly for plotting purposes, adds the first element to
    the end of the list so a closing segment is plotted with 
    matplotlib.pyplot.plot()'''
    new_line = list_of_points[[0], :]
    return np.vstack([list_of_points, new_line])


def get_random_tile():
    return Tile(random.choice(list(Tile.CD.keys())))

def get_random_non_blocking_tile():
    no_block_list = list(Tile.CD.keys())
    no_block_list.remove(ALIAS["tunnel_block"])
    return Tile(random.choice(no_block_list))
############################################################################################################################
#	Plotting Functions
############################################################################################################################

# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the MinBorders class
# --------------------------------------------------------------------------------------------------------------------------------------
class MinBorders:
    '''Class that keeps track of the highest and lowest coordinates
    in a sequence of tiles for plotting purposes'''

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

    def update_with_other_instance(self, other_instance):
        self.min_x = min(self.min_x, other_instance.min_x)
        self.min_y = min(self.min_y, other_instance.min_y)
        self.max_x = max(self.max_x, other_instance.max_x)
        self.max_y = max(self.max_y, other_instance.max_y)

    @property
    def borders(self):
        '''Returns the max and min coordinates that should be assigned 
        of the plotting axis so that the whole tree fits'''
        return self.min_x-1, self.min_y-1, self.max_x+1, self.max_y+1

#	 definition of the plot_tile function
# --------------------------------------------------------------------------------------------------------------------------------------


def plot_tile(tile, bounding_boxes=True, areas=True, exits=True, connections=True):
    '''Takes a tile as input and sents de matplotlib commands to plot the different 
    components.'''
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

    return min_borders

#	 definition of the plot_seq_of_tiles function
# --------------------------------------------------------------------------------------------------------------------------------------


def plot_seq_of_tiles(seq_of_tiles, bounding_boxes=True, areas=True, exits=True, connections=True):
    plt.gca().clear()
    for idx, tile in enumerate(seq_of_tiles):
        if idx == 0:
            borders = plot_tile(
                tile, bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections)
        else:
            borders.update_with_other_instance(plot_tile(
                tile, bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections))

    min_x, min_y, max_x, max_y = borders.borders

    size_x = max_x - min_x
    size_y = max_y - min_y

    final_size = max(size_x, size_y)

    plt.gca().set_xlim(min_x, min_x+final_size)
    plt.gca().set_ylim(min_y, min_y+final_size)
    plt.draw()
