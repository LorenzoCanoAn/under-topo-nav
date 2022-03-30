from math import ceil
import random
from matplotlib import pyplot as plt
import numpy as np
import yaml
from shapely.geometry import Polygon
from shapely import affinity
from scipy.spatial.transform import Rotation
import os.path
import os


ALIAS = {
    "tunnel_block": "tunnel_tile_blocker",
    "tunnel_rect": "tunnel_tile_5",
    "tunnel_t": "my_t",
    "tunnel_4_way_intersection": "tunnel_tile_1",
    "tunnel_curve": "tunnel_tile_2",
    "tunnel_wall": "hatch"
}
BLOCK_TILES = {"tunnel_tile_blocker", "hatch"}

############################################################################################################################
#	Loading of the yaml file with the info about the tiles
############################################################################################################################
PATH_TO_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/tile_definition_files")
tile_definitions = {}
files_in_dir = os.listdir(PATH_TO_DIR)
for file in files_in_dir:
    path_to_yaml = os.path.join(PATH_TO_DIR, file)
    with open(path_to_yaml, "r") as f:
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
    _scale = 1
    def __init__(self, i_type):
        self.params = self.CD[i_type]

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

        # Initialise an empty list of connections
        self.connections = [None for _ in range(
            len(self.params["connection_points"]))]
        
        # Initialise the tunnel axis
        self.axis = []
        for i in range(len(self.params["tunnel_axis"])):
            self.axis.append(TunnelAxis(self, i))
    @property
    def is_block(self):
        try:
            return bool(self.params["is_block"])
        except:
            return False

    def reset_connections(self):
        self.connections = [None for _ in range(
            len(self.params["connection_points"]))]

    @property
    def uri(self):
        return "model://" + self.params["model_name"]

    @classmethod
    def scale(cls, new_scale):
        cls._scale = new_scale / cls._scale
        for tile_type in cls.CD.keys():
            params = cls.CD[tile_type]
            k = "connection_points"
            for i in range(len(params[k])):
                params[k][i][0] *= cls._scale
                params[k][i][1] *= cls._scale
                params[k][i][2] *= cls._scale
            for k in ["bounding_boxes"]:
                params[k] = recursive_scaling(cls._scale, params[k])
            for k in ["tunnel_axis"]:
                params[k] = recursive_scaling(cls._scale, params[k])
           

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

        for cnp in self.connection_points:
            cnp.recalculate = True

        for bb in self.bounding_boxes:
            bb.recalculate = True

        for axs in self.axis:
            axs.recalculate = True

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
# --------------------------------------------------------------------------------------------------------------------------------------


class ChildGeometry:
    def __init__(self, parent, idx):
        self.parent = parent
        self.idx = idx
        self.recalculate = True

    @property
    def P_T_M(self):
        '''Returns the transformation matrix from the parent'''
        return self.parent.T_M

    def params(self, key):
        return self.parent.params[key][self.idx]

# --------------------------------------------------------------------------------------------------------------------------------------
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
        '''Return the world transform matrix to the connection'''
        if self.recalculate:
            self._T_M = self.P_T_M * self.C_T_M
            self.recalculate = False
            return self._T_M
        else:
            return self._T_M

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
        '''Returns the distance from this connection point to
        other connection point'''
        return np.math.sqrt(np.sum(np.square(self.xyz-other_connection.xyz)))

# --------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------


class BoundingBox(ChildGeometry):
    perimeter_key = "bounding_boxes"

    def __init__(self, parent, idx):
        super().__init__(parent, idx)

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
        if self.recalculate:
            self._points = np.zeros([self.n_perimeter_points, 3])
            for idx, point in enumerate(self.raw_perimeter_points):
                self._points[idx, :] = transform_point(point, self.P_T_M)
            return self._points
        else:
            return self._points

    def as_polygon(self) -> Polygon:
        return Polygon(self.perimeter_points)

# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------


class TunnelAxis(ChildGeometry):
    key = "tunnel_axis"

    def __init__(self, parent, idx, res=0.5):
        super().__init__(parent, idx)
        self.res = res
        self.set_n_intermediate_points()

    @property
    def raw_segment_points(self):
        return np.array(self.params(self.key))

    @property
    def n_segment_points(self):
        return len(self.raw_segment_points)

    @property
    def n_segments(self):
        return self.n_segment_points-1

    @property
    def segment_points(self):
        if self.recalculate:
            self._segment_points = np.zeros([self.n_segment_points, 3])
            for idx, point in enumerate(self.raw_segment_points):
                self._segment_points[idx, :] = transform_point(point, self.P_T_M)
            return self._segment_points
        else:
            return self._segment_points

    def set_n_intermediate_points(self):
        total_extra_points = 0
        self.segment_info = []
        for ns in range(self.n_segments):
            d = np.math.sqrt(np.square(np.sum(self.raw_segment_points[ns]-self.raw_segment_points[ns+1])))
            segment_extra_points = ceil(d/self.res) - 2 
            total_extra_points += segment_extra_points
            self.segment_info.append(segment_extra_points)
        total_n_points = total_extra_points + self.n_segment_points
        self._points = np.zeros((total_n_points,3))
    
    @property
    def n_points(self):
        return len(self._points)
    
    @property
    def points(self):
        '''This funciton is to be called once, after the tile has reached its final 
        location. It will generate intermediate points between the ones that define
        the segments'''
        if self.recalculate:
            idx = 0
            for ns in range(self.n_segments):
                self._points[idx,:] = self.segment_points[ns]
                idx+=1
                nsp = self.segment_info[ns]
                u = (self.segment_points[ns+1] - self.segment_points[ns])/nsp
                intra_segment_points = np.multiply(np.reshape(np.arange(1,nsp+0.01,1),(-1,1)), np.reshape(u,(1,3))) + self.segment_points[ns]
                self._points[idx:idx+nsp,:] = intra_segment_points
                idx +=nsp
            self._points[-1,:] = self.segment_points[-1]
            return self._points
        else:
            return self._points
    
    @property
    def x(self):
        return self.points[:,0]
    @property
    def y(self):
        return self.points[:,1]
    @property
    def z(self):
        return self.points[:,2]
            

############################################################################################################################
############################################################################################################################
#		FUNCTIONS
############################################################################################################################
############################################################################################################################

############################################################################################################################
#	Geometry functions
############################################################################################################################

# --------------------------------------------------------------------------------------------------------------------------------------


def xyzrot_to_TM(xyzrot):
    '''Transforms a [x,y,z,roll,pitch,yaw] vector to a transformation matrix'''
    assert len(xyzrot) == 6
    r = np.matrix(Rotation.from_euler("xyz", xyzrot[-3:]).as_dcm())
    p = np.matrix(xyzrot[:3]).T
    return np.vstack([np.hstack([r, p]), np.matrix([0, 0, 0, 1])])

# --------------------------------------------------------------------------------------------------------------------------------------


def TM_to_xyzrot(TM):
    '''Transforms a transformation matrix to a [x,y,z,roll,pitch,yaw] vector'''
    r = list(np.array(Rotation.from_dcm(TM[:3, :3]).as_euler("xyz")).flatten())
    p = list(np.array(TM[:3, -1]).flatten())
    return p + r

# --------------------------------------------------------------------------------------------------------------------------------------


def scale_geom(geom, scale):
    '''Wraper for the affinity.scale function from the shapely module, 
    so that all the dimensions are scaled equaly'''
    return affinity.scale(geom,
                          xfact=scale,
                          yfact=scale,
                          zfact=scale,
                          origin=(0, 0, 0))

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

# --------------------------------------------------------------------------------------------------------------------------------------


def recursive_scaling(scale, iterable):
    for i, element in enumerate(iterable):
        if type(element) == list:
            element = recursive_scaling(scale, element)
        else:
            iterable[i] *= scale
    return iterable

# --------------------------------------------------------------------------------------------------------------------------------------


def close_list_of_points(list_of_points: np.ndarray):
    '''Mainly for plotting purposes, adds the first element to
    the end of the list so a closing segment is plotted with 
    matplotlib.pyplot.plot()'''
    new_line = list_of_points[[0], :]
    return np.vstack([list_of_points, new_line])


# --------------------------------------------------------------------------------------------------------------------------------------
def get_random_tile():
    return Tile(random.choice(list(Tile.CD.keys())))


# --------------------------------------------------------------------------------------------------------------------------------------
def get_random_non_blocking_tile():
    no_block_list = list(Tile.CD.keys())
    no_block_list.remove(ALIAS["tunnel_block"])
    no_block_list.remove("hatch")
    return Tile(random.choice(no_block_list))

############################################################################################################################
#	Plotting Functions
############################################################################################################################
# --------------------------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------------------------------


def plot_tile(tile, bounding_boxes=True, connections=True, tunnel_axis = True):
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

    if tunnel_axis:
        for axs in tile.axis:
            plt.scatter(axs.x, axs.y, c="r")

    if connections:
        for cnp in tile.connection_points:
            assert isinstance(cnp, ConnectionPoint)
            plt.scatter(cnp.x, cnp.y, c="k")


    return min_borders

# --------------------------------------------------------------------------------------------------------------------------------------


def plot_seq_of_tiles(seq_of_tiles, bounding_boxes=True, areas=True, exits=True, connections=True):
    plt.gca().clear()
    for idx, tile in enumerate(seq_of_tiles):
        if idx == 0:
            borders = plot_tile(
                tile, bounding_boxes=bounding_boxes, connections=connections)
        else:
            borders.update_with_other_instance(plot_tile(
                tile, bounding_boxes=bounding_boxes, connections=connections))

    min_x, min_y, max_x, max_y = borders.borders

    size_x = max_x - min_x
    size_y = max_y - min_y

    final_size = max(size_x, size_y)

    plt.gca().set_xlim(min_x, min_x+final_size)
    plt.gca().set_ylim(min_y, min_y+final_size)
    plt.draw()
