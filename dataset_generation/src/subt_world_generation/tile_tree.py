from copyreg import pickle
from subt_world_generation.tile import Tile, plot_seq_of_tiles, BLOCK_TILES
import numpy as np
import os
import pickle
####################################################################################################################################
####################################################################################################################################
#		CLASSES
####################################################################################################################################
####################################################################################################################################

# -----------------------------------------------------------------------------------------------------------------------------------
#	 definition of the TileTree class
# -----------------------------------------------------------------------------------------------------------------------------------
class TileTree:
    def __init__(self):
        self._scale = 1.0
        self.tile_grid = TileGrid()
        self.tiles = []

    def __getitem__(self,i):
        return self.tiles[i]

    def __len__(self):
        return len(self.tiles)

    def set_scale(self, scale):
        '''This funciton changes the scale of the definitions of all the tiles'''
        self._scale = scale
        Tile.scale(self._scale)

    def move_add_and_connect_tile(self, t1, nc1, t2, nc2):
        '''Moves 1 as if connected to 1, then adds it to the tree'''
        assert(isinstance(t1, Tile))
        t1.move_to_connection(t2,nc2,nc1)
        self.add_and_connect_tile(t1,nc1,t2,nc2)

    def add_and_connect_tile(self, t1, nc1, t2, nc2):
        self.add_tile(t1)
        self.connect_two_tiles(t1, nc1, t2, nc2)

    def add_tile(self, t1):
        self.tiles.append(t1)
        self.tile_grid.add_tile(t1)

    def connect_two_tiles(self, t1, nc1, t2, nc2):
        '''Assigns the correct connectios to each 
        tile and updates the free connections of the tree'''
        t1.connect(t2, nc2, nc1)

    def del_tile(self, t1):
        # ttd means Tile To Delete
        for t2 in t1.neighbors:
            t2.connections[t2.connections.index(t1)] == None
        t1.reset_connections()
        self.tiles.remove(t1)
        self.tile_grid.remove_tile(t1)

    def check_collisions(self, tile):
        for other_tile in self.tile_grid.get_neighbors(tile):
            for bb1 in tile.bounding_boxes:
                for bb2 in other_tile.bounding_boxes:
                    inter = bb1.as_polygon().intersection(bb2.as_polygon()).area
                    if inter > 3: return True
        return False

    @property
    def non_blocking_tiles(self):
        self._non_blocking_tiles = set()
        for t in self:
            if not t.params["model_name"] in BLOCK_TILES:
                self._non_blocking_tiles.add(t)
        return self._non_blocking_tiles

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def full_save(self, path):
        '''This funciton saves a copy of the tree as a pickle file and
        the gazebo.world version so that it can be loaded to gazebo'''
        if not os.path.exists(path):
            os.mkdir(path)
        world_file_path = os.path.join(path,"gazebo.world")
        tree_file_path = os.path.join(path,"tree.pickle")
        save_tree_as_world_file(self, world_file_path)
        with open(tree_file_path, "wb") as f:
            pickle.dump(self,f)

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
class TileGrid():
    def __init__(self):
        self.__grid = dict()

    def add_tile(self, tile):
        assert isinstance(tile, Tile)
        coord = self.get_coord(tile)
        try:
            self.__grid[coord].add(tile)
        except:
            self.__grid[coord] = set([tile])
    
    def remove_tile(self, tile):
        coord = self.get_coord(tile)
        self.__grid[coord].remove(tile)

    def get_neighbors(self, tile):
        coord = np.array(self.get_coord(tile))
        neighbors = set()
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    f_coord = coord + np.array((dx,dy,dz))
                    f_coord = tuple(f_coord)
                    try:
                        for t in self.__grid[f_coord]:
                            neighbors.add(t)
                    except:
                        pass
        try:
            neighbors.remove(tile)
        except:
            pass
        return neighbors

    def get_coord(self, tile):
        return tuple(np.array(tile.xyz/15).flatten().astype(int))

####################################################################################################################################
####################################################################################################################################
#		Functions
####################################################################################################################################
####################################################################################################################################

#	 plot_tree function
# ----------------------------------------------------------------------------------------------------------------------------------
def plot_tree(tile_tree, bounding_boxes=True, connections=True, tunnel_axis = True):
    plot_seq_of_tiles(list(tile_tree.tiles), bounding_boxes=bounding_boxes, connections=connections, tunnel_axis = tunnel_axis)

####################################################################################################################################
#	Functions for saving the tree as a .world file for gazebo
####################################################################################################################################


#	Stablish the path to necessary files
TILE_DEFINITION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/base_of_tile_definition.txt")
WORLD_DEFINITION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/base_world_file.txt")
OBSTACLE_DEFINITION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/obstacle.txt")


def tile_to_text(tile: Tile, idx = "") -> str:
    assert isinstance(tile, Tile)
    with open(TILE_DEFINITION_FILE_PATH, "r") as f:
        raw_text = f.read()
    pose = str(tile.xyzrot).replace(",","")
    pose = pose.replace("[","").replace("]","")
    return raw_text.format(idx, pose, tile.uri)


def tree_tiles_to_text(tree: TileTree):
    text = []
    for idx, tile in enumerate(tree.tiles):
        text.append(tile_to_text(tile,idx=idx))
    return "".join(text)

def tree_to_text(tree: TileTree):
    tiles_text = tree_tiles_to_text(tree)
    with open(WORLD_DEFINITION_FILE_PATH, "r") as f:
        raw_text = f.read()
    return raw_text.format(tiles_text)

def save_tree_as_world_file(tree, save_path):
    with open(save_path, "w+") as f:
        f.write(tree_to_text(tree))