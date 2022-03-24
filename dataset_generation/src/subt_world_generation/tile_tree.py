from subt_world_generation.tile import Tile, plot_seq_of_tiles
import os

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
        self.tiles = []

    def __getitem__(self,i):
        return self.tiles[i]

    def __len__(self):
        return len(self.tiles)

    def set_scale(self, scale):
        '''This funciton changes the scale of the definitions of all the tiles'''
        Tile.scale(scale / self._scale)
        self._scale = scale

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

    def check_collisions(self, tile):
        for other_tile in self.get_tiles_in_range(tile, 20):
            for bb1 in tile.bounding_boxes:
                for bb2 in other_tile.bounding_boxes:
                    inter = bb1.as_polygon().intersection(bb2.as_polygon()).area
                    if inter > 2: return True
        return False

    def get_tiles_in_range(self, t1, r):
        t_in_range = set()
        for t2 in self.tiles:
            if t2 == t1:
                continue
            if t2.distance(t1) <= r:
                t_in_range.add(t2)
        return t_in_range

####################################################################################################################################
####################################################################################################################################
#		Functions
####################################################################################################################################
####################################################################################################################################

#	 plot_tree function
# ----------------------------------------------------------------------------------------------------------------------------------


def plot_tree(tile_tree, bounding_boxes=True, areas=True, exits=True, connections=True):
    plot_seq_of_tiles(list(tile_tree.tiles), bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections)

####################################################################################################################################
#	Functions for saving the tree as a .world file for gazebo
####################################################################################################################################


#	Stablish the path to necessary files
TILE_DEFINITION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/base_of_tile_definition.txt")
BOILERPLATE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/boilerplate.txt")
OBSTACLE_DEFINITION_FILE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data_files/urdf_template_files/obstacle.txt")


def tile_to_text(tile: Tile) -> str:
    assert isinstance(tile, Tile)
    with open(TILE_DEFINITION_FILE_PATH, "r") as f:
        raw_text = f.read()
    return raw_text.format(tile.idx, tile.xyzrot, tile.uri)


def tree_tiles_to_text(tree: TileTree):
    text = []
    for tile in tree.tiles:
        text.append(tile_to_text(tile))
    return "".join(text)
