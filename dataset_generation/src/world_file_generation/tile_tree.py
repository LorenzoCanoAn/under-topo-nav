from typing import overload
from world_file_generation.tile import Tile, plot_seq_of_tiles
import pickle
import os

##############################################################
#	Stablish the path to necessary files
##############################################################

TILE_DEFINITION_FILE_PATH = os.path.join(
    os.path.realpath(__file__), "data_files/urdf_template_files/tile_definition.txt")
BOILERPLATE_FILE_PATH = os.path.join(
    os.path.realpath(__file__), "data_files/urdf_template_files/boilerplate.txt")
OBSTACLE_DEFINITION_FILE_PATH = os.path.join(
    os.path.realpath(__file__), "data_files/urdf_template_files/obstacle.txt")

################################################################################################################################
#	Alias for the tile names
################################################################################################################################


##################################################################
##################################################################
#		CLASSES
##################################################################
##################################################################


##############################################################
#	 definition of the TileTree class
##############################################################

class TileTree:
    def __init__(self):
        self._scale = 1.0
        self.tiles = []

    def set_scale(self, scale):
        '''This funciton changes the scale of the definitions of all the tiles'''
        Tile.scale(scale / self._scale)
        self._scale = scale

    def add_tile(self, i_type, parent_idx:int, p_connection, c_connection):
        ''' Add a tile to the tree, if parent is set to None, the tile
        is added to the list with pose 0,0,0,0,0,0.
        The if the parent is not set to None, it should be the idx of the parent in the'''
        if parent_idx == None:
            tile = Tile(i_type)
            self.tiles.append(tile)
        else:
            assert isinstance(parent_idx, int)
            parent = self.tiles[parent_idx]
            if isinstance(parent.connections[p_connection], type(None)):
                tile = Tile(i_type)
                tile.connect(parent, c_connection, p_connection)
                self.tiles.append(tile)

                if not self.check_collisions(tile):
                    return True
                else:
                    print("Deleting tile for collission")
                    self.del_tile(tile)
                    return False
            else:
                print(
                    f"Parent tile has {parent.connections} connections, but connection p_connection was requested")
                return False

    def del_tile(self, ttd):
        # ttd means Tile To Delete
        assert isinstance(ttd, Tile)
        for conn in ttd.connections:  # Loop over connecitons of ttd
            if isinstance(conn, Tile):  # If a tile is in the connection
                idx = conn.connections.index(ttd)
                # delete ttd from that connection
                conn.connections[idx] == None
        self.tiles.remove(ttd)

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump(self, f)

    def load(self, path):
        self.tiles = pickle.load(open(path, "rb")).tiles

    def to_world_file_text(self, path):
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

        with open(BOILERPLATE_FILE_PATH, 'r') as f:
            final_text = f.read()
        final_text = final_text.format(complete_text)

        with open(path, "w+") as f:
            f.write(final_text)

    def check_collisions(self, tile):
        assert isinstance(tile, Tile)
        for other_tile in self.tiles:
            assert isinstance(other_tile, Tile)
            if not other_tile in tile.connections and other_tile != tile:
                print(
                    f"Checking collisions between {self.tiles.index(tile)} and {self.tiles.index(other_tile)}")
                for bb1 in tile.bounding_boxes:
                    for bb2 in other_tile.bounding_boxes:
                        result = bb1.as_polygon().intersects(bb2.as_polygon())
                        if result:
                            return result

        return False


####################################################################################################################################
####################################################################################################################################
#		Functions
####################################################################################################################################
####################################################################################################################################

#	 definition of the plot_tree function
# --------------------------------------------------------------------------------------------------------------------------------------
def plot_tree(tile_tree):
    assert isinstance(tile_tree, TileTree)
    plot_seq_of_tiles(list(tile_tree.tiles))

