from subt_world_generation.tile_tree import TileTree, plot_tree
from subt_world_generation.tile import Tile, get_random_non_blocking_tile, get_random_tile, ALIAS
import matplotlib.pyplot as plt
import random
from time import time_ns as ns

# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the RandomTreeGenerator class
# --------------------------------------------------------------------------------------------------------------------------------------
class RandomTreeGenerator(TileTree):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __init__(self, max_tiles = 200):
        super().__init__()
        random.seed(ns())
        self.__free_connections = FreeConnectionsTracker(self)
        self.max_tiles = max_tiles

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def gen_tree(self):
        tile = get_random_tile()
        self.add_tile(tile)
        i = 0
        while self.max_tiles > len(self) and len(self.free_connections) > 0:
            self.generation_step()
        self.close_all_open_connections()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def generation_step(self):
        '''
        This function:
        -1: tries to close all the loops posible
        -2: Adds a random tile to the rest of the open tiles
        '''
        self.close_all_loops()
        if (len(self.__free_connections)) ==0:
            return 1
        t2, nc2 = random.choice(self.free_connections)
        t1 = get_random_non_blocking_tile()
        nc1 = random.randint(0, len(t1.connections)-1)
        self.move_add_and_connect_tile(t1, nc1, t2, nc2)
        if self.check_collisions(self[-1]):
            self.del_tile(self[-1])
            self.move_add_and_connect_tile(Tile("hatch"), 0, t2, nc2)
        self.close_all_loops()
        return 1
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def close_all_loops(self):
        loop_closed= True
        while loop_closed:
            loop_closed = False
            for sml_dist in self.__free_connections.small_distances:
                t1, nc1 = sml_dist[1]
                has_closed_loop = self.close_loop(t1, nc1)
                if has_closed_loop:
                    loop_closed = True
                    break
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def close_all_open_connections(self):
        while len(self.free_connections) > 0:
            self.close_random_exit()
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def close_random_exit(self):
        t2, nc2 = random.choice(self.free_connections)
        self.move_add_and_connect_tile(
                Tile("hatch"), 0, t2, nc2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def add_tile(self, t1):
        '''This function only appends a tile to the list tree, and
        updates the free connections'''
        super().add_tile(t1)
        for nc1 in t1.empty_connections:
            self.__free_connections.add((t1, nc1))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def del_tile(self, t1):
        for nc1 in t1.empty_connections:
            self.__free_connections.remove((t1, nc1))

        for t2 in t1.neighbors:
            self.__free_connections.add((t2, t2.connections.index(t1)))

        super().del_tile(t1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def connect_two_tiles(self, t1, nc1, t2, nc2):
        '''Assigns the correct connectios to each 
        tile and updates the free connections of the tree'''
        super().connect_two_tiles(t1, nc1, t2, nc2)
        self.__free_connections.remove((t1, nc1))
        self.__free_connections.remove((t2, nc2))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @property
    def free_connections(self):
        '''This funciton returns a list of tuples, where the
        first element of the tuple is the index of the tile, 
        and the second element is the connection that is free'''
        return list(self.__free_connections)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def make_best_connection(self, t1, nc1, t2nc2_list):
        '''For two given open exits, finds all the tiles that connect them, if any'''
        t3 = None
        final_total_connections = []
        if len(t2nc2_list) == 0:
            return final_total_connections
        if len(t2nc2_list) > 0:
            possible_tile_types = []
            possible_tile_types.append(ALIAS["tunnel_curve"])
            possible_tile_types.append(ALIAS["tunnel_rect"])
        if len(t2nc2_list) > 1:
            possible_tile_types.append(ALIAS["tunnel_t"])
        if len(t2nc2_list) > 2:
            possible_tile_types.append(ALIAS["tunnel_4_way_intersection"])
        
        # Iterate over the possible tiles
        for t_type in possible_tile_types:
            t3 = Tile(t_type)
            if t3.params["symetric"]:
                iterator = range(1)
            else:
                iterator = range(t3.n_connections)
            # Iterate over the possible connections
            for nc3 in iterator:
                current_cand_possible_connections = []
                t3.move_to_connection(t1, nc1, nc3)
                if self.check_collisions(t3):
                    continue
                
                # Check the other connections of t3
                for nc3_, p3_ in enumerate(t3.connection_points):
                    if nc3_ == nc3:
                        continue
                    for t2, nc2 in t2nc2_list:
                        p2 = t2.connection_points[nc2]
                        if p3_.distance(p2) < 1:
                            current_cand_possible_connections.append(
                                (t3, nc3_, t2, nc2))

                if len(current_cand_possible_connections) > 0:
                    current_cand_possible_connections.append(
                        (t3, nc3, t1, nc1))

                if len(final_total_connections) < len(current_cand_possible_connections):
                    final_total_connections = current_cand_possible_connections.copy()
                    if len(final_total_connections) == len(t2nc2_list):
                        self.execute_connections(final_total_connections)
                        return final_total_connections

        self.execute_connections(final_total_connections)
        return final_total_connections

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def close_loop(self, t1, nc1):
        t2nc2_list = self.__free_connections.get_close_conn((t1, nc1))
        connections = self.make_best_connection(t1, nc1, t2nc2_list)

        if len(connections) > 0:
            return True
        return False

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def execute_connections(self, connections):
        '''
        - tile is suposed to be already moved
        - conections is a list of tuples (t1,nc1,t2,nc2)
        '''

        for i, (t1, nc1, t2, nc2) in enumerate(connections):
            if i == 0:
                self.move_add_and_connect_tile(t1, nc1, t2, nc2)
            else:
                self.connect_two_tiles(t1, nc1, t2, nc2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def pose_of_connection(self, p_tile, conn):
        '''Given the tile_idx and the connection in said tile, returns
        the position of the connection'''
        p_conn_pose = p_tile.connection_points[conn]
        return p_conn_pose
    

# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
class FreeConnectionsTracker(set):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __init__(self, parent, __iterable=set()) -> None:
        super().__init__(__iterable)
        self.small_distances = set()
        self.parent = parent

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def add(self, __element) -> None:
        for conn in self:
            if __element[0] == conn[0]:
                pass
            elif __element[0].connection_points[__element[1]].distance(conn[0].connection_points[conn[1]]) < 25 * self.parent._scale:
                self.small_distances.add((conn, __element))
        super().add(__element)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def remove(self, __element) -> None:
        to_remove = set()

        for r in self.small_distances:
            if __element in r:
                to_remove.add(r)
        for r in to_remove:
            self.small_distances.remove(r)

        super().remove(__element)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def get_close_conn(self, c1):
        close = set()
        for c1c2 in self.small_distances:
            if c1 in c1c2: 
                c1c2_l = list(c1c2)
                c1c2_l.remove(c1)
                close.add(c1c2_l[0])
        return close
    

####################################################################################################################################
####################################################################################################################################
#		FUNCTIONS
####################################################################################################################################
####################################################################################################################################


# -----------------------------------------------------------------------------------------------------------------------------------
def random_tile_type():
    """Returns a random selection of the types of intersections"""
    lista_de_tipos = Tile.CD.keys()
    return random.choice(lista_de_tipos)

# -----------------------------------------------------------------------------------------------------------------------------------


def plot_random_tree(tree, bounding_boxes=True, connections=False, tunnel_axis = False):
    plot_tree(tree, bounding_boxes=bounding_boxes, connections=connections, tunnel_axis = tunnel_axis)
    assert isinstance(tree, RandomTreeGenerator)
    for t, c in tree.free_connections:
        p = t.connection_points[c]
        plt.scatter(p.x, p.y, c="y")
    plt.draw()
