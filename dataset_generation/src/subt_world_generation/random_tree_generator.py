from subt_world_generation.tile_tree import TileTree, plot_tree
from subt_world_generation.tile import Tile, get_random_tile ,ALIAS, get_random_non_blocking_tile
import matplotlib.pyplot as plt
import random
from time import time_ns as ns
# --------------------------------------------------------------------------------------------------------------------------------------
#	 definition of the RandomTreeGenerator class
# --------------------------------------------------------------------------------------------------------------------------------------


class RandomTreeGenerator(TileTree):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __init__(self):
        super().__init__()
        self.__free_connections = set()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def gen_tree(self):
        tile = get_random_tile()
        self.add_tile(tile)
        while len(self.free_connections) > 0:
            self.generation_step()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def generation_step(self):
        '''
        This function:
        -1: tries to close all the loops posible
        -2: Adds a random tile to the rest of the open tiles
        '''
        # TRY TO CLOSE ALL POSIBLE LOOPS
        with open("debug.txt", "a+") as f:

            loops_closed = True

            a = ns()
            for t1, nc1 in self.__free_connections:
                if self.close_loop(t1, nc1):
                    loops_closed = True
                    break
            print(len(self.__free_connections))
            
            elapsed = str((ns()-a))
            f.write(f"{elapsed}||{len(self.__free_connections)}||\n")



                
        
            
        # Select a tile to put
        
        t2, nc2 = random.choice(self.free_connections)
        t1 = get_random_tile()
        nc1 = random.randint(0, len(t1.connections)-1)
        self.move_add_and_connect_tile(t1, nc1, t2, nc2)
        if self.check_collisions(self[-1]):
            self.del_tile(self[-1])
            self.move_add_and_connect_tile(Tile(ALIAS["tunnel_block"]),0,t2,nc2)
        return 1

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
    def get_loop_closure_candidate_connections(self, t1, nc1):
        '''This function takes a tile_id and a connection from that tile,
        then searches for the other free connections that are close to it.'''
        p1 = self.pose_of_connection(t1, nc1)
        candidates = []
        # Loop over all the free connections
        for t2, nc2 in self.free_connections:
            # If the free connection is different than the one to be connected
            if t2 != t1:
                # get the pose of the candidate connection and check its distance
                p2 = self.pose_of_connection(t2, nc2)
                if p1.distance(p2) < self._scale * 20:
                    # This candidate connection is close enough
                    candidates.append((t2, nc2))
        return candidates

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def make_best_connection(self, t1, nc1, t2nc2_list):
        '''For two given open exits, finds all the tiles that connect them, if any'''
        final_total_connections = []
        t3 = None
        prev_tile = None
        for t2, nc2 in t2nc2_list:
            p2 = t2.connection_points[nc2]
            for t_type in Tile.CD:
                t3 = Tile(t_type)
                for nc3 in t3.empty_connections:
                    # At this point, we are trying to evaluate a tile, in a certain position
                    t3.move_to_connection(t1, nc1, nc3)
                    # First check if the target connection has been closed
                    for nc3_, p3_ in enumerate(t3.connection_points):
                        if p2.distance(p3_) < 0.1:
                            base_connection = (t3, nc3, t1, nc1)
                            secondary_connection = (t3, nc3_, t2, nc2)
                            # If it has been closed, check if any other t2nc2 can be connected
                            extra_connections = []
                            for nc3__, p3__ in enumerate(t3.connection_points):
                                if nc3__ in [nc3_, nc3]:
                                    continue
                                for t2_, nc2_ in t2nc2_list:
                                    if (t2_, nc2_) == (t2, nc2):
                                        continue
                                    p2_ = t2_.connection_points[nc2_]
                                    if p2_.distance(p3__) < 0.1:
                                        extra_connections.append(
                                            (t3, nc3__, t2_, nc2_))
                            total_connections = [base_connection, secondary_connection] + extra_connections

                            if prev_tile is not None:
                                self.del_tile(prev_tile)
                            
                            try:
                                self.execute_connections(t3, total_connections)
                            except:
                                pass
                            
                            if self.check_collisions(t3):
                                self.del_tile(t3)
                                if prev_tile is not None:
                                    self.execute_connections(
                                        prev_tile, final_total_connections)
                            else:
                                final_total_connections = total_connections.copy()
                                prev_tile = t3
        return t3, final_total_connections
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def make_best_connection_v2(self, t1, nc1, t2nc2_list):
        '''For two given open exits, finds all the tiles that connect them, if any'''
        t3 = None
        final_total_connections = []
        definitive_tile = None
        if len(t2nc2_list) == 0:
            return final_total_connections


        for t_type in Tile.CD:
            t3 = Tile(t_type)
            for nc3 in range(t3.n_connections):
                current_cand_possible_connections = []
                t3.move_to_connection(t1, nc1, nc3)
                if self.check_collisions(t3):
                    continue                
                
                for nc3_, p3_ in enumerate(t3.connection_points):
                    if nc3_ == nc3: continue
                    for t2, nc2 in t2nc2_list:
                        p2 = t2.connection_points[nc2]
                        if p3_.distance(p2) < 0.1:
                            current_cand_possible_connections.append((t3,nc3_,t2,nc2))


                if len(current_cand_possible_connections) > 0:
                    current_cand_possible_connections.append((t3,nc3,t1,nc1))

                if len(final_total_connections) < len(current_cand_possible_connections):
                    final_total_connections = current_cand_possible_connections.copy()
                    if len(final_total_connections) == len(t2nc2_list):
                        self.execute_connections(final_total_connections)
                        return final_total_connections

        self.execute_connections(final_total_connections)     
        return final_total_connections

    def close_loop(self, t1, nc1):
        t2nc2_list = self.get_loop_closure_candidate_connections(t1, nc1)
        connections = self.make_best_connection_v2(t1, nc1, t2nc2_list)
        if len(connections) > 0:
            return True
        return False

    def execute_connections(self, connections):
        '''
        - tile is suposed to be already moved
        - conections is a list of tuples (t1,nc1,t2,nc2)
        '''

        for i, (t1, nc1, t2, nc2) in enumerate(connections):
            if i == 0:
                self.move_add_and_connect_tile(t1,nc1,t2,nc2)
            else:
                self.connect_two_tiles(t1, nc1, t2, nc2)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def pose_of_connection(self, p_tile, conn):
        '''Given the tile_idx and the connection in said tile, returns
        the position of the connection'''
        p_conn_pose = p_tile.connection_points[conn]
        return p_conn_pose
####################################################################################################################################
####################################################################################################################################
#		FUNCTIONS
####################################################################################################################################
####################################################################################################################################

#	 definition of the random_tile_type function
# -----------------------------------------------------------------------------------------------------------------------------------


def random_tile_type():
    """Returns a random selection of the types of intersections"""
    lista_de_tipos = Tile.CD.keys()
    return random.choice(lista_de_tipos)

def plot_random_tree(tree, bounding_boxes=True, areas=False, exits=False, connections=False):
    plot_tree(tree, bounding_boxes=bounding_boxes, areas=areas, exits=exits, connections=connections)
    assert isinstance(tree, RandomTreeGenerator)
    for t, c in tree.free_connections:
        p = t.connection_points[c]
        plt.scatter(p.x, p.y, c="y")
    plt.draw()