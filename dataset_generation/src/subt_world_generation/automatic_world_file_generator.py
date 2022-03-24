from math import sqrt
import threading
from subt_world_generation.tile import TileTree
import random
from datetime import datetime
import os
import time

class AutoMapGenerator(TileTree):
    possible_instructions = ["front", "right", "left"]
    add_instruction_tile_chance = 0.5

    def __init__(self, n_instructions=5, scale = 1.0):
        self.n_instructions = int(n_instructions)
        super().__init__("test_world",scale = scale)
        self.seed = datetime.now().microsecond + sqrt(datetime.now().second)
        random.seed(self.seed)

    def gen_map(self):
        self.gen_instructions()
        self.gen_random_tree()

    def gen_instructions(self):
        self.instructions = []
        for i in range(self.n_instructions):
            self.instructions.append(random.choice(self.possible_instructions))
        self.instruction_idx = 0

    def init_tree(self):
        print("INITTING TREE")
        self.tiles = []
        self.add_tile("rect", None, None, None, True)
        self.add_tile("block", 0, 1, 0)
        self.exit_to_continue = 0
        self.instruction_idx = 0
        self.tile_to_finish_idx = 0
        self.nexus_tiles = []

    def gen_random_tree(self):
        self.init_tree()
        while True:

            self.add_random_tile()
            self.complete_tile()

            if self.check_collisions():
                self.init_tree()

            else:
                self.tile_to_finish_idx = self.tile_to_complete_idx

            if self.instruction_idx == self.instructions.__len__():
                self.add_tile("rect", self.tile_to_finish_idx,
                              self.exit_to_continue, 0)
                self.add_tile("block", self.tiles[-1].id, 1, 0)
                if self.check_collisions():
                    self.init_tree()

                else:
                    break

    def complete_tile(self):
        possible_connections = range(
            self.tiles[self.tile_to_complete_idx].connections.__len__())
        for connection in possible_connections:
            if connection == self.exit_to_continue:
                continue
            elif type(self.tiles[self.tile_to_complete_idx].connections[connection]) == type(None):
                ch = random.choice([2, 3, 4])
                if ch == 1:
                    self.add_tile(
                        "block", self.tile_to_complete_idx, connection, 0)
                elif ch == 2:
                    self.add_tile(
                        "rect", self.tile_to_complete_idx, connection, 0)
                    self.add_tile(
                        "block", self.tiles[-1].id, 1, 0)
                elif ch == 3:
                    self.add_tile(
                        "curve", self.tile_to_complete_idx, connection, 0)
                    self.add_tile(
                        "block", self.tiles[-1].id, 1, 0)
                elif ch == 4:
                    self.add_tile(
                        "curve", self.tile_to_complete_idx, connection, 1)
                    self.add_tile(
                        "block", self.tiles[-1].id, 0, 0)

    def add_random_tile(self):
        # This function knows to which element of the parent has to connect to
        ch = random.choice([1, 2])

        if ch == 1:
            self.add_instruction_tile()
            self.tile_to_complete_idx = self.tiles[-1].id
            return 1
        else:
            self.add_non_instruction_tile()
            self.tile_to_complete_idx = self.tiles[-1].id
            return 0

    def delete(self, idx):
        for _ in range(idx, self.tiles.__len__()):
            self.tiles.pop(idx)

    def add_instruction_tile(self):
        if self.instructions[self.instruction_idx] == "front":
            self.add_tile_to_go_straigh()
        if self.instructions[self.instruction_idx] == "left":
            self.add_tile_to_go_left()
        if self.instructions[self.instruction_idx] == "right":
            self.add_tile_to_go_right()
        self.instruction_idx += 1

    def add_tile_to_go_left(self):
        # either normal intersection or t_intersection connected with 0 or 1
        rc = random.choice([1, 2, 3])
        if rc == 1:
            self.add_tile("inter", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)
            self.exit_to_continue = 3            
            print("INSTRUCTION_TILE: INTER LEFT")
        elif rc == 2:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)
            self.exit_to_continue = 2
            print("INSTRUCTION_TILE:   T   LEFT")
        elif rc == 3:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 1)
            self.exit_to_continue = 0
            print("INSTRUCTION_TILE:   D   LEFT")

    def add_tile_to_go_right(self):
        # either normal intersection or t_intersection connected with 0 or 2
        rc = random.choice([1, 2, 3])
        if rc == 1:
            self.add_tile("inter", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)
            self.exit_to_continue = 1            
            print("INSTRUCTION_TILE: INTER RIGHT")
        elif rc == 2:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)
            self.exit_to_continue = 1            
            print("INSTRUCTION_TILE:   T   RIGHT")
        elif rc == 3:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 2)
            self.exit_to_continue = 0            
            print("INSTRUCTION_TILE:   D   RIGHT")

    def add_tile_to_go_straigh(self):
        # either normal intersection or t_intersection connected with 1 or 2
        rc = random.choice([1, 2, 3])
        if rc == 1:
            self.add_tile("inter", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)            
            print("INSTRUCTION_TILE: INTER STRAIGH")
            self.exit_to_continue = 2
        elif rc == 2:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 1)            
            print("INSTRUCTION_TILE:   TI  STRAIGH")
            self.exit_to_continue = 2
        elif rc == 3:
            self.add_tile("t", self.tile_to_finish_idx,
                          self.exit_to_continue, 2)            
            print("INSTRUCTION_TILE:   TD  STRAIGH")
            self.exit_to_continue = 1

    def add_non_instruction_tile(self):
        rc = random.choice([1, 2])
        if rc == 1:
            self.add_tile("rect", self.tile_to_finish_idx,
                          self.exit_to_continue, 0)
            self.exit_to_continue = 1
        else:
            rc = random.choice([1, 2])
            if rc == 1:
                self.add_tile("curve", self.tile_to_finish_idx,
                              self.exit_to_continue, 0)
                self.exit_to_continue = 1

            else:
                self.add_tile("curve", self.tile_to_finish_idx,
                              self.exit_to_continue, 1)
                self.exit_to_continue = 0


if __name__ == "__main__":
    N_INSTRUCTIONS = 10
    BASE_PATH = "/home/lorenzo/catkin_ws/src/underground_semantic_navigation_ROS/dataset_generation/saved_worlds/random"
    hola = AutoMapGenerator(n_instructions=N_INSTRUCTIONS)
    for _ in range(10):
        hola.gen_map()
        name = str(time.time_ns())
        print(name)
        folder = os.path.join(BASE_PATH, str(N_INSTRUCTIONS))
        if not os.path.isdir(folder):
            os.mkdir(folder)
        hola.to_world_file_text(os.path.join(folder,name))

