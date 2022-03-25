from subt_dataset_generation.random_point_in_tile import random_point_in_tile
from subt_world_generation.tile import Tile

def main():
    tile = Tile("tunnel_tile_1")
    random_point_in_tile(tile, 1)


if __name__ == "__main__":
    main()