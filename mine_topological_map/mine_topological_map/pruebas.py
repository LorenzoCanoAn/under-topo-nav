from mine_map import MineMap
from node import Node


def main():
    my_mine_map = MineMap()
    my_mine_map.add_node(my_mine_map, 3)
    my_mine_map.add_node(my_mine_map, 2, 1)
    my_mine_map.add_node(my_mine_map, 2, 1)
    my_mine_map.add_node(my_mine_map, 1, 2)
    my_mine_map.add_node(my_mine_map, 3, 3)


if __name__ == "__main__":
    main()
