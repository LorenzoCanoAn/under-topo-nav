from shapely.geometry import Polygon, Point
import random
from subt_world_generation.tile import Tile
import numpy as np
from subt_world_generation.tile_tree import TileTree

def random_points_in_tile(tile: Tile, dist_from_border,n_points, n_layer = 0):
    borders = tile.bounding_boxes[0].as_polygon()
    interior_borders = borders.buffer(-dist_from_border)
    z = tile.params["height_of_bboxes"][n_layer]
    points = random_points_in_polygon(interior_borders,z, n_points)
    return points

def random_points_in_polygon(p:Polygon,z, num_points):
    min_x, min_y, max_x, max_y = p.bounds

    points = np.zeros([3,num_points])
    i = 0
    while i < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(p)):
            xy = np.array(random_point.coords).flatten()
            xyz = np.hstack([xy, z])
            points[:,i] = xyz
            i+=1

    return points

def random_points_in_tile_tree(tree: TileTree, ppt):
    tiles = tree.non_blocking_tiles
    points = np.zeros((3, ppt*len(tiles)))
    for idx, t in enumerate(tree.non_blocking_tiles):
        points[:,idx*ppt:(idx+1)*ppt] = random_points_in_tile(t,1,ppt)
    return points