from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
import random
from subt_world_generation.tile import Tile
import numpy as np

def random_point_in_tile(tile: Tile, dist_from_border):
    borders = tile.bounding_boxes[0].as_polygon()
    x,y = x,y = borders.exterior.xy
    plt.figure(figsize=(10,10))

    borders = borders.buffer(-dist_from_border)
    x,y = x,y = borders.exterior.xy
    plt.plot(x, y)

    points = random_points_in_polygon(borders, 20)
    points = np.array(points)


    plt.show()
    return points

def random_points_in_polygon(p:Polygon, num_points):
    min_x, min_y, max_x, max_y = p.bounds

    points = set()

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(p)):
            coords = tuple(random_point.coords)[0]
            print(coords)