from shapely.geometry import Polygon, Point
import random
from subt_world_generation.tile import Tile
import numpy as np
import math
from scipy.stats import norm

gaussian_witdth = np.arange(-3, 3, 0.1)
GAUSSIAN = norm.pdf(gaussian_witdth,0,1) / max(norm.pdf(gaussian_witdth,0,1))


def random_label_in_tile(tile: Tile, dist_from_border):
    borders = tile.bounding_boxes[0].as_polygon()
    interior_borders = borders.buffer(-dist_from_border)
    point = random_points_in_polygon(interior_borders)
    orientation = np.random.random(1)*2*np.math.pi
    label = make_label(tile,point,orientation,5)
    return point, orientation,  label

def random_points_in_polygon(p:Polygon):
    min_x, min_y, max_x, max_y = p.bounds

    i = 0
    while True:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(p)):
            xy = np.array(random_point.coords).flatten()
            break
    return xy

def put_gaussian_in_place(i_array, idx):
    for i in range(60):
        idx_ = (idx-30+i)%360
        i_array[idx_] = max(GAUSSIAN[i], i_array[idx_])
    return i_array

def label_from_angles(angles):
    label = np.zeros((360))
    for angle in angles:
        label = put_gaussian_in_place(label, angle)
    return label

def make_label(tile, robot_pose, robot_rotation, radius):
    neighbors = tile.neighbors
    relevant_tiles = neighbors + [tile]

    axes_points = None
    for tile in relevant_tiles:
            for ax in tile.axis:
                if axes_points is None:
                    axes_points = ax.points
                else:
                    axes_points = np.vstack((axes_points, ax.points))
    res = ax.res
    # get rid of z axis
    axes_points = axes_points[:, 0:-1]
    # get points at certain_distance of point
    d = abs(np.sqrt(np.sum(np.square(axes_points-robot_pose),-1)) - radius)
    points_at_correct_distance = axes_points[d<res/2*1.1]
    
    vectors = points_at_correct_distance - robot_pose
    angles_rad = np.arctan2(vectors[:,1],vectors[:,0])
    angles_rad = (angles_rad - robot_rotation + 2*math.pi) % (2*math.pi)
    angles_deg = (angles_rad*180/math.pi).astype(int)
    label = label_from_angles(angles_deg)
    return label