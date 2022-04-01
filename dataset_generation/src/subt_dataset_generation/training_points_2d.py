from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
import random
from subt_world_generation.tile import Tile
import numpy as np
from subt_world_generation.tile_tree import TileTree
from subt_dataset_generation.tunnel_axis_fusion import get_fusioned_axis
from scipy.stats import norm
x = np.arange(-3, 3, 0.1)
GAUSSIAN = norm.pdf(x,0,1)


def random_points_in_tile(tile: Tile, dist_from_border,n_points, n_layer = 0):
    borders = tile.bounding_boxes[0].as_polygon()
    interior_borders = borders.buffer(-dist_from_border)
    z = tile.params["height_of_bboxes"][n_layer]
    points = random_points_in_polygon(interior_borders,z, n_points)
    orientations = np.random.random(len(points.T))*2*np.math.pi
    labels = get_labels_for_points(points,orientations, tile)
    return points, orientations,  labels

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
    print(points.shape)
    labels = np.zeros((360, ppt*len(tiles)))
    orientations = np.zeros((1,ppt*len(tiles)))
    for idx, t in enumerate(tree.non_blocking_tiles):
        i_idx = idx*ppt
        f_idx = (idx+1)*ppt
        points[:,i_idx:f_idx], orientations[:,i_idx:f_idx],labels[:,i_idx:f_idx]  = random_points_in_tile(t,1,ppt)

    return points, orientations, labels

def get_labels_for_points(points, orientations,  tile:Tile, detection_distance = 4):
    '''Returns a list with the labels for each of the points'''
    tunnel_axis_per_conn = get_fusioned_axis(tile)
    labels = np.zeros((360,len(points.T)))
    
    for idx, p in enumerate(points.T):
        closest_conn , distance= get_closest_connection_point(p, tile)
        if distance > detection_distance:
            closest_conn = None
        axis_for_conn = tunnel_axis_per_conn[closest_conn]
        labels[:,idx] = gen_label(p,orientations[idx], axis_for_conn, detection_distance)
    return labels
def get_closest_connection_point(point, tile):
    '''This function selects the correct set of axis for a given 
    point'''
    min_conn_dist = 10000000
    for conn in tile.connection_points:
        d = conn.distance_to_point(point)
        if d < min_conn_dist:
            min_conn_dist = d
            closest_conn = conn
    return closest_conn, min_conn_dist

def gen_label(point, orientation, axis, dd):
    label = np.zeros((360))
    intersected_points = get_points_for_label(point, axis, dd)
    vectors = intersected_points - point
    label = np.zeros(360)
    for v in vectors:
        yaw = np.math.atan2(v[1], v[0]) - orientation
        yaw = int(yaw*180/3.1415)
        label = put_gaussian_in_place(label, yaw)
    return label
    
def get_points_for_label(point, axis, dd):
    points_for_label = None

    for axs in axis:
        vectors = axs-point
        vec_length = np.sqrt(np.sum(np.square((vectors)),axis=-1))
        dists = abs(vec_length - dd)
        in_range = axs[dists < 0.25,:]

        if len(in_range) > 0:
            p_of_axis = in_range
        else:
            p_of_axis =  axs[np.argmin(vec_length)]
        if points_for_label is None:
            points_for_label= p_of_axis
        else:
            points_for_label = np.vstack((points_for_label, p_of_axis))
    return points_for_label
    

def put_gaussian_in_place(i_array, idx):
    for i in range(60):
        idx_ = (idx-30+i)%360
        i_array[idx_] = max(GAUSSIAN[i], i_array[idx_])
    return i_array