#!/bin/python3
import time
from subt_world_generation.random_tree_generator import RandomTreeGenerator
from subt_world_generation.tile_tree import plot_tree, TileTree, Tile
from shapely.geometry.linestring import LineString
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

x = np.arange(-3, 3, 0.1)
GAUSSIAN = norm.pdf(x,0,1) / max(norm.pdf(x,0,1)) 

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - \
        np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - \
        np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + \
        np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return qx, qy, qz, qw

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

def make_label(tree, robot_pose, robot_rotation, radius):
    x, y = robot_pose
    axes_points = None
    assert(isinstance(tree, TileTree))
    tile = tree.get_tile_by_coords(x, y)
    if not isinstance(tile, Tile):
        return
    neighbors = tile.neighbors

    relevant_tiles = neighbors + [tile]

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
    points_at_correct_distance = axes_points[d<res/2]
    
    vectors = points_at_correct_distance - robot_pose
    angles_rad = np.arctan2(vectors[:,1],vectors[:,0])
    angles_rad = (angles_rad - robot_rotation + 2*math.pi) % (2*math.pi)
    angles_deg = (angles_rad*180/math.pi).astype(int)
    label = label_from_angles(angles_deg)
    return label



class FigureCallbackHandler:
    def __init__(self, tree):
        self.tree = tree
        self.is_complete = False
        self.plot_length = 10
        self.radius = 5
        self.x_for_label = np.linspace(0, math.pi*2, num=360)


        angles = np.matrix(2*math.pi * np.linspace(0, 1, 40)).T

        x = np.sin(angles)
        y = np.cos(angles)
        self.raw_circle_points = np.hstack((x, y)) * self.radius

        self.setup_matplotlib()

    def onclick(self, event):
        self.p1 = np.array((event.xdata, event.ydata))
        self.circle = (self.raw_circle_points + self.p1).T
        self.is_complete = False

    def on_declick(self, event):
        self.p2 = np.array((event.xdata, event.ydata))
        self.diff = self.p2 - self.p1
        self.dist = np.linalg.norm(self.diff)
        self.orientation_vector = self.diff / self.dist
        rotation = np.arctan2(self.orientation_vector[1],self.orientation_vector[0])
        self.label = make_label(self.tree, self.p1, rotation, self.radius)
        self.is_complete = True
        self.plot_callback()

    def get_orientation_line(self):
        x1, y1 = self.p1
        x2, y2 = self.p1 + self.orientation_vector*self.plot_length
        return np.reshape(np.array([x1, x2, y1, y2]), (2, 2))

    def plot_callback(self):

        self.fig.canvas.restore_region(self.axbackground)
        self.fig.canvas.restore_region(self.ax2background)


        if self.is_complete:
            self.label_plotter.set_data(self.x_for_label, self.label)

            self.ax2.draw_artist(self.label_plotter)

            self.robot_line.set_data(self.get_orientation_line())
            self.robot_circle.set_data(self.circle)
            self.robot_points.set_offsets(np.array(self.p1))

            self.ax1.draw_artist(self.robot_line)
            self.ax1.draw_artist(self.robot_circle)
            self.ax1.draw_artist(self.robot_points)


        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)
        self.fig.canvas.flush_events()

    def setup_matplotlib(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.ax1 = self.fig.add_subplot(121)
        plot_tree(self.tree, tunnel_axis=True)
        self.ax2 = self.fig.add_subplot(122, polar = True)
        self.label_plotter, = self.ax2.plot([], lw=3)
        self.robot_line, = self.ax1.plot([], lw=3)
        self.robot_circle, = self.ax1.plot([], color="g")
        self.robot_points = self.ax1.scatter([], [], color="r")
        self.ax2.set_ylim([0,1])
        self.ax2.set_theta_zero_location("N")
        # Setup of callbacks
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        cid = self.fig.canvas.mpl_connect('button_release_event', self.on_declick)
        # Initialization of pyplot
        self.fig.canvas.draw()
        self.axbackground = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.ax2background = self.fig.canvas.copy_from_bbox(self.ax2.bbox)
        plt.show()


def main():
    # Generation of the random tree
    tree = RandomTreeGenerator(max_tiles=20)
    tree.gen_tree()
    # Initialization of the callback handler
    handler = FigureCallbackHandler(tree=tree)
    # Setup of matplotlib parameters




if __name__ == "__main__":
    main()
