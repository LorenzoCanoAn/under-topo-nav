import numpy as np
from params import *


def min_distance(angle, obj):
    distance = (angle - obj) % (np.math.pi*2)
    if distance < -np.math.pi:
        distance += np.math.pi * 2
    elif distance > np.math.pi:
        distance -= np.math.pi * 2
    distance = abs(distance)
    return distance


def array_position_to_angle(array_position):
    return((180 - array_position)/180.0 * np.math.pi + 2*np.math.pi) % (2*np.math.pi)


def filter_and_inflate_ranges(i_ranges, angle_increment):
    n_angles = i_ranges.__len__()
    filtered_ranges = np.ones(n_angles) * MAX_OA_DISTANCE
    for n, r in enumerate(i_ranges):
        if r < MAX_OA_DISTANCE:
            delta_theta = np.math.atan(INFLATION/r)
            total_increment = 0
            j = 1
            filtered_ranges[n] = min(filtered_ranges[n], r)

            while total_increment < delta_theta:
                total_increment += angle_increment
                idx = (n + j) % n_angles
                filtered_ranges[idx] = min([r, filtered_ranges[idx]])
                idx = n-j
                filtered_ranges[idx] = min([r, filtered_ranges[idx]])
                j += 1
        else:
            filtered_ranges[n] = min([MAX_OA_DISTANCE, filtered_ranges[n]])
    return filtered_ranges


def filtered_to_gallery_angles(filtered):
    max_peak = np.max(filtered)
    ratio = 0.3
    galleries_indices = np.nonzero(filtered > max_peak * ratio)[0]
    galleries_angles = []
    for index in galleries_indices:
        galleries_angles.append(
            array_position_to_angle(index))
    true_gallery_angles = []
    for a1 in galleries_angles:
        passes = True
        for a2 in true_gallery_angles:
            if min_distance(a1, a2) < 0.17:  # 10 degrees
                passes = False
        if passes:
            true_gallery_angles.append(a1)
    return true_gallery_angles


def filter_vector(vector):
    filtered = np.zeros(360)
    for i in range(360):
        to_check = vector[i]
        filtered[i] = to_check
        a = 40
        for j in range(a):
            index_inside_subsection = ((-int(a/2) + j) + i) % 356
            if vector[index_inside_subsection] > to_check:
                filtered[i] = 0
    return filtered_to_gallery_angles(filtered)

def filter_vector_with_intermediates(vector):
    filtered = np.zeros(360)
    for i in range(360):
        to_check = vector[i]
        filtered[i] = to_check
        a = 40
        for j in range(a):
            index_inside_subsection = ((-int(a/2) + j) + i) % 356
            if vector[index_inside_subsection] > to_check:
                filtered[i] = 0
    return filtered, filtered_to_gallery_angles(filtered)