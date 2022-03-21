import numpy as np
from scipy.stats import norm
import torch


def gen_normal_distribution( n_elements):
    nd = norm(loc=2, scale=0.5)
    n = 0
    array = []
    for i in np.arange(0, 4, 4/n_elements):
        if n == n_elements:
            break
        n += 1
        array.append(nd.pdf(i))

    return array

def reshape_tensor(tensor):

    aperture = 40
    normal_array = gen_normal_distribution(aperture)
    centers = (tensor == torch.max(tensor)).nonzero(as_tuple=True)
    tensor *= 0
    for center in centers:
        for i in range(aperture):
            index = int(center - aperture/2 + i) % 360

            tensor[index] = normal_array[i]
    return tensor