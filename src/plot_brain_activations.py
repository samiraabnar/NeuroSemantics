import matplotlib
import matplotlib.pyplot as plt
import sys
from scipy import *
from scipy.spatial import *
import csv
import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import axes3d
from functions import *



if __name__ == '__main__':

    coords = []
    with open('../data/coords', 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)


    all_voxels = brain_activations_1 = genfromtxt('../data/data_1.csv', delimiter=',')

    all_voxels = np.asarray(all_voxels[2])
    all_voxels[all_voxels < 0] = 0
    #all_voxels[all_voxels > 4] = 12

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    low_coords = []
    for i in np.arange(len(coords)):
        print(coords[i])
        if int(all_voxels[i]) <= 4:
            low_coords.append(i)

    low_coords = np.asarray(low_coords)

    high_coords = []
    for i in np.arange(len(coords)):
        print(coords[i])
        if int(all_voxels[i]) > 8:
            high_coords.append(i)

    high_coords = np.asarray(high_coords)

    print("high coords:",high_coords.shape)



    c = ax.scatter(np.asarray(coords,dtype=int)[:,[0]],np.asarray(coords,dtype=int)[:,[1]],np.asarray(coords,dtype=int)[:,[2]],
                 s=1.0,cmap="jet",c=np.tanh(np.asarray(all_voxels)[:]),alpha=1.0)

    """c = ax.scatter(np.asarray(coords, dtype=int)[high_coords, [0]], np.asarray(coords, dtype=int)[high_coords, [1]],
               np.asarray(coords, dtype=int)[high_coords, [2]],
               s=1, c=color_codes[high_coords], alpha=1.0)
    """
    #ax.scatter(np.asarray(coords, dtype=int)[high_coords, [0]], np.asarray(coords, dtype=int)[high_coords, [1]],
    #               np.asarray(coords, dtype=int)[high_coords, [2]],
    #               s=1, cmap=colors.get_cmap(), c=all_voxels[high_coords], alpha=0.9)

    print(np.min(all_voxels))
    print(np.max(all_voxels))
    print(np.mean(all_voxels))

    angel = 180
    zangle = 270
    ax.view_init(zangle,angel)
    plt.colorbar(c)
    plt.show()