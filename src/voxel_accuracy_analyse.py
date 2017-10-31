import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
from mpl_toolkits.mplot3d import axes3d

#, Y, Z = axes3d.get_test_data(0.05)

def all():
    model_1 = "dep"
    model_2 = "word2vec"

    v_acc_model_1 = np.load("v_acc_"+model_1+"_limited.npy")

    v_acc_model_2 = np.load("v_acc_"+model_2+"_limited.npy")

    #plt.plot(np.arange(v_acc_model_2.shape[0]), v_acc_model_2)
    #plt.plot(np.arange(v_acc_model_1.shape[0]), v_acc_model_1)
    #plt.show()

    coords = []
    with open('../data/coords', 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)

    selected = np.load("general_selected_500_1.npy")

    selected_coords = np.asarray(coords)[selected,:]


    #all_voxels = brain_activations_1 = genfromtxt('../data/data_1.csv', delimiter=',')

    fMRI = np.zeros((51,61,23))

    #for i in np.arange(coords):
    #    fMRI[int(coords[i][0]) - 1][int(coords[i][1]) - 1][int(coords[i][2]) - 1] = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    coord = np.asarray(coords,dtype=int)
    #ax.scatter(np.asarray(coords,dtype=int)[:,[0]],np.asarray(coords,dtype=int)[:,[1]],np.asarray(coords,dtype=int)[:,[2]],
    #             s=0.1,c='grey',alpha=0.05)
    #ax.scatter(np.asarray(selected_coords,dtype=int)[:,[0]],np.asarray(selected_coords,dtype=int)[:,[1]],np.asarray(selected_coords,dtype=int)[:,[2]],
    #             s=0.5,c='grey',alpha=0.01)

    model_1_best = selected[np.argsort(v_acc_model_1)[-50:]]

    model_1_coord =  np.asarray(coords)[model_1_best,:]
    ax.scatter(np.asarray(model_1_coord,dtype=int)[:,[0]],
               np.asarray(model_1_coord,dtype=int)[:,[1]],
               np.asarray(model_1_coord,dtype=int)[:,[2]],
                 s=2,c='r',alpha=0.9)


    model_2_best = selected[np.argsort(v_acc_model_2)[-50:]]

    model_2_coord =  np.asarray(coords)[model_2_best,:]
    ax.scatter(np.asarray(model_2_coord,dtype=int)[:,[0]],
               np.asarray(model_2_coord,dtype=int)[:,[1]],
               np.asarray(model_2_coord,dtype=int)[:,[2]],
                 s=2,c='b',alpha=0.9)


    common_best = []
    for s in model_1_best:
        if s in model_2_best:
            common_best.append(s)

    common_coord = np.asarray(coords)[common_best,:]

"""
ax.scatter(np.asarray(common_coord,dtype=int)[:,[0]],
           np.asarray(common_coord,dtype=int)[:,[1]],
           np.asarray(common_coord,dtype=int)[:,[2]],
             s=2,c='g',alpha=0.9)

ax.set_axis_off()
angel = 180
zangle = 0
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 90
zangle = 0
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 270
zangle = 0
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 360
zangle = 0
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 360
zangle = 90
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 360
zangle = 270
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

angel = 150
zangle = -30
ax.view_init(zangle,angel)
fig.savefig(model_1+"_"+model_2+str(angel)+str(zangle)+".png",format="png",transparent=True,dpi=3000)

print('Hi')
"""

import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin




def plot_activations_for_words(coords,brain_activations,words):
    X = np.asarray(coords, dtype=int32)[:, 0]
    Y = np.asarray(coords, dtype=int32)[:, 1]
    Z = np.asarray(coords, dtype=int32)[:, 2]


    for i in np.arange(len(words)):
        fig = plt.figure(figsize=(8, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(words[i])

        the_fourth_dimension = brain_activations[i]
        colors = cm.jet(the_fourth_dimension)
        colmap = cm.ScalarMappable(cmap=cm.jet)
        colmap.set_array(the_fourth_dimension)
        yg = ax.scatter(X, Y, Z, c=colors, marker=',', s=1, alpha=0.5)

        # ax.plot_surface(X, Y, Z, color=colors)
        # yg = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=colors)
        cb = fig.colorbar(colmap)
        ax.set_axis_off()
        angel = 180
        zangle = 270
        ax.view_init(zangle, angel)

        fig.savefig(words[i]+"_activations_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True, dpi=720)
        fig.clf()

def plot_accuracy(all_coords,coords,accuracies,model_name):
    X = np.asarray(coords, dtype=int32)[:, 0]
    Y = np.asarray(coords, dtype=int32)[:, 1]
    Z = np.asarray(coords, dtype=int32)[:, 2]

    fMRI = np.zeros((51, 61, 23))

    left_accuracies = []
    right_accuracies = []
    x_left = []
    y_left = []
    z_left = []
    x_right = []
    y_right = []
    z_right = []
    for i in np.arange(coords.shape[0]):
        fMRI[int(X[i])][int(Y[i])][int(Z[i])] = accuracies[i]
        if(X[i] < 26):
            left_accuracies.append(accuracies[i])
            x_left.append(X[i])
            y_left.append(Y[i])
            z_left.append(Z[i])
        else:
            right_accuracies.append(accuracies[i])
            x_right.append(X[i])
            y_right.append(Y[i])
            z_right.append(Z[i])


    fig = plt.figure(figsize=(8, 6))


    ax = fig.add_subplot(111, projection='3d')

    the_fourth_dimension = np.tanh(left_accuracies)
    colors = cm.jet(the_fourth_dimension)
    colmap = cm.ScalarMappable(cmap=cm.jet)
    colmap.set_array(the_fourth_dimension)
    ax.scatter(np.asarray(all_coords, dtype=int)[:, [0]],
               np.asarray(all_coords, dtype=int)[:, [1]],
               np.asarray(all_coords, dtype=int)[:, [2]], s=0.1,c='grey',alpha=0.05)
    yg = ax.scatter(x_left, y_left, z_left, c=colors, marker=',', s=1, alpha=0.5)

    # ax.plot_surface(X, Y, Z, color=colors)
    # yg = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=colors)
    cb = fig.colorbar(colmap)
    ax.set_axis_off()
    angel = 180
    zangle = 270
    ax.view_init(zangle, angel)

    fig.savefig("left_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    angel = 180
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig("left_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    angel = 0
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig("left_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    fig.clf()

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection='3d')

    the_fourth_dimension = np.tanh(right_accuracies)
    colors = cm.jet(the_fourth_dimension)
    colmap = cm.ScalarMappable(cmap=cm.jet)
    colmap.set_array(the_fourth_dimension)
    ax.scatter(np.asarray(all_coords, dtype=int)[:, [0]],
               np.asarray(all_coords, dtype=int)[:, [1]],
               np.asarray(all_coords, dtype=int)[:, [2]], s=0.1, c='grey', alpha=0.05)
    yg = ax.scatter(x_right, y_right, z_right, c=colors, marker=',', s=1, alpha=0.5)

    # ax.plot_surface(X, Y, Z, color=colors)
    # yg = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=colors)
    cb = fig.colorbar(colmap)
    ax.set_axis_off()
    angel = 180
    zangle = 270
    ax.view_init(zangle, angel)

    fig.savefig("right_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    angel = 180
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig("right_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    angel = 0
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig("right_"+model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png", transparent=True,
                dpi=720)

    fig.clf()

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection='3d')

    the_fourth_dimension = np.tanh(accuracies)
    colors = cm.jet(the_fourth_dimension)
    colmap = cm.ScalarMappable(cmap=cm.jet)
    colmap.set_array(the_fourth_dimension)
    ax.scatter(np.asarray(all_coords, dtype=int)[:, [0]],
               np.asarray(all_coords, dtype=int)[:, [1]],
               np.asarray(all_coords, dtype=int)[:, [2]], s=0.1, c='grey', alpha=0.05)
    yg = ax.scatter(X, Y, Z, c=colors, marker=',', s=1, alpha=0.5)

    # ax.plot_surface(X, Y, Z, color=colors)
    # yg = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=colors)
    cb = fig.colorbar(colmap)
    ax.set_axis_off()
    angel = 180
    zangle = 270
    ax.view_init(zangle, angel)

    fig.savefig(model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png",
                transparent=True,
                dpi=720)

    angel = 180
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig(model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png",
                transparent=True,
                dpi=720)

    angel = 0
    zangle = 0
    ax.view_init(zangle, angel)

    fig.savefig(model_name + "_accuracies_0_" + str(angel) + str(zangle) + ".png", format="png",
                transparent=True,
                dpi=720)

    fig.clf()


def plot_accuracy_2d(all_coords,coords,accuracies,model_name):
    X = np.asarray(coords, dtype=int32)[:, 0]
    Y = np.asarray(coords, dtype=int32)[:, 1]
    Z = np.asarray(coords, dtype=int32)[:, 2]

    fMRI = np.zeros((X.shape[0],Y.shape[0],Z.shape[0]))

    for i in np.arange(coords.shape[0]):
        fMRI[int(X[i])][int(Y[i])][int(Z[i])] = accuracies[i]

    accuracies_2d = np.sum(fMRI,axis=2)

    accuracies_2d_flattened = []
    for i in np.arange(coords.shape[0]):
        accuracies_2d_flattened.append(accuracies_2d[X[i]][Y[i]])

    fig = plt.figure()

    ax = fig.add_subplot(111)

    the_fourth_dimension = np.tanh(accuracies_2d_flattened)
    colors = cm.jet(the_fourth_dimension)
    colmap = cm.ScalarMappable(cmap=cm.jet)
    colmap.set_array(the_fourth_dimension)
    ax.scatter(np.asarray(all_coords, dtype=int)[:, [0]],
               np.asarray(all_coords, dtype=int)[:, [1]]
               , s=0.1, c='grey', alpha=0.5)
    yg = ax.scatter(X, Y, c=colors, marker=',', s=1, alpha=1)

    # ax.plot_surface(X, Y, Z, color=colors)
    # yg = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, facecolors=colors)
    cb = fig.colorbar(colmap)
    ax.set_axis_off()

    fig.savefig(model_name + "_accuracies_2D_0"+ ".png", format="png",
                transparent=True,
                dpi=720)



    fig.clf()


if __name__ == '__main__':
    words_1 = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words_1 = list(reader)

    words = []
    words.extend([w[0] for w in words_1])

    coords = []
    with open('../data/coords', 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)

    selected = np.load("general_selected_500_1.npy")

    selected_coords = np.asarray(coords)[selected, :]

    brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')

    model_1 = "dep"
    model_2 = "word2vec"

    v_acc_model_1 = np.load("v_acc_"+model_1+"_limited.npy")
    v_acc_model_2 = np.load("v_acc_"+model_2+"_limited.npy")

    plot_accuracy_2d(all_coords=coords,coords=selected_coords,accuracies=v_acc_model_2,model_name=model_2)
    #plot_activations_for_words(coords=coords,brain_activations=brain_activations_1,words=words[:60])