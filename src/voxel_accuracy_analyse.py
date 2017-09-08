import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy import genfromtxt
from mpl_toolkits.mplot3d import axes3d

#, Y, Z = axes3d.get_test_data(0.05)

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
ax.scatter(np.asarray(coords,dtype=int)[:,[0]],np.asarray(coords,dtype=int)[:,[1]],np.asarray(coords,dtype=int)[:,[2]],
             s=0.1,c='grey',alpha=0.05)
ax.scatter(np.asarray(selected_coords,dtype=int)[:,[0]],np.asarray(selected_coords,dtype=int)[:,[1]],np.asarray(selected_coords,dtype=int)[:,[2]],
             s=0.5,c='grey',alpha=0.01)

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