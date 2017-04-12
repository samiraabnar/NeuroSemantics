import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold
from sklearn import preprocessing
from numpy import genfromtxt
import csv

import sys
sys.path.append('../../')
from NeuroSemantics.src.functions import *

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2, activity_l2


def plot_embedding(features, classes, labels, title=None):
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    features = (features - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(features.shape[0]):
        plt.text(features[i, 0], features[i, 1], str(labels[i]),
                 color=plt.cm.Set1(float(classes[i])),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(features.shape[0]):
            dist = np.sum((features[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
                # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [features[i]]]
            """imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)"""
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



def plot_distribution_t_SNE(activations,words,labels):
    print("Computing t-SNE embedding")

    x = np.asarray(activations)
    #x = preprocessing.normalize(x, norm='l2')

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2,n_iter=20000,early_exaggeration=4,learning_rate=200, method="exact")
    X_tsne = tsne.fit_transform(x)

    plot_embedding(X_tsne, np.asarray(words), labels,
                                       "t-SNE embedding of the brain activations")

    plt.show()


if __name__ == '__main__':
    from sklearn.preprocessing import scale
    brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')
    brain_activations = brain_activations_1#scale(brain_activations_1, axis=1, with_mean=True, with_std=True, copy=True)#stats.zscore(brain_activations_1,axis=1)


    words_1 = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words_1 = list(reader)

    conds_1 = []
    with open('../data/conds', 'r') as f:
        reader = csv.reader(f)
        conds_1 = list(reader)

    words = []
    words.extend([w[0] for w in words_1])
    word_set = list(set(words))

    conds = [int(c[0]) for c in conds_1]

    dic, word_representations = get_word_representation("F25", words)

    #plot_distribution_t_SNE(np.asarray(brain_activations)[:,all_selected[0]],[word_set.index(word) for word in words],words)

    number_of_features = len(word_representations[0])

    activations = brain_activations
    selected = select_stable_voxels(brain_activations, word_set, words, 6)
    acts = np.asarray(activations)[:,selected] #* np.sum(w.get_weights()[0],axis=1)

    print(acts.shape)
    plot_distribution_t_SNE(acts,
                            conds# [word_set.index(word) for word in words]
                            , words)

