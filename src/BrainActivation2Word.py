import keras
from numpy import genfromtxt
import csv
import sys
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import *
from scipy.spatial.distance import *
from scipy.spatial import *
sys.path.append('../../')
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold

from NeuroSemantics.src.WordEmbeddingLayer import *
from NeuroSemantics.src.functions import *

def plot_embedding(features, classes, labels, title=None):
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    features = (features - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(features.shape[0]):
        plt.text(features[i, 0], features[i, 1], str(labels[i]),
                 color=plt.cm.Set1(float(classes[i]/60)),
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

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2,n_iter=20000,early_exaggeration=10,learning_rate=300, method="exact")
    X_tsne = tsne.fit_transform(x)

    plot_embedding(X_tsne, np.asarray(words), labels,
                                       "t-SNE embedding of the brain activations")

    plt.show()



brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')
brain_activations = brain_activations_1

words_1 = []
with open('../data/words', 'r') as f:
    reader = csv.reader(f)
    words_1 = list(reader)

words = []
words.extend([w[0] for w in words_1])
word_set = list(set(words))
print(len(word_set))
print(words[0])
print(words[0]+" "+words[60]+" "+words[120])


#wem = WordEmbeddingLayer()
#wem.load_filtered_embedding("../data/neuro_words")

#embedded_words = wem.embed(words)
#embedded_words_2 = np.asarray([e[0] for e in embedded_words])

dic,word_representations = get_word_representation("F25",words)
number_of_features = len(word_representations[0])

word_tree = cKDTree(word_representations)


all_activations = np.load("../models/all_activations_simple.npy")
all_features = np.load("../models/all_features_simple.npy")
the_pairs = np.load("../models/the_pairs_simple.npy")
all_words = np.load("../models/all_words_simple.npy")
all_selected = np.load("../models/all_selected_simple.npy")


results = []
for  k in np.arange(len(the_pairs)):
    global_all_activations_k = np.zeros((brain_activations.shape[0]-12,brain_activations.shape[1])) + np.min(brain_activations)
    global_all_brain_activations = np.zeros(brain_activations.shape) + np.min(brain_activations)
    (i, j) = the_pairs[k]

    normalized =[]
    words_k = []
    activations = {}
    avg_activations = []
    for x in np.arange(len(word_set)):
            if word_set[x] != word_set[j] and word_set[x] != word_set[j]:
                normalized.append(word_representations[x])
                if word_set[x] not in activations.keys():
                    activations[word_set[x]] = []
                activations[word_set[x]].append(brain_activations[x][all_selected[k]])
                if word_set[x] not in words_k:
                    words_k.append(word_set[x])

    for x in np.arange(len(words_k)):
        avg_activations.append(np.mean(activations[words_k[x]], axis=1))

    avg_activations = np.asarray(avg_activations)

    normalized = np.asarray(normalized)
    activations = softmax(np.asarray(avg_activations))
    m = 0
    for x in np.arange(global_all_activations_k.shape[0]):
        if x!= i and x!=j:
            for s in all_selected[k]:
                global_all_activations_k[m][s] = brain_activations[x][s]
            m += 1

    global_all_activations_k = softmax(global_all_activations_k)

    for x in np.arange(global_all_brain_activations.shape[0]):
        for s in all_selected[k]:
            global_all_brain_activations[x][s] = brain_activations[x][s]

    model = Sequential()
    model.add(Dense(input_dim=activations.shape[1], output_dim=1000))
    model.add(Dense(input_dim= 1000,#number_of_features,
                    output_dim=normalized.shape[1], activation='linear'))

    rmsprop = keras.optimizers.RMSprop(lr=0.009)
    model.compile(rmsprop, "mse", metrics=['mse'])
    model.fit(normalized,avg_activations,batch_size=58,nb_epoch=100)

    """w = model.layers[0].weights[0].eval()

    plt.title("Weight for Predicting Nouns from Brain Activations")
    plt.imshow(w, interpolation='none', aspect='auto',
               cmap='gray')
    plt.show()"""

    dd,ii = word_tree.query(model.predict(np.asarray([brain_activations[i][all_selected[k]]])))
    results.append(words[ii[0]] == words[i])
    dd, ii = word_tree.query(model.predict(np.asarray([brain_activations[j][all_selected[k]]])))
    results.append(words[ii[0]] == words[j])
    print("Accuracy: " + str(sum(results) / len(results)))


