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
    (i, j) = the_pairs[k]

    normalized =[]
    words_k = []
    activations = {}
    avg_activations = []
    all_activations = {}
    avg_all_activations = []
    all_words_k = []
    all_representations = []
    for x in np.arange(len(words)):
        all_representations.append(word_representations[x])
        if words[x] != word_set[j] and words[x] != word_set[j]:
            normalized.append(word_representations[x])
            if words[x] not in activations.keys():
                activations[words[x]] = []
            activations[words[x]].append(brain_activations[x][all_selected[k]])
            if words[x] not in words_k:
                words_k.append(words[x])
        if words[x] not in all_activations.keys():
            all_activations[words[x]] = []
        all_activations[words[x]].append(brain_activations[x][all_selected[k]])
        all_words_k.append(words[x])

    for x in np.arange(len(words_k)):
        avg_activations.append(np.mean(activations[words_k[x]], axis=0))

    for x in np.arange(len(all_words_k)):
        avg_all_activations.append(np.mean(all_activations[all_words_k[x]], axis=0))

    avg_activations = np.concatenate(list(activations.values()),axis=0) #np.asarray(avg_activations)
    avg_all_activations = np.concatenate(list(all_activations.values()),axis=0) #np.asarray(avg_all_activations)
    normalized = np.asarray(normalized)


    model = Sequential()
    #model.add(Dense(input_dim=normalized.shape[1], output_dim=1000))
    model.add(Dense(input_dim= normalized.shape[1],#1000,#number_of_features,
                    output_dim=avg_activations.shape[1], activation='linear'))

    rmsprop = keras.optimizers.RMSprop(lr=0.009)
    model.compile(rmsprop, "mse", metrics=['mse'])
    model.fit(normalized,avg_activations,batch_size=58,nb_epoch=100)


    predicted_1 = model.predict(np.asarray([(np.asarray(all_representations))[i]]))
    predicted_2 =  model.predict(np.asarray([(np.asarray(all_representations))[j]]))

    result_1 = match_prediction(predicted_1[0], predicted_2[0], (i,j), avg_all_activations,
                                all_words_k, word_set)
    results.append(result_1)
    print(result_1)
    print("Accuracy: " + str(sum(results) / len(results)))




#avg: Accuracy: 0.948587570621
#not avg: Accuracy: 0.731638418079
