from numpy import genfromtxt
import csv
import sys
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
import numpy as np
from scipy import *
from scipy import *
from scipy.spatial.distance import *
from sklearn import linear_model


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

sys.path.append('../../../')

from NeuroSemantics.src.WordEmbeddingLayer import *
from NeuroSemantics.src.functions import *
from sklearn import preprocessing


brain_activations_1 = genfromtxt('../../data/data.csv', delimiter=',')
brain_activations = sigmoid(brain_activations_1)

"""from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(brain_activations)
brain_activations = std_scale.transform(brain_activations)
minmax_scale = preprocessing.MinMaxScaler().fit(brain_activations)
brain_activations = minmax_scale.transform(brain_activations)
"""

words_1 = []
with open('../../data/words', 'r') as f:
    reader = csv.reader(f)
    words_1 = list(reader)

words = []
words.extend([w[0] for w in words_1])
word_set = list(set(words))
print(len(word_set))
print(words[0])
print(words[0]+" "+words[60]+" "+words[120])


wem = WordEmbeddingLayer()
wem.load_filtered_embedding("../../data/neuro_words")

embedded_words = wem.embed([[word] for word in words])
word_representations = np.asarray([e[0] for e in embedded_words])

number_of_features = len(word_representations[0])

pairs = get_pairs(60)

all_activations = []
all_features = []
the_pairs = []
all_words = []









result = []

all_activations = np.load("../all_activations_simple.npy")
the_pairs = np.load("../the_pairs_simple.npy")
all_words = np.load("../all_words_simple.npy")
all_selected = np.load("../all_selected_simple.npy")


all_features = []
results = []
for (i, j) in pairs:
    word_vectors = []
    print(str((i, j)))
    for k in np.arange(brain_activations.shape[0]):
        if words[k] != word_set[i] and words[k] != word_set[j]:
            word_vectors.append(word_representations[k])

    all_features.append(word_vectors)



for k in np.arange(len(the_pairs)):
    i = the_pairs[k][0]
    j = the_pairs[k][1]
    model = Sequential()
    model.add(Dense(input_dim=number_of_features, output_dim=all_activations[-1].shape[1], W_regularizer=l2(0.0001)))
    model.add(    Dense(input_dim=1000, output_dim=all_activations[-1].shape[1], W_regularizer=l2(0.0001)))
    model.compile("rmsprop", "mse", metrics=['mse'])
    model.fit(np.asarray(all_features[k]), sigmoid(all_activations[k]), batch_size=60, nb_epoch=20)

    predicted_1 = model.predict(np.asarray([all_features[k][i]]))
    predicted_2 = model.predict(np.asarray([all_features[k][j]]))
    result_1 = avereaged_match_prediction(predicted_1, predicted_2, the_pairs[k], brain_activations, all_selected[k], words, word_set)
    results.append(result_1)
    print(result_1)
    print("Accuracy: " + str(sum(results) / len(results)))



