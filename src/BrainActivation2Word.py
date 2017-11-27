import keras
import csv
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from scipy import *
from scipy.spatial import *

from WordEmbeddingLayer import *
from functions import *

brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')
brain_activations = brain_activations_1

words_1 = []
with open('../data/words', 'r') as f:
    reader = csv.reader(f)
    words_1 = list(reader)

words = []
words.extend([w[0] for w in words_1])
word_set = list(set(words))
print("number of words: %d " % len(word_set))

wem = WordEmbeddingLayer()
wem.load_filtered_embedding("../data/neuro_words")


embedded_words = wem.embed_words(word_set)
word_representations = embedded_words

#dic,word_representations = get_word_representation("F25",words)
number_of_features = len(word_representations[0])

word_tree = cKDTree(word_representations)

#all_features = np.load("../models/all_features_simple.npy")
the_pairs = np.load("../models/the_pairs_simple.npy")
all_words = np.load("../models/all_words_simple.npy")
all_selected = select_stable_voxels(brain_activations_1,word_set,words,number_of_trials=6,
                         size_of_selection=3000)

#np.load("../models/all_selected_simple.npy")

results = []
for k in np.arange(len(the_pairs)):
    (i, j) = the_pairs[k]

    word_vectors = []
    activations = {}
    avg_activations = []
    all_activations = []
    word_reps_index = {}
    for x in np.arange(len(words)):
        the_word = words[x]
        if word_set.index(the_word) != j and word_set.index(the_word) != i:
            if the_word not in activations.keys():
                activations[the_word] = []
                word_reps_index[the_word] = []
            activations[the_word].append(brain_activations[x][all_selected])
            all_activations.append(brain_activations[x][all_selected])
            word_reps_index[the_word].append(x)
        #word_vectors.append(word_representations[word_set.index(the_word)])

    for w in activations.keys():
        avg_activations.append(np.mean(activations[w], axis=0))
        word_vectors.append(word_representations[word_reps_index[w][0]])

    # avg_activations = np.asarray(avg_activations)

    word_vectors = np.asarray(word_vectors)
    activations = (np.asarray(avg_activations))

    model = Sequential()
    model.add(Dense(input_dim=activations.shape[1],
#                    output_dim=100))
#    model.add(Dense(input_dim=100,
                    output_dim=word_vectors.shape[1], activation='linear'))

    rmsprop = keras.optimizers.RMSprop(lr=0.0001)
    model.compile(rmsprop, "mse", metrics=['mse'])
    model.fit(activations, word_vectors, batch_size=10, nb_epoch=200)

    """w = model.layers[0].weights[0].eval()

    plt.title("Weight for Predicting Nouns from Brain Activations")
    plt.imshow(w, interpolation='none', aspect='auto',
               cmap='gray')
    plt.show()"""

    dd, ii = word_tree.query(model.predict(np.asarray([brain_activations[i][all_selected]])))
    print(str(ii[0]) + " " + str(i) + " " + words[i] + " " + words[ii[0]])
    results.append(words[ii[0]] == words[i])
    dd, ii = word_tree.query(model.predict(np.asarray([brain_activations[j][all_selected]])))
    results.append(words[ii[0]] == words[j])
    print("Accuracy: " + str(sum(results) / len(results)))
