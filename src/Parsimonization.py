import numpy as np
from numpy import genfromtxt
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats.stats import pearsonr
from scipy import *

import csv
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import *

import sys
sys.path.append('../../')
from matplotlib import pyplot as plt


from NeuroSemantics.src.WordEmbeddingLayer import *
from NeuroSemantics.src.functions import *



class ParsimoniousVector(object):

    def __init__(self, raw_vector, background_vector):
        if  raw_vector.shape != background_vector.shape:  #
            raise AttributeError
        self._number_of_iterations = 100 #number of em itterations
        self._alpha = 0.5 # parsimonization parameter -> higher value ->  more parsimoniztion
        self.treshold = 0.001
        self.raw_vector = ParsimoniousVector.normalize(raw_vector)
        self.background_vector = ParsimoniousVector.normalize(background_vector)
        self.parsimonious_vector = np.copy(self.raw_vector)
        self.parsimonisation()


    def parsimonisation(self):
        for _ in np.arange(self._number_of_iterations):
            self.parsimonious_vector = self.raw_vector * \
                                       ((1-self._alpha) * self.parsimonious_vector) / \
                                       ((1-self._alpha) * self.parsimonious_vector + self._alpha * self.background_vector)
            self.parsimonious_vector = ParsimoniousVector.normalize(self.parsimonious_vector)
            low_values_indices = self.parsimonious_vector < self.treshold
            self.parsimonious_vector[low_values_indices] = 0


    @staticmethod
    def normalize(mat):
        # if len(mat.shape) == 1:
        #     return mat / np.sum(mat)
        # if len(mat.shape) == 2:
        #     return mat / np.sum(mat, axis=1)[:, np.newaxis]
        if len(mat.shape) == 1:
            e_mat = np.exp(mat - np.max(mat))
            return e_mat / np.sum(e_mat)
        if len(mat.shape) == 2:
            e_mat = np.exp(mat - np.max(mat))
            return e_mat / e_mat.sum(axis=1)[:, np.newaxis]

    @staticmethod
    def scale_linear_bycolumn(rawpoints, high=10.0, low=0.0):
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng)


def plot_it(data):
    def reverse_colourmap(cmap, name='my_cmap_r'):
        return mpl.colors.LinearSegmentedColormap(name, mpl.cm.revcmap(cmap._segmentdata))
    cmap = mpl.cm.hot
    cmap_r = reverse_colourmap(cmap)
    im = plt.matshow(data, cmap = cmap_r, aspect='auto', vmin=np.min(data), vmax=np.max(data))
    plt.colorbar(im)
    plt.show()


def select_stable_voxels(the_brain_activations, words, allwords,
                         number_of_trials=6,
                         size_of_selection=1000):
    stability_matrix = np.zeros((len(the_brain_activations[0]), number_of_trials, len(words)))

    for k in np.arange(len(the_brain_activations[0])):
        word_trials = {}
        for m in np.arange(len(the_brain_activations)):
            if allwords[m] not in word_trials.keys():
                word_trials[allwords[m]] = 0
            else:
                word_trials[allwords[m]] += 1
            stability_matrix[k][word_trials[allwords[m]]][words.index(allwords[m])] = the_brain_activations[m][k]

    stability = np.zeros(len(the_brain_activations[0]))
    for k in np.arange(stability_matrix.shape[0]):
        pairs_corr = []
        for i in np.arange(number_of_trials):
            for j in np.arange(i, number_of_trials):
                pairs_corr.append(pearsonr(stability_matrix[k][i], stability_matrix[k][j])[0])
        stability[k] = mean(pairs_corr)

    ind = np.argpartition(stability, -size_of_selection)[-size_of_selection:]

    return ind



if __name__ == '__main__':

    brain_activations = genfromtxt('../data/data.csv', delimiter=',')
    words= []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words = list(reader)
    words = [w[0] for w in words]
    # print(brain_activations)
    # print(words)

    word_set = list(set(words))

    selected = select_stable_voxels(brain_activations, word_set, words, 6,1000)
    brain_activations = brain_activations[:,selected]
    brain_activations = ParsimoniousVector.scale_linear_bycolumn(brain_activations)
    brain_activations = ParsimoniousVector.normalize(brain_activations)


    word_activations_dic = {}

    for word in word_set:
        word_activations_dic[word] = np.array([])
    for i in np.arange(len(words)):
        if len(word_activations_dic[words[i]]) == 0:
            word_activations_dic[words[i]] = [brain_activations[i]]
        else:
            word_activations_dic[words[i]] = np.concatenate((word_activations_dic[words[i]],
                                                       [brain_activations[i]]), axis=0)
    # print(word_activations_dic[words[0]].shape)



    all_words_activations = np.array([])
    for word in word_set:
        # print(word)
        word_activations = word_activations_dic[word]
        # print(word_activations.shape)
        word_activations_avg = np.mean(word_activations, axis=0)
        # print(word_activations_avg.shape)
        if len(all_words_activations) == 0:
            all_words_activations = [word_activations_avg]
        else:
            all_words_activations = np.concatenate((all_words_activations,
                                                    [word_activations_avg]), axis=0)


    all_words_activations = ParsimoniousVector.normalize(all_words_activations)



    all_words_activations_parsimonized = np.array([])
    background = np.mean(all_words_activations, axis=0)
    for row in all_words_activations:
        parsimonizer = ParsimoniousVector(row, background)
        if len(all_words_activations_parsimonized) == 0:
            all_words_activations_parsimonized = [parsimonizer.parsimonious_vector]
        else:
            all_words_activations_parsimonized = np.concatenate((all_words_activations_parsimonized,
                                                    [parsimonizer.parsimonious_vector]), axis=0)


    plot_it(all_words_activations)
    plot_it(all_words_activations_parsimonized)

    print(np.sum(all_words_activations))
    print(np.sum(all_words_activations_parsimonized))

    print(all_words_activations)
    print(all_words_activations_parsimonized)

    print(np.sum(all_words_activations_parsimonized,axis=0))
    a = np.where(np.sum(all_words_activations_parsimonized,axis=0) > 0.02)[0]
    print(a)
    print(a.shape)

    selected = a

    dic, word_representations = get_word_representation("F25", words)
    number_of_features = len(word_representations[0])

    pairs = get_pairs(60)

    all_activations = []
    all_features = []
    the_pairs = []
    all_words = []
    all_selected = []

    results = []
    for (i, j) in pairs:
        activations = []
        the_pairs.append((i, j))
        word_vectors = []
        tmp_words = []
        all_tmp_words = []
        print(str((i, j)))
        for k in np.arange(brain_activations.shape[0]):
            if words[k] != word_set[i] and words[k] != word_set[j]:
                activations.append(brain_activations[k][selected])
                word_vectors.append(word_representations[k])
                if words[k] not in tmp_words:
                    tmp_words.append(words[k])
                all_tmp_words.append(words[k])

        all_features.append(word_vectors)
        all_words.append(tmp_words)

        activations = np.asarray(activations)
        model = Sequential()
        model.add(Dense(input_dim=number_of_features, output_dim=200, activation='linear'))
        model.add(Dense(input_dim=200, output_dim=activations.shape[1], activation='linear'))

        model.compile("Nadam", "mse", metrics=['mse'])
        model.fit(np.asarray(all_features[-1]), activations, batch_size=20, nb_epoch=20)
        # model.save("model_data_1_" +str((i, j)))

        predicted_1 = model.predict(np.asarray([all_features[-1][i]]))
        predicted_2 = model.predict(np.asarray([all_features[-1][j]]))
        result_1 = avereaged_match_prediction(predicted_1, predicted_2, the_pairs[-1],
                                              brain_activations[:,selected], words, word_set)
        results.append(result_1)
        print(result_1)
        print("Accuracy: " + str(sum(results) / len(results)))