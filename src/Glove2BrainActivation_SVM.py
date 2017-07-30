from sklearn.svm import SVR
from scipy.spatial import *
import csv
from functions import *
from WordEmbeddingLayer import  *
import numpy as np

if __name__ == '__main__':

    brain_activations_1 = genfromtxt('../data/data_2.csv', delimiter=',')
    brain_activations = brain_activations_1
    brain_activations = np.tanh(brain_activations - np.mean(brain_activations,axis=0))

    words = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words = list(reader)

    coords = []
    with open('../data/coords', 'r') as f:
        reader = csv.reader(f)
        coords = list(reader)

    words = [w[0] for w in words]
    word_set = list(set(words))
    print(len(word_set))

    wem = WordEmbeddingLayer()
    wem.load_filtered_embedding("../data/neuro_words")

    embedded_words = wem.embed_words(word_set)
    word_representations = embedded_words
    #dic, word_representations = get_word_representation("F25", words)
    number_of_features = len(word_representations[0])

    word_tree = cKDTree(word_representations)

    """all_activations = np.load("../models/all_activations_simple.npy")
    all_features = np.load("../models/all_features_simple.npy")
    the_pairs = np.load("../models/the_pairs_simple.npy")
    all_words = np.load("../models/all_words_simple.npy")
    all_selected = np.load("../models/all_selected_simple.npy")
    """

    """selected = select_stable_voxels(brain_activations,word_set,words,
                         number_of_trials=6,
                        size_of_selection=100)
    np.save("../models/general_selected_100_2.npy",selected)
    """
    selected = np.load("../models/general_selected_100_2.npy")

    the_pairs = np.load("../models/the_pairs_simple.npy")

    all_activations = {}
    avg_all_activations = []
    all_activations_list = []
    all_representations = []
    all_activations_indexes = {}
    for x in np.arange(len(words)):
        the_word = words[x]
        if the_word not in all_activations.keys():
            all_activations[the_word] = []
            all_activations_indexes[the_word] = []

        all_activations[the_word].append((np.asarray(brain_activations[x])[selected]))
        all_activations_indexes[the_word].append(x)


    all_activations_tree = cKDTree((np.asarray(brain_activations)[:,selected]))
    avg_all_activations_dic = {}
    for word_key in all_activations:
        avg_all_activations_dic[word_key] = [np.mean(all_activations[word_key],axis=0)]
    results_average = []
    results_all = []


    voxel_accuracies = np.zeros(len(selected))
    # for each pair of words as the test test
    for k in np.arange(len(the_pairs)):
        (i, j) = the_pairs[k]

        # contain representation for all words except the pairs
        training_word_representations = []
        training_activations_dic = {}
        training_activations_all = []
        training_words_all = []
        # dictionary for keeping the activation for words except the pairs
        training_avg_activations = []

        for the_word in word_set:
            if the_word != word_set[i] and the_word != word_set[j]:
                training_activations_dic[the_word] = all_activations[the_word]
                training_avg_activations.append(np.mean(all_activations[the_word],axis=0))
                training_word_representations.append(word_representations[word_set.index(the_word)])
                for activation in training_activations_dic[the_word]:
                    training_activations_all.append(activation)
                    training_words_all.append(word_representations[word_set.index(the_word)])



        training_avg_activations = np.asarray(training_avg_activations)
        avg_all_activations = np.asarray(avg_all_activations)
        training_words = np.asarray(training_word_representations)
        all_representations = np.asarray(all_representations)
        training_activations_all = np.asarray(training_activations_all)
        training_words_all = np.asarray(training_words_all)
        model = {}
        predicted_1 = []
        predicted_2 = []
        for h in np.arange(training_avg_activations.shape[1]):
            model[h] = SVR(C=1.0, epsilon=0.2)
            model[h].fit(training_words_all,training_activations_all[:,h])

            predicted_1.append(np.tanh(model[h].predict(np.asarray([np.asarray(word_representations)[i]]))))
            predicted_2.append(np.tanh(model[h].predict(np.asarray([np.asarray(word_representations)[j]]))))

        diff_1_1 = np.asarray(predicted_1)[:,0] - np.asarray(avg_all_activations_dic[word_set[i]][0])
        diff_1_2 = np.asarray(predicted_1)[:,0] - np.asarray(avg_all_activations_dic[word_set[j]][0])
        diff_2_2 = np.asarray(predicted_2)[:,0] - np.asarray(avg_all_activations_dic[word_set[j]][0])
        diff_2_1 = np.asarray(predicted_2)[:,0] - np.asarray(avg_all_activations_dic[word_set[i]][0])

        voxel_accuracies += (diff_1_1 + diff_2_2) < (diff_1_2+diff_2_1)



        result_1 = avereaged_match_prediction(predicted_1, predicted_2, (i, j), avg_all_activations_dic,
                                              word_set)

        result_2 = avereaged_match_prediction(predicted_1, predicted_2, (i, j), all_activations,
                                              word_set)
        results_average.append(result_1)
        results_all.append(result_2)
        print("Accuracy on average: " + str(sum(results_average) / len(results_average)))
        print("Accuracy on all: " + str(sum(results_all) / len(results_all)))

        """
        if len(results_all) % 20 == 0:

            from pylab import *

            plot(np.arange(len(voxel_accuracies)),voxel_accuracies / len(results_all))
            grid(True)
            show()
        """
    # sigmoid: train on average - Accuracy on all: 0.736723163842
    # sigmoid: train on all - Accuracy on all: 0.800564971751
    # not sigmoid - train on all - accuracy on all: 0.79604519774