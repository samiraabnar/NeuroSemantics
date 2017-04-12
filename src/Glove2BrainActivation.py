import csv

import keras
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import offsetbox
from scipy.spatial import *
from sklearn import manifold

from src.functions import *
from src.WordEmbeddingLayer import  *

if __name__ == '__main__':

    brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')
    brain_activations = brain_activations_1

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

    #selected = np.load("general_selected_100.npy")
    #selected = select_stable_voxels(brain_activations,word_set,words)
    #np.save("general_selected_500.npy",selected)
    selected = np.load("general_selected_500.npy")

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

        all_activations[the_word].append(np.tanh(np.asarray(brain_activations[x])[selected]))
        all_activations_indexes[the_word].append(x)


    all_activations_tree = cKDTree(np.tanh(np.asarray(brain_activations)[:,selected]))
    avg_all_activations_dic = {}
    for word_key in all_activations:
        avg_all_activations_dic[word_key] = [np.mean(all_activations[word_key],axis=0)]
    results_average = []
    results_all = []


    # for each pair of words as the test test
    for k in np.arange(len(the_pairs)):
        (i, j) = the_pairs[k]

        # contain representation for all words except the pairs
        training_word_representations = []
        training_activations_dic = {}
        # dictionary for keeping the activation for words except the pairs
        training_avg_activations = []
        training_activations_all = []
        training_words_all = []

        for the_word in word_set:
            if the_word != word_set[i] and the_word != word_set[j]:
                training_activations_dic[the_word] = all_activations[the_word]
                training_avg_activations.append(np.mean(all_activations[the_word],axis=0))
                training_word_representations.append(word_representations[word_set.index(the_word)])

                for activation in training_activations_dic[the_word]:
                    training_activations_all.append(activation)
                    training_words_all.append(word_representations[word_set.index(the_word)])

        training_activations_all = np.asarray(training_activations_all)
        training_words_all = np.asarray(training_words_all)
        training_avg_activations = np.asarray(training_avg_activations)
        avg_all_activations = np.asarray(avg_all_activations)
        training_words = np.asarray(training_word_representations)
        all_representations = np.asarray(all_representations)

        model = Sequential()
        model.add(Dense(input_dim=training_words.shape[1], output_dim=512,activation='linear', W_regularizer = keras.regularizers.l2(.0001)))
        model.add(Dense(input_dim= 512,
                        output_dim=training_avg_activations.shape[1], activation='tanh',W_regularizer = keras.regularizers.l2(.0001)))

        rmsprop = keras.optimizers.RMSprop(lr=0.01)
        model.compile(rmsprop, "cosine", metrics=['cosine'])
        model.fit(training_words_all,training_activations_all, batch_size=5, nb_epoch=10)

        predicted_1 = model.predict(np.asarray([np.asarray(word_representations)[i]]))
        predicted_2 = model.predict(np.asarray([np.asarray(word_representations)[j]]))

        result_1 = avereaged_match_prediction(predicted_1[0], predicted_2[0], (i, j), avg_all_activations_dic,
                                              word_set)
        result_2 = avereaged_match_prediction(predicted_1[0], predicted_2[0], (i, j), all_activations,
                                              word_set)
        results_average.append(result_1)
        results_all.append(result_2)
        print("Accuracy on average: " + str(sum(results_average) / len(results_average)))
        print("Accuracy on all: " + str(sum(results_all) / len(results_all)))


    # binary_crossentropy_loss: one layer average - Learning Rate - 0.001: 0.791525423729
    # binary_crossentropy_loss: one layer average - Learning Rate - 0.01:
    # binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.01: 0.749717514124
    # binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.01: 0.740677966102
    # binary_crossentropy_loss: two layer  train on all - accuracy on average - Learning Rate - 0.001: 0.764406779661
    # binary_crossentropy_loss: two layer  train on all - accuracy on all - Learning Rate - 0.001: 0.733333333333
    # regularization=0.001 - binary_crossentropy_loss: two layer  train on all - accuracy on average - Learning Rate - 0.001: 0.776271186441
    # regularization=0.001 - binary_crossentropy_loss: two layer  train on all - accuracy on all - Learning Rate - 0.001: 0.73615819209

    # hidden=256 regularization=0.001 - binary_crossentropy_loss: two layer  train on all - accuracy on all - Learning Rate - 0.001: 0.775141242938
    # hidden=256 regularization=0.001 - binary_crossentropy_loss: two layer  train on all - accuracy on average - Learning Rate - 0.001: 0.742372881356

    # regularization=0.001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.78813559322
    # regularization=0.001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.745197740113


    # regularization=0.001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.01: 0.770056497175
    # regularization=0.001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.01: 0.74011299435

    # regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.01:
    # regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.01:

    # regularization=0.0001 - mse_loss: one layer  train on all - accuracy on average - Learning Rate - 0.01:
    # regularization=0.0001 - mse_loss: one layer  train on all - accuracy on all - Learning Rate - 0.01:

    # regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.790960451977
    # regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.742937853107

    # relu input, relu output, regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.75593220339
    # relu input, relu output, regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.711299435028

    # relu input, relu output, regularization=0.0001 - mse_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.766101694915
    # relu input, relu output, regularization=0.0001 - mse_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.75197740113

    # tanh input, tanh output, regularization=0.0001 - mse_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.793220338983
    # tanh input, tanh output, regularization=0.0001 - mse_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.753107344633

    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on average - Learning Rate - 0.001: 0.806214689266
    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss: one layer  train on all - accuracy on all - Learning Rate - 0.001: 0.75988700565

    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss: two layer(512)  train on all - accuracy on average - Learning Rate - 0.001: 0.78813559322
    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss: two layer(512)  train on all - accuracy on all - Learning Rate - 0.001: 0.750847457627

    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss, metric=cosine: two layer(512)  train on all - accuracy on average - Learning Rate - 0.001: 0.78813559322
    # tanh input, tanh output, regularization=0.0001 - binary_crossentropy_loss, metric=cosine: two layer(512)  train on all - accuracy on all - Learning Rate - 0.001: 0.750847457627