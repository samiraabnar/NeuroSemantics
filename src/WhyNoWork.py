import csv

import keras
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import offsetbox
from scipy.spatial import *
from sklearn import manifold

from src.functions import *
from src.WordEmbeddingLayer import  *

def plot_embedding(features, classes, labels, title=None):
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    features = (features - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(features.shape[0]):
        plt.text(features[i, 0], features[i, 1], str(labels[i]),
                 color=plt.cm.Set1(float(classes[i] / 60)),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(features.shape[0]):
            dist = np.sum((features[i] - shown_images) ** 2, 1)
            # if np.min(dist) < 4e-3:
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


def plot_distribution_t_sne(activations, words, labels):
    print("Computing t-SNE embedding")

    x = np.asarray(activations)
    # x = preprocessing.normalize(x, norm='l2')

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2, n_iter=20000, early_exaggeration=10,
                         learning_rate=300, method="exact")
    x_tsne = tsne.fit_transform(x)

    plot_embedding(x_tsne, np.asarray(words), labels,
                   "t-SNE embedding of the brain activations")

    plt.show()


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


    dic, word_representations = get_word_representation("F25", word_set)
    number_of_features = len(word_representations[0])

    word_tree = cKDTree(word_representations)

    """all_activations = np.load("../models/all_activations_simple.npy")
    all_features = np.load("../models/all_features_simple.npy")
    the_pairs = np.load("../models/the_pairs_simple.npy")
    all_words = np.load("../models/all_words_simple.npy")
    all_selected = np.load("../models/all_selected_simple.npy")
    """

    selected = np.load("general_selected_500.npy")
    the_pairs = np.load("../models/the_pairs_simple.npy")

    all_activations = {}
    avg_all_activations = []
    all_representations = []

    for x in np.arange(len(words)):
        the_word = words[x]
        if the_word not in all_activations.keys():
            all_activations[the_word] = []
        all_activations[the_word].append(np.asarray(sigmoid(brain_activations[x])[selected]))

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
        # model.add(Dense(input_dim=traiall_representationsning_words.shape[1], output_dim=64))
        model.add(Dense(input_dim=training_words_all.shape[1],  # 64,#number_of_features,
                        output_dim=training_avg_activations.shape[1], activation='sigmoid'))

        rmsprop = keras.optimizers.RMSprop(lr=0.01)
        model.compile(rmsprop, "binary_crossentropy", metrics=['mse'])
        model.fit(training_words_all,training_activations_all, batch_size=5, nb_epoch=20)

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

    """
    brain_activations_1_selected = np.asarray(brain_activations)[:, selected]
    coords_selected = np.asarray(coords)[selected, :]
    print(mean(brain_activations[0]))
    print(max(brain_activations[0]))
    print(min(brain_activations[0]))
    fMRI = np.zeros((51, 61, 23)) + np.min(brain_activations_1_selected)
    print(len(coords))
    print(int(coords[0][2]) - 1)
    print(fMRI[int(coords[0][0]) - 1][int(coords[0][1]) - 1][int(coords[0][2]) - 1])

    for i in np.arange(len(coords_selected)):
        fMRI[int(coords_selected[i][0]) - 1][int(coords_selected[i][1]) - 1][int(coords_selected[i][2]) - 1] = \
            predicted_1[0][i]

    from matplotlib import pyplot as plt

    # fMRI = fMRI / np.sum(fMRI)

    plt.figure(1)
    for z in np.arange(23):
        plt.subplot(5, 5, z + 1)
        im = plt.imshow(fMRI[:, :, z], aspect='auto', vmin=np.min(fMRI), vmax=np.max(fMRI))
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)

    plt.colorbar(im)
    plt.show()
    """
    # Accuracy on averaged: 0.794350282486
    # linear output - mse loss - one layer  - Train on all, Accuracy on average: 0.814689265537
    # linear output - mse loss - one layer  - Train on all, Accuracy on all: 0.779096045198
    # linear output - binary cross_entropy loss - one layer  - Train on all, Accuracy on average: 0.801694915254
    # linear output - binary cross_entropy loss - one layer  - Train on all, Accuracy on all: 0.766101694915
    # sigmoid output - binary cross_entropy loss - one layer  - Train on all, Accuracy on average: 0.830508474576
    # sigmoid output - binary cross_entropy loss - one layer  - Train on all, Accuracy on all: 0.789830508475
    # sigmoid output - mse loss - one layer  - Train on all, Accuracy on average:
    # sigmoid output - mse loss - one layer  - Train on all, Accuracy on all: