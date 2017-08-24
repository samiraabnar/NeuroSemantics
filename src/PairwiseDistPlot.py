import os
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse
from functions import *
from WordEmbeddingLayer import *


class PairwiseDistPlot(object):

    def __init__(self,metric="cosine"):
        self.metric = metric


    def compute_dists(self,list_of_vectors):
        return pairwise_distances(list_of_vectors,metric=self.metric, n_jobs=1)

    def compute_and_print_dists(self,list_of_vectors,id_dic):
        dists =  self.compute_dists(list_of_vectors)

        return dists

    def compute_and_plot_dists(self,list_of_vectors,id_dic):
        dists = self.compute_dists(list_of_vectors)

        # Display matrix
        plt.imshow(dists, cmap=plt.cm.gray, interpolation='none')
        plt.xticks(np.arange(len(id_dic)),id_dic)
        plt.yticks(np.arange(len(id_dic)),id_dic)
        plt.xticks(rotation=90)
        plt.show()

    def compute_and_plot_diff_dists(self,list_of_vectors_1,list_of_vectors_2,id_dic):
        dists_1 = np.tanh(self.compute_dists(list_of_vectors_1))
        dists_2 = np.tanh(self.compute_dists(list_of_vectors_2))

        # Display matrix
        plt.imshow(abs(dists_1 - dists_2), cmap=plt.cm.gray, interpolation='none')
        plt.xticks(np.arange(len(id_dic)),id_dic)
        plt.yticks(np.arange(len(id_dic)),id_dic)
        plt.xticks(rotation=90)
        plt.show()



if __name__ == '__main__':

    pdp = PairwiseDistPlot()

    fMRI_data_path = "../data/"
    fMRI_data_filename = "data_"
    fMRI_data_postfix = ".csv"
    subject_id = str(1)
    subject = subject_id

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Single Layer Feed Forward Network for Brain Activation Prediction Task')
    parser.add_argument('--subject', '-s', type=str, nargs='?',
                        help='An optional integer specifying the subject id', default="1")

    args = parser.parse_args()
    print("subject id %s" % args.subject)

    fMRI_file = fMRI_data_path + fMRI_data_filename + args.subject + fMRI_data_postfix

    brain_activations_1 = genfromtxt(fMRI_file, delimiter=',')
    brain_activations = brain_activations_1 - np.mean(brain_activations_1, axis=0)
    brain_activations = np.tanh(brain_activations)

    words_1 = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words_1 = list(reader)

    words = []
    words.extend([w[0] for w in words_1])


    conds_1 = []
    with open('../data/conds', 'r') as f:
        reader = csv.reader(f)
        conds_1 = list(reader)
    conds = [int(c[0]) for c in conds_1]

    cond_sorted_set = np.argsort(conds[:60])
    word_set = np.asarray(words)[cond_sorted_set]

    print("number of words: %d " % len(word_set))

    selected_file_name = "general_selected_500_" + subject + ".npy"

    if not os.path.isfile(selected_file_name):
        selected = select_stable_voxels(brain_activations_1, word_set, words, number_of_trials=6,
                                        size_of_selection=500)
        np.save(selected_file_name, selected)

    selected = np.load(selected_file_name)

    mean_Activations = []

    words = np.asarray(words)
    for word in word_set:
        indices = np.where(words == word)[0]
        mean_Activations.append(np.mean(brain_activations[indices, :], axis=0))

    mean_Activations = np.asarray(mean_Activations)



    pdp.compute_and_plot_dists(mean_Activations[:,selected],word_set)

    words = word_set
    wem = WordEmbeddingLayer()
    wem.load_filtered_embedding("../data/neuro_words")

    embedded_words = wem.embed_words(word_set)
    word_representations = embedded_words

    pdp.compute_and_plot_diff_dists(mean_Activations[:,selected],word_representations,word_set)