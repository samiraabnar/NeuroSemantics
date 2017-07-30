import os

import itertools
import numpy as np
import pylab
import scipy
import tensorflow as tf
import csv
from scipy import *
from WordEmbeddingLayer import *
from functions import *
from sklearn.model_selection import KFold, cross_val_score


class LRModel(object):
    LOGDIR = './LRModel_graphs'

    def __init__(self, input_dim, output_dim, hidden_dim=50, training_steps=300):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = 0.09
        self.training_steps = training_steps

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.define_model()
        self.sess.run(tf.global_variables_initializer())

    def define_model(self):
        with tf.name_scope('input'):
            self.word_representation = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32, name='x-input')
            self.brain_representation = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32, name='y-input')

        with tf.name_scope('model'):
            self.W_1 = tf.Variable(tf.random_normal([self.input_dim, self.hidden_dim],stddev=0.1),
                                   name='W_1')
            self.bias_1 = tf.Variable(tf.zeros([self.hidden_dim]), name='bias_1')
            #self.W_2 = tf.Variable(tf.random_uniform([self.hidden_dim, self.output_dim], minval=-.1, maxval=.1),
            #                       name='W_2')
            #self.bias_2 = tf.Variable(tf.zeros([self.output_dim]), name='bias_2')
            dropped_input = tf.nn.dropout(self.word_representation, .8)
            # This is the same as y = tf.add(tf.mul(m, x_placeholder), b), but looks nicer
            self.h = tf.nn.dropout(tf.matmul(dropped_input, self.W_1) + self.bias_1, 1)
            self.y = tf.tanh(self.h)

            self.h_test = tf.matmul(self.word_representation, self.W_1) + self.bias_1
            self.y_test = tf.tanh(self.h_test)

        with tf.name_scope('training'):
            with tf.name_scope('loss'):
                self.beta = 0.001
                self.mse = tf.contrib.losses.mean_squared_error(labels=self.brain_representation,predictions=self.y)
                self.pair_loss = tf.contrib.losses.mean_pairwise_squared_error(predictions=self.y,labels=self.brain_representation)

                self.loss = self.mse  + self.pair_loss +self.beta * ( tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.bias_1))

                self.mse_test = tf.contrib.losses.mean_squared_error(labels=self.brain_representation, predictions=self.y_test)
                self.pair_loss_test = tf.contrib.losses.mean_pairwise_squared_error(predictions=self.y_test, labels=self.brain_representation)

                self.loss_test = self.mse_test + self.pair_loss_test + self.beta * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.bias_1))

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                self.train_op = self.optimizer.minimize(self.loss)

        with tf.name_scope("evaluation"):
            with tf.name_scope("test_accuracy"):
                p1 = tf.matmul(
                    tf.expand_dims(tf.reduce_sum(tf.square(self.y_test), 1), 1),
                    tf.ones(shape=(1, 2))
                )
                p2 = tf.transpose(tf.matmul(
                    tf.reshape(tf.reduce_sum(tf.square(self.brain_representation), 1), shape=[-1, 1]),
                    tf.ones(shape=(2, 1)),
                    transpose_b=True
                ))

                p = p1 + p2
                pq = 2 * tf.matmul(self.y_test, self.brain_representation, transpose_b=True)
                dists_matrix = tf.sqrt(p - pq)

                min_indices = tf.argmin(dists_matrix, dimension=1)
                correct_max_indices = 1 - tf.minimum(tf.abs(min_indices - np.arange(2)), 1)
                self.accuracy_test = tf.reduce_min(correct_max_indices)



                # Attach summaries to Tensors (for TensorBoard visualization)
                # tf.summary.histogram('W_1', self.W_1)
                # tf.summary.histogram('bias_1', self.bias_1)
                # tf.summary.scalar('loss', self.loss)

                # This op will calculate our summary data when run
                # self.summary_op = tf.summary.merge_all()

    def train(self, x_train, y_train, x_test, y_test):

        # Write the graph
        # self.writer = tf.summary.FileWriter(LRModel.LOGDIR)
        # self.writer.add_graph(self.sess.graph)
        indexes = np.arange(x_train.shape[0])
        batch_size = 58
        number_of_batches = x_train.shape[0] / batch_size
        for step in range(self.training_steps):

            batch_number = 0
            random.shuffle(indexes)
            while batch_number < number_of_batches:
                # Session will run two ops:
                # - summary_op prepares summary data we'll write to disk in a moment
                # - train will use the optimizer to adjust our variables to reduce loss
                start = batch_number * batch_size
                end = np.max([(batch_number + 1) * batch_size,x_train.shape[0]])
                loss,_ = self.sess.run([self.loss,self.train_op],
                              feed_dict={self.word_representation: x_train[indexes[start:end]],
                self.brain_representation: y_train[indexes[start:end]]})

                #print(loss)
                batch_number += 1
                # write the summary data to disk
                # self.writer.add_summary(summary_result, step)

            # Uncomment the following two lines to watch training happen real time.
            if step % 50 == 0:
                print(step, self.sess.run([self.loss], feed_dict={self.word_representation: x_train,
                                                                  self.brain_representation: y_train}))
                # self.test(x_test,y_test)

                # close the writer when we're finished using it
                # self.writer.close()

    def test(self, x_test, y_test):

        y,loss, acc = self.sess.run([self.y,self.loss_test, self.accuracy_test], feed_dict={self.word_representation: x_test,

                                                                                 self.brain_representation: y_test})

        dist_1_1 = scipy.spatial.distance.euclidean(y[0],y_test[0])
        dist_1_2 = scipy.spatial.distance.euclidean(y[0], y_test[1])
        dist_2_1 = scipy.spatial.distance.euclidean(y[1], y_test[0])
        dist_2_2 = scipy.spatial.distance.euclidean(y[1], y_test[1])

        acc2 =  (dist_1_1 + dist_2_2) < (dist_1_2+dist_2_1)

        print("test loss, accuracy: ", loss, acc,acc2)

        return loss, acc, acc2

    @staticmethod
    def make_noisy_data(w=0.1, b=0.3, shape=(100, 2)):
        x = np.random.rand(*shape)
        noise = np.random.normal(scale=0.01, size=shape)
        y = w * x + b + noise
        return x, y

    @staticmethod
    def prepare_data(participant=""):
        brain_activations_1 = genfromtxt('../data/data'+participant+'.csv', delimiter=',')
        brain_activations = brain_activations_1 - np.mean(brain_activations_1,axis=0)
        brain_activations = np.tanh(brain_activations)


        words_1 = []
        with open('../data/words', 'r') as f:
            reader = csv.reader(f)
            words_1 = list(reader)

        words = []
        words.extend([w[0] for w in words_1])
        word_set = list(set(words))
        print("number of words: %d " % len(word_set))


        mean_Activations = []

        words = np.asarray(words)
        for word in word_set:
            indices = np.where(words == word)[0]
            mean_Activations.append(np.mean(brain_activations[indices,:],axis=0))


        words = word_set
        wem = WordEmbeddingLayer()
        wem.load_filtered_embedding("../data/neuro_words")

        embedded_words = wem.embed_words(words)
        word_representations = embedded_words

        #selected = select_stable_voxels(brain_activations_1, word_set, words, number_of_trials=6,
        #                                size_of_selection=500)

        selected = np.load("general_selected_500.npy")

        return words, np.asarray(word_representations), np.asarray(mean_Activations)[:, selected]


if __name__ == '__main__':
    words, x_all, y_all = LRModel.prepare_data()
    word_set = list(set(words))
    accuracies = []
    print(x_all.shape[0])
    print(len(list(itertools.combinations(range(len(word_set)), 2))))
    words = np.asarray(words)
    for test_word_indices in itertools.combinations(range(len(word_set)), 2):
        test_indices_1 = np.where(words == word_set[test_word_indices[0]])[0]
        test_indices_2 = np.where(words == word_set[test_word_indices[1]])[0]
        mask = np.ones(x_all.shape[0])
        test_indices = np.append(test_indices_1, test_indices_2)
        mask[test_indices] = 0
        train_indices = list(itertools.compress(range(x_all.shape[0]), mask))
        # print('Train: %s | test: %s' % (train_indices, test_indices))
        x_train, y_train = x_all[train_indices], y_all[train_indices]
        x_test, y_test = np.asarray([x_all[test_indices_1[0]],x_all[test_indices_2[0]]]), \
                         np.asarray([np.mean(y_all[test_indices_1],axis=0),
                                   np.mean(y_all[test_indices_2], axis=0)])

        print("x_train shape: " + str(x_train.shape))
        print("y_train shape: " + str(y_train.shape))
        print("x_test shape: " + str(x_test.shape))
        print("y_test shape: " + str(y_test.shape))
        lrm = LRModel(x_train.shape[1], y_train.shape[1],hidden_dim=y_train.shape[1])
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        loss, acc, acc2 = lrm.test(x_test=x_test, y_test=y_test)
        lrm.sess.close()
        accuracies.append(acc2)

    print("accuracy: ", np.mean(accuracies))
