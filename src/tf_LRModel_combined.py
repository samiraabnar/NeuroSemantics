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

    def __init__(self, input_dim, output_dim, learning_rate=0.1, hidden_dim=50, training_steps=500, batch_size=29):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.batch_size = batch_size

        tf.reset_default_graph()
        self.sess = tf.Session()

        self.define_model()
        self.sess.run(tf.global_variables_initializer())

    def define_model(self):
        with tf.name_scope('input'):
            self.word_representation = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32, name='x-input')
            self.brain_representation = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32, name='y-input')

        with tf.name_scope('model'):
            self.W_1 = tf.Variable(tf.random_uniform([self.input_dim, self.hidden_dim],minval=-0.1,maxval=0.1),
                                   name='W_1')
            self.bias_1 = tf.Variable(tf.zeros([self.hidden_dim]), name='bias_1')
            self.W_0 = tf.Variable(tf.random_uniform([self.input_dim, self.input_dim], minval=-.1, maxval=.1),
                                   name='W_0')
            #self.bias_2 = tf.Variable(tf.zeros([self.output_dim]), name='bias_2')
            dropped_input = tf.matmul(self.W_0,self.word_representation)

            # This is the same as y = tf.add(tf.mul(m, x_placeholder), b), but looks nicer
            self.h = tf.matmul(dropped_input, tf.nn.dropout(self.W_1,0.6)) + self.bias_1
            self.y = tf.tanh(self.h)

            self.h_test = tf.matmul(self.word_representation, self.W_1) + self.bias_1
            self.y_test = tf.tanh(self.h_test)

        with tf.name_scope('training'):
            with tf.name_scope('loss'):
                self.beta = 0.001
                self.mse = tf.losses.huber_loss(labels=self.brain_representation,predictions=self.y)
                self.pair_loss = tf.losses.mean_pairwise_squared_error(predictions=self.y,labels=self.brain_representation,weights=0.5)

                self.loss = self.mse  + self.pair_loss +self.beta * ( tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.bias_1))

                self.mse_test = tf.losses.mean_squared_error(labels=self.brain_representation, predictions=self.y_test)
                self.pair_loss_test = tf.losses.mean_pairwise_squared_error(predictions=self.y_test, labels=self.brain_representation)

                self.loss_test = self.mse_test + self.pair_loss_test + self.beta * (tf.nn.l2_loss(self.W_1) + tf.nn.l2_loss(self.bias_1))

            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer() #tf.train.MomentumOptimizer(self.learning_rate, momentum=0.95)
                self.train_op = self.optimizer.minimize(self.loss)

        """with tf.name_scope("evaluation"):
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
            """


                # Attach summaries to Tensors (for TensorBoard visualization)
                # tf.summary.histogram('W_1', self.W_1)
                # tf.summary.histogram('bias_1', self.bias_1)
                # tf.summary.scalar('loss', self.loss)

                # This op will calculate our summary data when run
                # self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    def train(self, x_train, y_train, x_test, y_test):

        # Write the graph
        # self.writer = tf.summary.FileWriter(LRModel.LOGDIR)
        # self.writer.add_graph(self.sess.graph)
        indexes = np.arange(x_train.shape[0])
        batch_size = self.batch_size
        number_of_batches = x_train.shape[0] / batch_size
        for step in range(self.training_steps):

            batch_number = 0
            random.shuffle(indexes)
            while batch_number <= number_of_batches:
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
            if step % 100 == 0:
                print(step, self.sess.run([self.loss], feed_dict={self.word_representation: x_train,
                                                                  self.brain_representation: y_train}))
                # self.test(x_test,y_test)

                # close the writer when we're finished using it
                # self.writer.close()

    def test(self, x_test, y_test):

        y,loss = self.sess.run([self.y,self.loss_test], feed_dict={self.word_representation: x_test,

                                                                                 self.brain_representation: y_test})

        dist_1_1 = scipy.spatial.distance.cosine(y[0],y_test[0])
        dist_1_2 = scipy.spatial.distance.cosine(y[0], y_test[1])
        dist_2_1 = scipy.spatial.distance.cosine(y[1], y_test[0])
        dist_2_2 = scipy.spatial.distance.cosine(y[1], y_test[1])

        acc2 =  (dist_1_1 + dist_2_2) < (dist_1_2+dist_2_1)

        print("test loss: %f , test accuracy: %f" % (loss,acc2))

        return loss, acc2

    @staticmethod
    def make_noisy_data(w=0.1, b=0.3, shape=(100, 2)):
        x = np.random.rand(*shape)
        noise = np.random.normal(scale=0.01, size=shape)
        y = w * x + b + noise
        return x, y

    @staticmethod
    def prepare_data(fMRI_file,subject,type="glove",mode="none"):
        brain_activations_1 = genfromtxt(fMRI_file, delimiter=',')
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



        selected_file_name = "general_selected_500_"+subject+".npy"

        if not os.path.isfile(selected_file_name) :
            selected = select_stable_voxels(brain_activations_1, word_set, words, number_of_trials=6,
                                            size_of_selection=500)
            np.save(selected_file_name,selected)

        selected = np.load(selected_file_name)

        mean_Activations = []


        if mode =='limited':
            with open('../data/experimental_wordList.csv', 'r') as f:
                word_set = [w[0] for w in list(csv.reader(f))]

        words = np.asarray(words)
        for word in word_set:
            indices = np.where(words == word)[0]
            mean_Activations.append(np.mean(brain_activations[indices, :], axis=0))

        words = word_set

        if type == 'glove':
            wem = WordEmbeddingLayer()
            wem.load_filtered_embedding("../data/neuro_words_glove_6B_300d") #neuro_words
            embedded_words = wem.embed_words(words)
        elif type == 'word2vec':
            wem = WordEmbeddingLayer()
            wem.load_filtered_embedding("../data/neuro_words_word2vec")
            embedded_words = wem.embed_words(words)
        elif type == 'fasttext':
            wem = WordEmbeddingLayer()
            wem.load_filtered_embedding("../data/neuro_words_fasttext")
            embedded_words = wem.embed_words(words)
        elif type == 'lexvec':
            wem = WordEmbeddingLayer()
            wem.load_filtered_embedding("../data/neuro_words_lexvec")
            embedded_words = wem.embed_words(words)
        elif type == 'experimental':
            embedding_dic, embedded_words = get_word_representation(type='experimental',words=word_set)
        elif type == 'deps':
            embedding_dic , embedded_words = get_word_representation(type='deps',words=word_set)
        elif type == 'F25':
            embedding_dic, embedded_words = get_word_representation(type='F25', words=word_set)
        elif type == 'non-distributional':
            wem = WordEmbeddingLayer()
            wem.load_filtered_embedding("../data/neuro_words_cnd")
            embedded_words = wem.embed_words(words)
        elif type == 'deps-exp':
            embedding_dic_deps , embedded_words_deps = get_word_representation(type='deps',words=word_set)
            embedding_dic_exp, embedded_words_exp = get_word_representation(type='experimental', words=word_set)
            embedded_words = np.concatenate((embedded_words_deps,embedded_words_exp),axis=1)
        elif type == 'deps-F25':
            embedding_dic_deps , embedded_words_deps = get_word_representation(type='deps',words=word_set)
            embedding_dic_exp, embedded_words_exp = get_word_representation(type='F25', words=word_set)
            embedded_words = np.concatenate((embedded_words_deps,embedded_words_exp),axis=1)


        word_representations = np.asarray(embedded_words)

        return words, np.asarray(word_representations), np.asarray(mean_Activations)[:, selected]


    def save_model(self,name):
        save_path = self.saver.save(self.sess, name)
        print("Model saved in file: %s" % save_path)

    def load_model(self,name):
        self.saver.restore(self.sess, name)
        print("Model restored.")


