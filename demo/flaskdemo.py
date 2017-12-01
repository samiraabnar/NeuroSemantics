import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, jsonify, render_template, request
from flask_bootstrap import Bootstrap


import sys
from scipy import *
from scipy.spatial import *

sys.path.append("../src")

import tf_LRModel_GPU
from WordEmbeddingLayer import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pylab import *

import tf_LRModel_reversed

class ExpSetup(object):
    def __init__(self,
                 learning_rate,
                 batch_size,
                 number_of_epochs
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs


    def __str__(self):
        return "learning_rate: "+str(self.learning_rate)\
               + "batch_size: "+str(self.batch_size)\
            + "number_of_epochs: "+str(self.number_of_epochs)






expSetup = ExpSetup(learning_rate=0.001, batch_size=29, number_of_epochs=700)

fMRI_data_path = "../data/"
fMRI_data_filename = "data_"
fMRI_data_postfix = ".csv"
subject_id = str(1)

words, x_all, y_all = tf_LRModel_GPU.LRModel.prepare_data(
        fMRI_file=fMRI_data_path + fMRI_data_filename + subject_id + fMRI_data_postfix,
        subject=subject_id, type="glove", select=False)

words_r, x_all_r, y_all_r = tf_LRModel_reversed.LRModel.prepare_data(
        fMRI_file=fMRI_data_path + fMRI_data_filename + subject_id + fMRI_data_postfix,
        subject=subject_id, type="glove", select=False)


x_train, y_train = x_all, y_all

lrm = tf_LRModel_GPU.LRModel(x_train.shape[1], y_train.shape[1], learning_rate=expSetup.learning_rate,
                  hidden_dim=y_train.shape[1], training_steps=expSetup.number_of_epochs,
                  batch_size=expSetup.batch_size)
lrm.load_model("../glove_all.model")

expSetup_reversed = ExpSetup(learning_rate=0.01, batch_size=29, number_of_epochs=2000)


lrm_reversed = tf_LRModel_reversed.LRModel(y_train.shape[1], x_train.shape[1], learning_rate= expSetup_reversed.learning_rate,hidden_dim=x_train.shape[1],training_steps=expSetup_reversed.number_of_epochs, batch_size=expSetup_reversed.batch_size)
lrm_reversed.load_model("../glove_reversed_all_3.model")

wem = WordEmbeddingLayer()
wem.load_embeddings_from_glove_file(filename="../data/glove.6B/glove.6B.300d.txt", filter=[], dim=300)
#wem.load_filtered_embedding("../data/glove_all_6B_300d")  # neuro_words

all_words = [w for w in wem.word2vec.keys()]
all_embedded_words = wem.embed_words(all_words[:len(all_words)])
all_embedded_words = np.asarray(all_embedded_words)
print(all_embedded_words.shape)
word_tree = cKDTree(all_embedded_words)
word_tree_cheat = cKDTree(x_all)



coords = []
with open('../data/coords', 'r') as f:
    reader = csv.reader(f)
    coords = list(reader)
coords = np.asarray(coords)
coord = np.asarray(coords, dtype=int)

the_words = [None, None]
brain_acts = [None, None]
fake_acts = [None, None]

app = Flask(__name__,static_url_path='/static')
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify/', methods=['GET'])
def classify():
    predicted = ["", "", "", ""]
    classified_as_word_1 = lrm_reversed.get_prediction(fake_acts[0])
    dd, a = word_tree.query(classified_as_word_1)
    print(a)
    print(all_words[a[0][0]])

    classified_as_word_2 = lrm_reversed.get_prediction(fake_acts[1])
    dd, b = word_tree.query(classified_as_word_2)
    print(b)
    print(all_words[b[0][0]])

    if (brain_acts[0] is not None):
        classified_as_word_3 = lrm_reversed.get_prediction(brain_acts[0])
        dd, c = word_tree_cheat.query(classified_as_word_3)
        predicted[2] = "<b>"+words[c[0][0]]+"</b>"

    if (brain_acts[1] is not None):
        classified_as_word_4 = lrm_reversed.get_prediction(brain_acts[1])
        dd, d = word_tree_cheat.query(classified_as_word_4)
        predicted[3] = "<b>"+words[d[0][0]]+"</b>"

    predicted[0] = "<b>"+all_words[a[0][0]]+"</b>"
    predicted[1] = "<b>"+all_words[b[0][0]]+"</b>"
    return jsonify(predicted)


@app.route('/reset/', methods=['GET'])
def reset():
    plot_urls = reset_plots()

    return jsonify(['<img  src="data:image/png;base64,{}">'.format(plot_urls[0]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[1]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[2]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[3])])


@app.route('/echo/', methods=['GET'])
def echo():
    word_1 = request.args.get('word_1')
    word_2 = request.args.get('word_2')



    print(word_1)
    print(word_2)
    #    return jsonify(ret_data)



    input_words = np.asarray([word_1,word_2], dtype=object)
    print(input_words.shape)
    print(input_words)
    embedded_words = wem.embed_words(input_words)

    plot_urls = plot_func(embedded_words, input_words)

    #return '<img src="data:image/png;base64,{}">'.format(plot_url)
    return jsonify(['<img  src="data:image/png;base64,{}">'.format(plot_urls[0]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[1]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[2]),
            '<img  src="data:image/png;base64,{}">'.format(plot_urls[3])])

def plot_func(embedded_words, input_words):
    i = 1;
    angel = 0
    zangle = 90
    img = [io.BytesIO(),io.BytesIO(),io.BytesIO(),io.BytesIO()]
    for e_word, word in zip(embedded_words, input_words):
        predicted_brain_activation = lrm.get_prediction([e_word])
        print(word)
        print(predicted_brain_activation[0].shape)

        fig = plt.figure()
        ax = fig.add_subplot("111", projection='3d')
        ax.scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                          np.asarray(coords, dtype=int)[:, [2]],
                          s=2, c=predicted_brain_activation[0], alpha=0.8)
        ax.view_init(zangle, angel)
        ax.set_axis_off()
        #ax.set_title(word)
        plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1,
                        wspace=0, hspace=0)
        fig.tight_layout(pad=0)
        plt.savefig(img[i-1], format='png',transparent=True,bbox_inches = 'tight',
    pad_inches = 0)
        img[i-1].seek(0)

        the_words[i - 1] = word
        fake_acts[i - 1] = predicted_brain_activation[0]

        fig = plt.figure()
        ax = fig.add_subplot("111", projection='3d')
        if (word in words):
            real_activation = y_all_r[np.where(np.asarray(words_r) == word)]
            print("real_Activation_shape: ",real_activation.shape)
            brain_acts[i - 1] = real_activation
            print(word)


            ax.scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                           np.asarray(coords, dtype=int)[:, [2]],
                           s=2, c=real_activation, alpha=0.8)

            img[i - 1].seek(0)
        else:
            brain_acts[i - 1] = None
            ax.scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                  np.asarray(coords, dtype=int)[:, [2]],
                  s=2, c='grey', alpha=0.8)

        ax.view_init(zangle, angel)
        ax.set_axis_off()
        #ax.set_title(word)
        plt.subplots_adjust(left=0, bottom=0, right=0.1, top=0.1,
                            wspace=0, hspace=0)
        fig.tight_layout(pad=0)
        plt.savefig(img[i - 1 + 2], format='png',transparent=True,bbox_inches = 'tight',
    pad_inches = 0)
        img[i - 1 + 2].seek(0)

        i += 1

    plot_urls = [base64.b64encode(img[0].getvalue()).decode(),
                 base64.b64encode(img[1].getvalue()).decode(),
                 base64.b64encode(img[2].getvalue()).decode(),
                 base64.b64encode(img[3].getvalue()).decode()]
    return plot_urls

def reset_plots():
    angel = 0
    zangle = 90
    img = [io.BytesIO(),io.BytesIO(),io.BytesIO(),io.BytesIO()]
    for i in np.arange(1):
        fig = plt.figure()
        ax = fig.add_subplot("111", projection='3d')
        ax.scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                          np.asarray(coords, dtype=int)[:, [2]],
                          s=2, c='grey', alpha=0.8)
        ax.view_init(zangle, angel)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0)
        plt.savefig(img[i], format='png',transparent=True,bbox_inches = 'tight',
    pad_inches = 0)
        img[i].seek(0)



    plot_urls = [base64.b64encode(img[0].getvalue()).decode(),
                 base64.b64encode(img[0].getvalue()).decode(),
                 base64.b64encode(img[0].getvalue()).decode(),
                 base64.b64encode(img[0].getvalue()).decode()]
    return plot_urls


if __name__ == '__main__':
    app.run(port=8080, debug=True)
