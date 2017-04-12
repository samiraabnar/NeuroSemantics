from numpy import genfromtxt
import csv
import sys
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
import numpy as np
from scipy.stats.stats import pearsonr
from scipy import *
from scipy.spatial.distance import *
from sklearn import preprocessing, linear_model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

sys.path.append('../../')

from NeuroSemantics.src.WordEmbeddingLayer import *
from NeuroSemantics.src.functions import *


brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')
brain_activations = brain_activations_1
#brain_activations = sigmoid(brain_activations) #preprocessing.normalize(brain_activations, norm='l1')

words_1 = []
with open('../data/words', 'r') as f:
    reader = csv.reader(f)
    words_1 = list(reader)

words = []
words.extend([w[0] for w in words_1])
word_set = list(set(words))
print(len(word_set))



#wem = WordEmbeddingLayer()
#wem.load_filtered_embedding("../data/neuro_words")

#embedded_words = wem.embed(words)
#embedded_words_2 = np.asarray([e[0] for e in embedded_words])

dic,word_representations = get_word_representation("F25",words)
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
            activations.append(brain_activations[k])
            word_vectors.append(word_representations[k])
            if words[k] not in tmp_words:
                tmp_words.append(words[k])
            all_tmp_words.append(words[k])

    all_features.append(word_vectors)
    all_words.append(tmp_words)

    selected = select_stable_voxels(activations, tmp_words,all_tmp_words, 6)
    all_activations.append(np.asarray(activations)[:,selected])
    all_selected.append(selected)

    model = Sequential()
    model.add(Dense(input_dim=number_of_features, output_dim=200, activation='linear'))
    model.add(Dense(input_dim=200, output_dim=all_activations[-1].shape[1],activation='linear'))



    model.compile("Nadam","mse", metrics=['mse'])
    model.fit(np.asarray(all_features[-1]), scale_linear_bycolumn(all_activations[-1]),batch_size=20,nb_epoch=20)
    #model.save("model_data_1_" +str((i, j)))
    """model = linear_model.LinearRegression()
    model.fit(np.asarray(all_features[-1]), all_activations[-1])"""
    predicted_1 = model.predict(np.asarray([all_features[-1][i]]))
    predicted_2 = model.predict(np.asarray([all_features[-1][j]]))
    result_1 = avereaged_match_prediction(predicted_1,predicted_2, the_pairs[-1],scale_linear_bycolumn(brain_activations[:,selected ]), ,words,word_set)
    results.append(result_1)
    print(result_1)
    print("Accuracy: " + str(sum(results) / len(results)))



"""np.save("sig_norm/all_activations_simple",all_activations)
np.save("sig_norm/all_features_simple",all_features)
np.save("sig_norm/the_pairs_simple",the_pairs)
np.save("sig_norm/all_words_simple",all_words)
np.save("sig_norm/all_selected_simple",all_selected)
f.close()
"""


result = []

"""all_activations = np.load("all_data.bin")
all_features = np.load("all_features")
the_pairs = np.load("the_pairs")
all_words = np.load("all_words")

"""
"""for k in np.arange(441,len(the_pairs)):
    model = Sequential()
    model.add(Dense(input_dim=300, output_dim=1000, W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.0001)))
    model.add(Dense(input_dim=1000, output_dim=brain_activations[0].shape[0], W_regularizer=l2(0.0001),
                    activity_regularizer=activity_l2(0.0001)))
    model.compile("rmsprop", "mse", metrics=['accuracy'])
    model.fit(np.asarray(all_features[k]),all_activations[k],nb_epoch=10)
    model.save("model_data_1_"+str(the_pairs[k]))
"""


