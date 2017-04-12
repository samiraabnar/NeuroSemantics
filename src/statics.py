from numpy import genfromtxt
import csv
import numpy as np
from matplotlib import pyplot as plt



import sys
sys.path.append('../../')

from NeuroSemantics.src.WordEmbeddingLayer import *
from NeuroSemantics.src.functions import *


brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')

brain_activations_2 = genfromtxt('../data/data_2.csv', delimiter=',')

brain_activations_3 = genfromtxt('../data/data_3.csv', delimiter=',')

brain_activations_4 = genfromtxt('../data/data_4.csv', delimiter=',')

brain_activations_5 = genfromtxt('../data/data_5.csv', delimiter=',')

brain_activations_6 = genfromtxt('../data/data_6.csv', delimiter=',')

brain_activations_7 = genfromtxt('../data/data_7.csv', delimiter=',')

brain_activations_8 = genfromtxt('../data/data_8.csv', delimiter=',')

brain_activations_9 = genfromtxt('../data/data_9.csv', delimiter=',')





number_of_voxels = len(brain_activations_1[0])
words = []
with open('../data/words', 'r') as f:
    reader = csv.reader(f)
    words = list(reader)

words = [w[0] for w in words]
word_set = list(set(words))

brain_activations_all = []
words_all = []

brain_activations_all.extend(brain_activations_1)
brain_activations_all.extend(brain_activations_2)
brain_activations_all.extend(brain_activations_3)
brain_activations_all.extend(brain_activations_4)
brain_activations_all.extend(brain_activations_5)
brain_activations_all.extend(brain_activations_6)
brain_activations_all.extend(brain_activations_7)
brain_activations_all.extend(brain_activations_8)
brain_activations_all.extend(brain_activations_9)

words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)
words_all.extend(words)


brain_activations_1 = brain_activations_all[:,1900]
selected = select_stable_voxels(brain_activations_1, word_set,words_all, 6*9,100)
softmax_activations = sigmoid(brain_activations_1[:,selected])

word_activations_dic = {}
real_activatoins_1 = {}
"""real_activatoins_2 = {}
real_activatoins_3 = {}
real_activatoins_4 = {}
real_activatoins_5 = {}
real_activatoins_6 = {}
real_activatoins_7 = {}
real_activatoins_8 = {}
real_activatoins_9 = {}
"""
for word in word_set:
    word_activations_dic[word] = []
"""    real_activatoins_1[word] = []
    real_activatoins_2[word] = []
    real_activatoins_3[word] = []
    real_activatoins_4[word] = []
    real_activatoins_5[word] = []
    real_activatoins_6[word] = []
    real_activatoins_7[word] = []
    real_activatoins_8[word] = []
    real_activatoins_9[word] = []"""

for i in np.arange(len(words_all)):
    word_activations_dic[words[i]].append(softmax_activations[i])
    """real_activatoins_1[words[i]].append(brain_activations_1)
    real_activatoins_2[words[i]].append(brain_activations_2)
    real_activatoins_3[words[i]].append(brain_activations_3)
    real_activatoins_4[words[i]].append(brain_activations_4)
    real_activatoins_5[words[i]].append(brain_activations_5)
    real_activatoins_6[words[i]].append(brain_activations_6)
    real_activatoins_7[words[i]].append(brain_activations_7)
    real_activatoins_8[words[i]].append(brain_activations_8)
    real_activatoins_9[words[i]].append(brain_activations_9)"""


"""
x = []
y_1 = []
y_2 = []
y_3 = []
y_4 = []
y_5 = []
y_6 = []
y_7 = []
y_8 = []
y_9 = []
for key_word in word_activations_dic.keys():
    divs_1 = []
    divs_2 = []
    divs_3 = []
    divs_4 = []
    divs_5 = []
    divs_6 = []
    divs_7 = []
    divs_8 = []
    divs_9 = []

    for i in np.arange(len(word_activations_dic[key_word])):
        for j in np.arange(i,len(word_activations_dic[key_word])):
            d1_1 = real_activatoins_1[key_word][i]
            d2_1 = real_activatoins_1[key_word][j]

            d1_2 = real_activatoins_2[key_word][i]
            d2_2 = real_activatoins_2[key_word][j]

            d1_3 = real_activatoins_3[key_word][i]
            d2_3 = real_activatoins_3[key_word][j]

            d1_4 = real_activatoins_4[key_word][i]
            d2_4 = real_activatoins_4[key_word][j]

            d1_5 = real_activatoins_5[key_word][i]
            d2_5 = real_activatoins_5[key_word][j]

            d1_6 = real_activatoins_6[key_word][i]
            d2_6 = real_activatoins_6[key_word][j]

            d1_7 = real_activatoins_7[key_word][i]
            d2_7 = real_activatoins_7[key_word][j]

            d1_8 = real_activatoins_8[key_word][i]
            d2_8 = real_activatoins_8[key_word][j]

            d1_9 = real_activatoins_9[key_word][i]
            d2_9 = real_activatoins_9[key_word][j]


            divs_1.append(jsdiv(d1_1,d2_1))
            divs_2.append(jsdiv(d1_2,d2_2))
            divs_3.append(jsdiv(d1_3,d2_3))
            divs_4.append(jsdiv(d1_4,d2_4))
            divs_5.append(jsdiv(d1_5,d2_5))
            divs_6.append(jsdiv(d1_6,d2_6))
            divs_7.append(jsdiv(d1_7,d2_7))
            divs_8.append(jsdiv(d1_8,d2_8))
            divs_9.append(jsdiv(d1_9,d2_9))

    x.append(word_set.index(key_word))
    y_1.append(np.mean(divs_1))
    y_2.append(np.mean(divs_2))
    y_3.append(np.mean(divs_3))
    y_4.append(np.mean(divs_4))
    y_5.append(np.mean(divs_5))
    y_6.append(np.mean(divs_6))
    y_7.append(np.mean(divs_7))
    y_8.append(np.mean(divs_8))
    y_9.append(np.mean(divs_9))

plt.title("Js between brain activations for each word for P"+str(1))
plt.bar(np.arange(len(x)),y_1)
plt.bar(np.arange(len(x)),y_2)
plt.bar(np.arange(len(x)),y_3)
plt.bar(np.arange(len(x)),y_4)
plt.bar(np.arange(len(x)),y_5)
plt.bar(np.arange(len(x)),y_6)
plt.bar(np.arange(len(x)),y_7)
plt.bar(np.arange(len(x)),y_8)
plt.bar(np.arange(len(x)),y_9)

plt.xlabel(x)
plt.show()
"""

for key_word in word_activations_dic.keys():
    averaged = np.mean(word_activations_dic[key_word],axis=0)
    sort_index = np.argsort(averaged)
    """x = np.arange(number_of_voxels)

    with plt.style.context('fivethirtyeight'):
        for activation in word_activations_dic[key_word]:
            plt.plot(x, activation)

        plt.show()
    """
    print ("plotting")
    plt.title(key_word)
    im = plt.imshow(np.asarray(word_activations_dic[key_word])[:,sort_index], interpolation='none',aspect='auto',cmap='gray')
    plt.colorbar(im)
    plt.show()

