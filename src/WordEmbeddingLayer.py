import numpy as np
import funcy
from funcy import project
import pickle


import sys
sys.path.append('../../')
from Util.util.math.MathUtil import *


class WordEmbeddingLayer(object):

    def __init__(self):
        self.word2vec = {}
        self.vec2word = {}


    def load_embeddings_from_glove_file(self,filename,filter):
        self.word2vec = {}
        self.vec2word = {}

        with open(filename,'r') as gfile:
            for line in gfile:
                parts = line.split()
                i = 0
                word = ''
                while not MathUtil.is_float(parts[i]):
                    word += " "+parts[i]
                    i += 1
                word = word.strip().lower()
                if (len(filter) == 0) or (word in filter):
                    vector = [float(p) for p in parts[i:]]
                    vector = np.asarray(vector)
                    self.word2vec[word] = vector
                    self.vec2word[vector.tostring()] = word
        print(self.word2vec.keys())
        print(list(self.word2vec.keys())[0])
        self.word2vec['UNK'] = np.zeros(self.word2vec[list(self.word2vec.keys())[0]].shape)
        self.vec2word[self.word2vec['UNK'].tostring()] = "UNK"



    def load_embeddings_from_word2vex(self,filename,filter):
        self.word2vec = {}
        self.vec2word = {}

        with open(filename,'r') as gfile:
            for line in gfile:
                parts = line.split()
                i = 0
                word = ''
                while not MathUtil.is_float(parts[i]):
                    word += " "+parts[i]
                    i += 1
                word = word.strip().lower()
                if (len(filter) == 0) or (word in filter):
                    vector = [float(p) for p in parts[i:]]
                    vector = np.asarray(vector)
                    self.word2vec[word] = vector
                    self.vec2word[vector.tostring()] = word
        print(self.word2vec.keys())
        print(list(self.word2vec.keys())[0])
        self.word2vec['UNK'] = np.zeros(self.word2vec[list(self.word2vec.keys())[0]].shape)
        self.vec2word[self.word2vec['UNK'].tostring()] = "UNK"

    def load_embeddings_from_fasttext_file(self,filename,filter):
        self.word2vec = {}
        self.vec2word = {}

        with open(filename,'r') as gfile:
            firstLine = True
            for line in gfile:
                if firstLine:
                    firstLine = False
                    continue
                parts = line.split()
                i = 0
                word = ''
                while not MathUtil.is_float(parts[i]):
                    word += " "+parts[i]
                    i += 1
                word = word.strip().lower()
                if (len(filter) == 0) or (word in filter):
                    vector = [float(p) for p in parts[i:]]
                    vector = np.asarray(vector)
                    self.word2vec[word] = vector
                    self.vec2word[vector.tostring()] = word
        print(self.word2vec.keys())
        print(list(self.word2vec.keys())[0])
        self.word2vec['UNK'] = np.zeros(self.word2vec[list(self.word2vec.keys())[0]].shape)
        self.vec2word[self.word2vec['UNK'].tostring()] = "UNK"


    def load_embeddings_from_lexvec_file(self,filename,filter):
        self.word2vec = {}
        self.vec2word = {}

        with open(filename,'r') as gfile:
            firstLine = True
            for line in gfile:
                if firstLine:
                    firstLine = False
                    continue
                parts = line.split()
                i = 0
                word = ''
                while not MathUtil.is_float(parts[i]):
                    word += " "+parts[i]
                    i += 1
                word = word.strip().lower()
                if (len(filter) == 0) or (word in filter):
                    vector = [float(p) for p in parts[i:]]
                    vector = np.asarray(vector)
                    self.word2vec[word] = vector
                    self.vec2word[vector.tostring()] = word
        print(self.word2vec.keys())
        print(list(self.word2vec.keys())[0])
        self.word2vec['UNK'] = np.zeros(self.word2vec[list(self.word2vec.keys())[0]].shape)
        self.vec2word[self.word2vec['UNK'].tostring()] = "UNK"

    def save_embedding(self,filename):
        with open(filename+"_word2vec.pkl","wb") as f:
            pickle.dump(self.word2vec,f)

        with open(filename+"_vec2word.pkl","wb") as f:
            pickle.dump(self.vec2word,f)


    def load_filtered_embedding(self,filename):
        with open(filename+"_word2vec.pkl","rb") as f:
            self.word2vec = pickle.load(f)

        with open(filename+"_vec2word.pkl","rb") as f:
            self.vec2word = pickle.load(f)

    def get_vector(self,word):
        if word in self.word2vec:
            return self.word2vec[word]
        else:
            return self.word2vec['UNK']

    def get_word(self,vector):
        return self.vec2word[vector.tostring()]

    def filter_unseen_vocab(self,vocab):
        self.word2vec = project(self.word2vec, vocab)
        self.vec2word = project(self.vec2word, [self.word2vec[word].tostring() for word in vocab])

    @staticmethod
    def load_embedded_data(path, name, representation):
        embedded, labels = [], []

        with open(path+"embedded_"+name+"_"+representation+".pkl", "rb") as f:
            embedded = pickle.load(f)
        with open(path+"labels_"+name+".pkl", "rb") as f:
            labels = pickle.load(f)

        return np.asarray(embedded), np.asarray(labels)

    def embed_and_save(self,sentences,labels,path,name,representation):
        embedded = self.embed(sentences)

        with open(path+"embedded_"+name+"_"+representation+".pkl","wb") as f:
            pickle.dump(embedded,f)
        with open(path+"labels_"+name+".pkl", "wb") as f:
            pickle.dump(labels,f)

    def embed(self,sentences):
        embedded_sentences = []
        for sentence in sentences:
            embedded_sentences.append([self.get_vector(vocab) for vocab in sentence])

        return embedded_sentences

    def embed_words(self,words):
        embedded_words = []
        for word in words:
            embedded_words.append(self.get_vector(word) )

        return embedded_words












if __name__ == '__main__':
    import csv

    wem = WordEmbeddingLayer()

    words = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words = list(reader)

    """
    print(words[0][0])
    wem.load_embeddings_from_glove_file(filename="../data/glove.840B.300d.txt",filter = [word[0] for word in words])

    wem.save_embedding("../data/neuro_words")
    """

    """
    wem.load_embeddings_from_fasttext_file("../data/wiki.en/wiki.en.vec",filter = [word[0] for word in words])
    wem.save_embedding("../data/neuro_words_fasttext")
    """

    """
    wem.load_embeddings_from_fasttext_file("../data/lexvec.enwiki+newscrawl.300d.W.pos.vectors", filter=[word[0] for word in words])
    wem.save_embedding("../data/neuro_words_lexvec")"""


    import gzip

    word2vec = {}
    vec2word = {}
    """
    filter = [word[0] for word in words]
    with gzip.open("../data/binary-vectors.txt.gz", 'rt') as gfile:
        for line in gfile:
            parts = line.split()
            i = 0
            word = ''
            while not MathUtil.is_float(parts[i]):
                word += " " + parts[i]
                i += 1
            word = word.strip().lower()
            if (len(filter) == 0) or (word in filter):
                vector = [float(p) for p in parts[i:]]
                vector = np.asarray(vector)
                word2vec[word] = vector
                vec2word[vector.tostring()] = word
    print(word2vec.keys())
    print(list(word2vec.keys())[0])
    word2vec['UNK'] = np.zeros(word2vec[list(word2vec.keys())[0]].shape)
    vec2word[word2vec['UNK'].tostring()] = "UNK"

    
    with open("../data/neuro_words_nd" + "_word2vec.pkl", "wb") as f:
        pickle.dump(word2vec, f)

    with open("../data/neuro_words_nd" + "_vec2word.pkl", "wb") as f:
        pickle.dump(vec2word, f)
    """

    with open("../data/neuro_words_nd" + "_word2vec.pkl", "rb") as f:
        word2vec = pickle.load(f)


    selected_feature_index = np.where(np.sum(list(word2vec.values()),axis=1) > 0)

    print(selected_feature_index)

    for word in word2vec:
        word2vec[word] = np.asarray(word2vec[word])[selected_feature_index]
        vec2word[word2vec[word].tostring()] = word

    with open("../data/neuro_words_cnd" + "_word2vec.pkl", "wb") as f:
        pickle.dump(word2vec, f)

    with open("../data/neuro_words_cnd" + "_vec2word.pkl", "wb") as f:
        pickle.dump(vec2word, f)



