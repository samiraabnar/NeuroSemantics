import argparse
from tf_LRModel_reversed import *
import sys
from scipy import *
from scipy.spatial import *

class ExpSetup(object):
    def __init__(self,
                 learning_rate,
                 batch_size,
                 number_of_epochs,
                 mode = 'train'
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.mode = mode

    def __str__(self):
        return "learning_rate: "+str(self.learning_rate)\
               + "batch_size: "+str(self.batch_size)\
            + "number_of_epochs: "+str(self.number_of_epochs)



if __name__ == '__main__':
    wem = WordEmbeddingLayer()
    wem.load_embeddings_from_glove_file(filename="../data/glove.6B/glove.6B.300d.txt", filter=[], dim=300)
    # wem.load_filtered_embedding("../data/glove_all_6B_300d")  # neuro_words

    all_words = [w for w in wem.word2vec.keys()]
    all_embedded_words = wem.embed_words(all_words[:len(all_words)])
    all_embedded_words = np.asarray(all_embedded_words)
    print(all_embedded_words.shape)
    word_tree = cKDTree(all_embedded_words)


    expSetup = ExpSetup(learning_rate=0.01,batch_size=29,number_of_epochs=7000)

    fMRI_data_path = "../data/"
    fMRI_data_filename = "data_"
    fMRI_data_postfix = ".csv"
    subject_id = str(1)

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Single Layer Feed Forward Network for Brain Activation Prediction Task')
    parser.add_argument('--subject','-s', type=str, nargs='?',
                        help='An optional integer specifying the subject id',default="1")

    args = parser.parse_args()
    print("subject id %s" % args.subject)

    words, y_all, x_all = LRModel.prepare_data(
        fMRI_file=fMRI_data_path + fMRI_data_filename + args.subject + fMRI_data_postfix,
        subject=args.subject, type="glove", select=False)
    x_train, y_train = x_all, y_all

    load = True

    if load == False:
        word_set = list(set(words))
        print(x_all.shape[0], x_all.shape[1])
        words = np.asarray(words)
        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train)
        lrm.save_model("../glove_reversed_all_3.model")
        lrm.sess.close()

        print(str(expSetup))
    else:
        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.load_model("../glove_reversed_all_3.model")

        print(words[0])
        classified_as_word = lrm.get_prediction([x_train[0]])
        dd, b = word_tree.query(classified_as_word)
        print(b)
        print(all_words[b[0][0]])

        lrm.sess.close()
