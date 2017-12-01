import argparse
from tf_LRModel_GPU import *
from WordEmbeddingLayer import *
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pylab import *



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



if __name__ == '__main__':

    expSetup = ExpSetup(learning_rate=0.001,batch_size=29,number_of_epochs=700)

    fMRI_data_path = "../data/"
    fMRI_data_filename = "data_"
    fMRI_data_postfix = ".csv"
    subject_id = str(1)

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Single Layer Feed Forward Network for Brain Activation Prediction Task')
    parser.add_argument('--subject','-s', type=str, nargs='?',
                        help='An optional integer specifying the subject id',default="1")
    parser.add_argument('--words', type=str, nargs='+',
                        help='words')

    args = parser.parse_args()
    print("subject id %s" % args.subject)

    input_words = []

    for word in args.words:
        print(word)
        input_words.append(word)

    input_words = np.asarray(input_words,dtype=object)
    print(input_words.shape)

    words, x_all, y_all = LRModel.prepare_data(fMRI_file=fMRI_data_path+fMRI_data_filename+args.subject+fMRI_data_postfix,
                                               subject=args.subject,type="glove",select=False)

    load = True
    if load == False:
        word_set = list(set(words))
        accuracies = []
        print(x_all.shape[0])

        x_train, y_train = x_all, y_all

        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train)
        lrm.save_model("../glove_all.model")
        lrm.sess.close()
        print(str(expSetup))

    else:
        x_train, y_train = x_all, y_all

        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate=expSetup.learning_rate,
                      hidden_dim=y_train.shape[1], training_steps=expSetup.number_of_epochs,
                      batch_size=expSetup.batch_size)
        lrm.load_model("../glove_all.model")



        print(input_words)

        wem = WordEmbeddingLayer()
        wem.load_filtered_embedding("../data/glove_all_6B_300d")  # neuro_words
        embedded_words = wem.embed_words(words)


        coords = []
        with open('../data/coords', 'r') as f:
            reader = csv.reader(f)
            coords = list(reader)
        coords = np.asarray(coords)

        fig = plt.figure()
        coord = np.asarray(coords, dtype=int)
        print(coords.shape)
        ax = [fig.add_subplot("221", projection='3d'),
              fig.add_subplot("222", projection='3d')]

        i = 1;
        for e_word,word in zip(embedded_words,input_words):
            predicted_brain_activation = lrm.get_prediction([e_word])
            print(word)
            print(predicted_brain_activation[0].shape)


            ax[i-1].scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                       np.asarray(coords, dtype=int)[:, [2]],
                       s=2,c=predicted_brain_activation[0], alpha=0.8)

            if(word in words):
                real_activation = y_all[np.where(np.asarray(words) == word)]
                print(word)
                print(words)
                print(np.where(np.asarray(words) == word))

                ax.append(fig.add_subplot("22"+str(i+2), projection='3d'))
                ax[-1].scatter(np.asarray(coords, dtype=int)[:, [0]], np.asarray(coords, dtype=int)[:, [1]],
                                  np.asarray(coords, dtype=int)[:, [2]],
                                  s=2, c=real_activation, alpha=0.8)

            angel = 0
            zangle = 90
            ax[i-1].view_init(zangle, angel)
            if (word in words):
                ax[-1].view_init(zangle, angel)
            i += 1


        plt.show()
        lrm.sess.close()