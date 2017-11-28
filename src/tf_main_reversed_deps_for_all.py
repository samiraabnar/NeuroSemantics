import argparse
from tf_LRModel_reversed import *


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

    expSetup = ExpSetup(learning_rate=0.001,batch_size=29,number_of_epochs=600)

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

    load = False

    if load == False:
        word_set = list(set(words))
        accuracies = []
        print(x_all.shape[0], x_all.shape[1])
        words = np.asarray(words)
        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train)
        lrm.save("../glove_reversed_all.model")
        lrm.sess.close()

        print("accuracy: ", np.mean(accuracies))
        print(str(expSetup))
    else:
        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.load("../glove_reversed_all.model")
        lrm.sess.close()
