import argparse
from tf_LRModel_GPU import *


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

    args = parser.parse_args()
    print("subject id %s" % args.subject)

    words, x_all, y_all = LRModel.prepare_data(fMRI_file=fMRI_data_path+fMRI_data_filename+args.subject+fMRI_data_postfix,
                                               subject=args.subject,type="deps",select=False)

    load = False
    if load == False:
        word_set = list(set(words))
        accuracies = []
        print(x_all.shape[0])

        x_train, y_train = x_all, y_all

        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_train, y_test=y_train)
        lrm.save_model("../deps_all.model")
        lrm.sess.close()
    else:
        word_set = list(set(words))
        accuracies = []
        print(x_all.shape[0])

        x_train, y_train = x_all, y_all

        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate=expSetup.learning_rate,
                      hidden_dim=y_train.shape[1], training_steps=expSetup.number_of_epochs,
                      batch_size=expSetup.batch_size)
        lrm.load_model("../deps_all.model")
        predicted_brain_activation = lrm.get_prediction(x_train)
        print(word_set[0])
        print(predicted_brain_activation[0].shape)
    print(str(expSetup))
