import argparse
from tf_LRModel_reversed_GPU import *


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

    words, y_all, x_all = LRModel.prepare_data(fMRI_file=fMRI_data_path+fMRI_data_filename+args.subject+fMRI_data_postfix,
                                               subject=args.subject,type="lexvec")


    word_set = list(set(words))
    accuracies = []
    print(x_all.shape[0], x_all.shape[1])
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
        x_test, y_test = np.asarray([np.mean(x_all[test_indices_1],axis=0),np.mean(x_all[test_indices_2],axis=0)]), \
                         np.asarray([y_all[test_indices_1][0],
                                   y_all[test_indices_2][0]])

        #print("x_train shape: " + str(x_train.shape))
        #print("y_train shape: " + str(y_train.shape))
        #print("x_test shape: " + str(x_test.shape))
        #print("y_test shape: " + str(y_test.shape))
        lrm = LRModel(x_train.shape[1], y_train.shape[1], learning_rate= expSetup.learning_rate,hidden_dim=y_train.shape[1],training_steps=expSetup.number_of_epochs, batch_size=expSetup.batch_size)
        lrm.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        print("pair: %s" % word_set[test_word_indices[0]]+","+word_set[test_word_indices[1]])
        loss, acc2 = lrm.test(x_test=x_test, y_test=y_test)
        lrm.sess.close()
        accuracies.append(acc2)

    print("accuracy: ", np.mean(accuracies))
    print(str(expSetup))
