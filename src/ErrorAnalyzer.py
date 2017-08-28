import  csv
import numpy as np

class ErrorAnalyzer(object):

    def read_results(self,result_file_name):

        confusion_matrix = np.zeros()
        with open(result_file_name,"r") as rfile:
            lines = rfile.readlines()
            #line 1 is the subject id
            #line 2 is number of words
            #line 3 is also number of words
            #line 4 is total number of pairs

            #accuracy per iteration on training
            #pair nouns
            #test results



if __name__ == '__main__':

    words_1 = []
    with open('../data/words', 'r') as f:
        reader = csv.reader(f)
        words_1 = list(reader)

    words = []
    words.extend([w[0] for w in words_1])


    conds_1 = []
    with open('../data/conds', 'r') as f:
        reader = csv.reader(f)
        conds_1 = list(reader)
    conds = [int(c[0]) for c in conds_1]