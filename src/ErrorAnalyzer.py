import  csv
import numpy as np


class Result(object):
    def __init__(self,id):
        self.id = id
        self.ccm_dic = {}

    def convert2Matrix(self,word_order):

        ccm = np.zeros(len(word_order,word_order)) - 1
        for i in np.arange(len(word_order)):
            for j in np.arange(i):
                ccm[i][j] = self.ccm_dic[word_order[i]][word_order[j]]

        return ccm



class ErrorAnalyzer(object):

    def read_results(self,result_file_name):

        confusion_matrix = np.zeros()
        with open(result_file_name,"r") as rfile:
            res = Result(id=result_file_name)
            lines = rfile.readlines()
            #line 1 is the subject id
            res.subject = int(lines[0])
            #line 2 is number of words
            res.number_of_words = int(lines[1])
            #line 3 is also number of words
            #line 4 is total number of pairs
            res.total_pair_count = int(lines[3])
            #accuracy per iteration on training
            i = 4
            while i <len(lines):
                if lines[i].startswith("pair"):
                    word_pair = lines[i].split(" ")[1].split(",")
                    if word_pair[0] not in res.ccm_dic:
                        res.ccm_dic[word_pair[0]]={}

                    if word_pair[1] not in res.ccm_dic:
                        res.ccm_dic[word_pair[1]]={}

                        res.ccm_dic[word_pair[0]][word_pair[1]] = lines[i+1].split(" ")[-1]
                        res.ccm_dic[word_pair[1]][word_pair[0]] = lines[i + 1].split(" ")[-1]
                    i = i + 2
                else:
                    i = i + 1
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