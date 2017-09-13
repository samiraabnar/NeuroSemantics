import  csv
import numpy as np
import matplotlib.pyplot as plt
from prettyplotlib import brewer2mpl
from matplotlib import colors

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class Result(object):
    def __init__(self,id):
        self.id = id
        self.accuracy = 0
        self.ccm_dic = {}

    def convert2Matrix(self,word_order):

        ccm = np.zeros((word_order.shape[0],word_order.shape[0]))
        for i in np.arange(word_order.shape[0]):
            for j in np.arange(i):
                ccm[i][j] = 1.0 - self.ccm_dic[word_order[i]][word_order[j]]

        return ccm



class ErrorAnalyzer(object):

    def read_results(self,result_file_name):

        res = Result(id=result_file_name)
        with open(result_file_name,"r") as rfile:

            lines = rfile.readlines()
            #line 1 is the subject id
            res.subject = int(lines[0].strip().split()[-1])
            #line 2 is number of words
            res.number_of_words = int(lines[1].strip().split()[-1])
            #line 3 is also number of words
            #line 4 is total number of pairs
            res.total_pair_count = int(lines[3].strip().split()[-1])
            #accuracy per iteration on training
            i = 4
            while i <len(lines):
                if lines[i].startswith("pair"):
                    #print(lines[i])
                    word_pair = lines[i].strip().split(" ")[1].split(",")
                    if word_pair[0] not in res.ccm_dic.keys():
                        res.ccm_dic[word_pair[0]]={}

                    if word_pair[1] not in res.ccm_dic.keys():
                        res.ccm_dic[word_pair[1]]={}

                    res.ccm_dic[word_pair[0]][word_pair[1]] = float(lines[i+1].strip().split(" ")[-1])
                    res.ccm_dic[word_pair[1]][word_pair[0]] = float(lines[i + 1].strip().split(" ")[-1])

                    i = i + 2
                elif lines[i].startswith("accuracy"):
                    res.accuracy = float(lines[i].strip().split()[-1])
                    i = i + 1
                else:
                    #print(lines[i])
                    i = i + 1

            #pair nouns
            #test results

            return res

    def print_table_mat(self,mat,words):

        for w in words:
            print(w, end=",")
        print("\n")

        for i in np.arange(mat.shape[0]):
            row = mat[i]
            print(words[i],end=",")
            for item in row:
                print(item, end=",")
            print("\n")




    def compute_mean_accuracy(self,results):

        sum_of_accuracies = 0.0
        for res in results:
            sum_of_accuracies += res.accuracy

        return sum_of_accuracies / len(results)


    def compute_mean_matrix(self,mats):
        ccm = np.zeros_like(mats[0])
        for m in mats:
            ccm += m

        return ccm


def get_res_mat(words,model_name, model_prefix,mode,):
    mode = "limited"
    if mode == 'limited':
        with open('../data/experimental_wordList.csv', 'r') as f:
            word_set = [w[0] for w in list(csv.reader(f))]

        sorted_words = []
        for c_index in conds_sorted_indexes:
            if words[c_index] in word_set:
                sorted_words.append(words[c_index])

        sorted_words = np.asarray(sorted_words)

    ea = ErrorAnalyzer()

    res = {}
    ccm = {}

    for sub_id in np.arange(1, 10):
        res[sub_id] = ea.read_results(model_prefix +model_name+ str(sub_id))
        ccm[sub_id] = res[sub_id].convert2Matrix(word_order=sorted_words)
    print("mean accuracy for "+model_name+" :", ea.compute_mean_accuracy(list(res.values())))
    # print("mean matrix: ", ea.compute_mean_matrix(list(ccm.values())))
    mean_mat = ea.compute_mean_matrix(list(ccm.values()))

    return res,ccm,mean_mat


def plot_heatmap(mat,model_name, cmap=plt.get_cmap('gray_r', 11)):
    fig, ax = plt.subplots()
    heatmap = ax.pcolormesh(mat, cmap=cmap)
    # format
    fig = plt.gcf()
    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5, minor=False)
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.set_xticklabels(sorted_words, minor=False, family='sans-serif', size='9')
    ax.set_yticklabels(sorted_words, minor=False, family='sans-serif', size='9')

    # rotate
    plt.xticks(rotation=90)
    # remove gridlines
    ax.grid(False)
    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    plt.style.use('fivethirtyeight')
    plt.savefig("../plots/" + model_name + ".pdf", format="pdf", transparent=True)


def one_subject_diff_plot(ccm_deps_limited,ccm_experiential,name):
    subject = 1
    diff_ccm = np.zeros_like(ccm_deps_limited[subject])
    for i in np.arange(diff_ccm.shape[0]):
        for j in np.arange(diff_ccm.shape[1]):
            if (ccm_deps_limited[subject][i][j] == 1) and (ccm_experiential[subject][i][j] == 1):
                diff_ccm[i][j] = 3
            elif (ccm_deps_limited[subject][i][j] == 1):
                diff_ccm[i][j] = 1
            elif (ccm_experiential[subject][i][j] == 1):
                diff_ccm[i][j] = 2
    plot_heatmap(diff_ccm, model_name=name,
                 cmap=colors.ListedColormap(['white', 'red', 'blue', 'purple']))


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
    conds = [int(c[0]) for c in conds_1][:60]


    conds_sorted_indexes = np.argsort(conds)
    print("sorted conds:", conds)

    sorted_words = np.asarray(words)[conds_sorted_indexes]




    #model_name = "res_deps_limited_sub"
    #model_name = "experiential_2_sub"


    res, ccm, mean_mat_F25 = get_res_mat(model_name="res_F25_limited_sub",
                                         model_prefix="../backup_results/", words=sorted_words, mode="limited")

    res, ccm, mean_mat_experiential = get_res_mat(model_name="experiential_2_sub",
                                                  model_prefix="../backup_results/", words=sorted_words, mode="limited")

    plot_heatmap(np.abs(mean_mat_F25 - mean_mat_experiential), model_name="diff_F25-exp")
    plot_heatmap(mean_mat_F25, model_name="F25-limited")
    plot_heatmap(mean_mat_experiential, model_name="exp")


    res, ccm, mean_mat_deps =get_res_mat(model_name = "res_deps_sub",
                model_prefix="../backup_results/",words=sorted_words,mode="none")

    res, ccm, mean_mat_glove = get_res_mat(model_name="res_glove_new_sub",
                                     model_prefix="../results_gpu/", words=sorted_words, mode="none")

    plot_heatmap(np.abs(mean_mat_deps - mean_mat_glove),model_name="diff_glove-deps")

    plot_heatmap(mean_mat_deps, model_name="deps")
    plot_heatmap(mean_mat_glove, model_name="glove")

    res, ccm, mean_mat_deps_limited = get_res_mat(model_name="res_deps_limited_sub",
                                           model_prefix="../backup_results/", words=sorted_words, mode="limited")


    plot_heatmap(mean_mat_deps_limited, model_name="deps-limited")
    plot_heatmap(np.abs(mean_mat_deps_limited - mean_mat_experiential),model_name="diff_exp-deps")

    res, ccm, mean_mat_F25 = get_res_mat(model_name="res_F25_sub",
                                         model_prefix="../backup_results/", words=sorted_words, mode="none")
    plot_heatmap(mean_mat_F25, model_name="F25")

    plot_heatmap(np.abs(mean_mat_deps - mean_mat_F25),model_name="diff_F25-deps")




    res, ccm_F25, mean_mat_F25 = get_res_mat(model_name="res_F25_limited_sub",
                                                               model_prefix="../backup_results/", words=sorted_words,
                                                               mode="limited")

    res, ccm_experiential, mean_mat_experiential = get_res_mat(model_name="experiential_2_sub",
                                                  model_prefix="../backup_results/", words=sorted_words, mode="limited")

    res, ccm_deps_limited, mean_mat_deps_limited = get_res_mat(model_name="res_deps_limited_sub",
                                                  model_prefix="../backup_results/", words=sorted_words, mode="limited")

    res, ccm_word2vec_limited, mean_mat_word2vec_limited = get_res_mat(model_name="res_word2vec_limited_sub",
                                                               model_prefix="../b_results/", words=sorted_words,
                                                               mode="limited")

    res, ccm_glove_limited, mean_mat_glove_limited = get_res_mat(model_name="res_glove_limited_new_sub",
                                                                       model_prefix="../results_gpu/", words=sorted_words,
                                                                       mode="limited")

    res, ccm_fasttext_limited, mean_mat_fasttext_limited = get_res_mat(model_name="res_fasttext_limited_sub",
                                                                 model_prefix="../backup_results/", words=sorted_words,
                                                                 mode="limited")

    res, ccm_lexvec_limited, mean_mat_lexvec_limited = get_res_mat(model_name="res_lexvec_limited_sub",
                                                                       model_prefix="../backup_results/", words=sorted_words,
                                                                       mode="limited")

    res, ccm_nondist_limited, mean_mat_nondist_limited = get_res_mat(model_name="res_nondist_limited_sub",
                                                                   model_prefix="../backup_results/", words=sorted_words,
                                                                   mode="limited")

    plot_heatmap(np.sum([mean_mat_F25,mean_mat_experiential, mean_mat_deps_limited,
                         mean_mat_fasttext_limited,mean_mat_glove_limited, mean_mat_word2vec_limited,
                         mean_mat_lexvec_limited,mean_mat_nondist_limited
                         ],axis=0
                        ),
                 model_name="sum_all_limited")


    one_subject_diff_plot(ccm_F25, ccm_experiential, name="diff_F25-exp_sub1")
    one_subject_diff_plot(ccm_deps_limited, ccm_experiential, name="diff_deps-exp_sub1")
    one_subject_diff_plot(ccm_deps_limited,ccm_word2vec_limited,name="diff_deps-word2vec_sub1")
    one_subject_diff_plot(ccm_deps_limited, ccm_glove_limited, name="diff_deps-glove_sub1")




