from scipy.stats.stats import pearsonr
from scipy import *
import scipy
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold
import numpy as np

def select_stable_voxels(the_brain_activations, words, allwords,
                         number_of_trials=6,
                         size_of_selection=500):
    stability_matrix = np.zeros((len(the_brain_activations[0]), number_of_trials, len(words)))

    for k in np.arange(len(the_brain_activations[0])):
        word_trials = {}
        for m in np.arange(len(the_brain_activations)):
                if allwords[m] not in word_trials.keys():
                    word_trials[allwords[m]] = 0
                else:
                    word_trials[allwords[m]] += 1
                stability_matrix[k][word_trials[allwords[m]]][words.index(allwords[m])] = the_brain_activations[m][k]

    stability = np.zeros(len(the_brain_activations[0]))
    for k in np.arange(stability_matrix.shape[0]):
        pairs_corr = []
        for i in np.arange(number_of_trials):
            for j in np.arange(i, number_of_trials):
                pairs_corr.append(pearsonr(stability_matrix[k][i], stability_matrix[k][j])[0])
        stability[k] = mean(pairs_corr)

    ind = np.argpartition(stability, -size_of_selection)[-size_of_selection:]

    return ind


def get_pairs(num):
    pairs = []
    for i in np.arange(num):
        for j in np.arange(i):
            pairs.append((i, j))

    return pairs


def get_word_representation(type,words):
    word_features = {}
    features = []
    if type == "F25":
        with open("../data/F25/word_features.txt") as f:
            content = f.readlines()
            for line in content:
                parts = line.split(" ")
                word_features[parts[0]] = [ float(p) for p in parts[1:]]
        for i in np.arange(len(words)):
            if words[i] in word_features.keys():
                features.append(word_features[words[i]])
            else:
                features.append(np.zeros(len(list(word_features.values())[0])))

    return word_features, features


def match_prediction(predicted_1,predicted_2, pair,b_activations,words,word_set):
    matched = []
    mismatched = []
    i1s = []
    i2s = []

    for i in np.arange(len(words)):
        if words[i] == word_set[pair[0]]:
            i1s.append(b_activations[i])
        if words[i] == word_set[pair[1]]:
            i2s.append(b_activations[i])



    for i1 in i1s:
        for i2 in i2s:
            cosin_1_1, cosin_1_2, cosin_2_1, cosin_2_2 = scipy.spatial.distance.euclidean(predicted_1,i1), \
                                                         scipy.spatial.distance.euclidean(predicted_1,i2), \
                                                         scipy.spatial.distance.euclidean(predicted_2,i1), \
                                                         scipy.spatial.distance.euclidean(predicted_2,i2)

            matched_score = cosin_1_1 + cosin_2_2
            mismached_score = cosin_1_2 + cosin_2_1


            if matched_score < mismached_score:
                matched.append(1)
            else:
                mismatched.append(1)


    print(str(matched)+" "+str(mismatched))

    """if matched == 0:
        return 0
    if mismatched == 0:
        return 1
    """
    return np.sum(matched) > np.sum(mismatched) # (matched_score/matched) < (mismached_score/mismatched)


def avereaged_match_prediction(predicted_1,predicted_2, pair,b_activations_dic,words):
    matched = 0
    mismatched = 0

    i1s = b_activations_dic[words[pair[0]]]
    i2s = b_activations_dic[words[pair[1]]]
    #i1s = [np.mean(i1s,axis=0)]
    #i2s = [np.mean(i2s,axis=0)]

    matched_score = []
    mismached_score = []
    for i1 in i1s:
        for i2 in i2s:
            cosin_1_1, cosin_1_2, cosin_2_1, cosin_2_2 = scipy.spatial.distance.cosine(predicted_1,i1), \
                                                         scipy.spatial.distance.cosine(predicted_1,i2), \
                                                         scipy.spatial.distance.cosine(predicted_2,i1), \
                                                         scipy.spatial.distance.cosine(predicted_2,i2)

            matched_score.append(cosin_1_1 + cosin_2_2)
            mismached_score.append(cosin_1_2 + cosin_2_1)

            """if matched_score < mismached_score:
                matched +=1
            else:
                mismatched +=1
            """


    return (np.min(matched_score) < np.min(mismached_score))


def nearest_neighbor(predicted,all_targets_tree,true_targets):
    dd, ii = all_targets_tree.query(predicted)


def relu(mat):
    return np.max(np.asarray([np.zeros(mat.shape),mat]),axis=0)

def softmax(mat):
    if len(mat.shape) == 1:
        e_mat = np.exp(mat - np.max(mat))
        return e_mat / np.sum(e_mat)
    if len(mat.shape) == 2:
        e_mat = np.exp(mat - np.max(mat))
        return e_mat / e_mat.sum(axis=1)[:, np.newaxis]

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def normalize(mat):
    if len(mat.shape) == 1:
        return mat / np.sum(mat)
    if len(mat.shape) == 2:
        return mat / np.sum(mat, axis=1)[:, np.newaxis]

def scale_linear_bycolumn(rawpoints, high=1.0, low=-1.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

def jsdiv(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) +_kldiv(Q, M))


def plot_embedding(features, classes, labels, title=None):
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    features = (features - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(features.shape[0]):
        plt.text(features[i, 0], features[i, 1], str(labels[i]),
                 color=plt.cm.Set1(float(classes[i]/60)),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(features.shape[0]):
            dist = np.sum((features[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
                # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [features[i]]]
            """imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)"""
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



def plot_distribution_t_SNE(activations,words,labels):
    print("Computing t-SNE embedding")

    x = np.asarray(activations)
    #x = preprocessing.normalize(x, norm='l2')

    tsne = manifold.TSNE(n_components=2, init='pca', perplexity=2,n_iter=20000,early_exaggeration=10,learning_rate=300, method="exact")
    X_tsne = tsne.fit_transform(x)

    plot_embedding(X_tsne, np.asarray(words), labels,
                                       "t-SNE embedding of the brain activations")

    plt.show()
