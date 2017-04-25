#encoding=utf8

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
import os
import KaggleWord2VecUtility
import time

def create_bag_of_centroids(wordlist,word_centroid_map):

    """

    :param wordlist:
    :param word_centroid_map:
    :return:
    """
    num_centroids = max(word_centroid_map.values()) + 1

    bag_of_centroids = np.zeros(num_centroids,dtype='float32')

    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

def loadW2V():
    """

    :return:
    """
    model = Word2Vec.load("300features_40minwords_10context")

    print model.doesnt_match("man woman child kitchen".split())
    print model.most_similar("man")
    return model

if __name__ == '__main__':

    startTime = time.time()

    model = loadW2V()
    word_vectors = model.syn0   #(16490,300)
    print word_vectors.shape
    num_cluster = word_vectors.shape[0] / 5

    kmeans = KMeans(n_clusters=num_cluster)
    idx = kmeans.fit_predict(word_vectors)
    print idx
    word_centroid_map = dict(zip(model.index2word, idx))
    for cluster in xrange(0, 10):
        #
        # Print the cluster number
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
                words.append(word_centroid_map.keys()[i])
        print words

    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)

    print "Cleaning training reviews"
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, \
                                                                            remove_stopwords=True))

    print "Cleaning test reviews"
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, \
                                                                           remove_stopwords=True))
    train_centroids = np.zeros((train["review"].size, num_cluster), \
                               dtype="float32")

    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, \
                                                           word_centroid_map)
        counter += 1

    test_centroids = np.zeros((test["review"].size, num_cluster), \
                              dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, \
                                                          word_centroid_map)
        counter += 1

    forest = RandomForestClassifier(n_estimators=100)
    print "Fitting a random forest to labeled training data..."
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)
    print "Wrote BagOfCentroids.csv"
    endTime = time.time()

    print 'time is :',endTime-startTime