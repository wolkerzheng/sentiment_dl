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
