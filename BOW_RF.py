#encoding=utf8

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import KaggleWord2VecUtility
import pandas as pd
import numpy as np
"""
    ROC     scored
100 0.84528 0.54256
200         0.54448



"""
def readData():
    """
    读取数据
    :return:
    """
    print 'loading data...'
    train = pd.read_csv(os.path.join(os.path.dirname(__file__),'data',
                        'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)

    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
                   quoting=3)

    clean_train_reviews = []
    print 'cleaning data ... '
    for i in xrange(0,len(train['review'])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    clean_test_reviews = []
    for i in xrange(0,len(test['review'])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))
    return clean_train_reviews,train['sentiment'],clean_test_reviews,test['id']

def getBOW(clean_train_reviews, clean_test_reviews):
    """
    调用sklearn包来进行词袋统计
    :param clean_train_reviews:
    :param clean_test_reviews:
    :return:
    """
    print 'processing bag of words...'
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=5000)
    #fit_transform：首先fit模型，获得词表，其次，
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    test_data_features = vectorizer.fit_transform(clean_test_reviews)
    return train_data_features,test_data_features

def trainRF(train_data_features,train_label):
    """
    训练随机森林
    :param train_data_features:
    :param train_label:
    :return:
    """

    rf = RandomForestClassifier(n_estimators=200)
    rf = rf.fit(train_data_features,train_label)
    return rf

if __name__ == '__main__':
    clean_train_reviews,train_label ,clean_test_reviews,idd = readData()
    train_data_features,test_data_features = getBOW(clean_train_reviews, clean_test_reviews)


    rf = trainRF(train_data_features,train_label)
    predicted = rf.predict(test_data_features)
    output = pd.DataFrame(data={"id":idd, "sentiment":predicted} )
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_rf_model.csv'),
                  index=False, quoting=3)
    print "Wrote results to Bag_of_Words_model.csv"
