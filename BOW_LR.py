#encoding=utf8

"""
0.84392
accuracy:0.86
得分：0.52912
"""
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import KaggleWord2VecUtility
import pandas as pd
import numpy as np
import csv


def funcLR():
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
    print 'cleaning trainset'
    for i in xrange(0,len(train['review'])):
        clean_train_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))

    clean_test_reviews = []
    for i in xrange(0,len(test['review'])):
        clean_test_reviews.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))

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

    """
    训练逻辑回归
    :param train_data_features:
    :param train_label:
    :return:
    """

    lr = LogisticRegression()
    lr.fit(train_data_features,train['sentiment'])
    predicted = lr.predict(test_data_features)
    output = pd.DataFrame(data={"id": test['id'], "sentiment": predicted})
    output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_lr_model.csv'),
                  index=False, quoting=3)
    print "Wrote results to Bag_of_Words_lr_model.csv"


if __name__ == '__main__':
    funcLR()


