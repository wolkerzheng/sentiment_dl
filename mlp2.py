#!/bin/bash/env python
# -*- coding: utf-8 -*-
__author__ = 'wolker'
#encoding=utf8

import pandas as pd

import  os

from nltk.corpus import stopwords

import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
# from sklearn.ensemble import RandomForestClassifier
import KaggleWord2VecUtility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input

"""
kaggle得分：0.9436
22500/22500 [==============================] - 1s - loss: 0.3935 - acc: 0.8314 - val_loss: 0.3291 - val_acc: 0.8640
Epoch 2/10
22500/22500 [==============================] - 1s - loss: 0.3242 - acc: 0.8640 - val_loss: 0.3177 - val_acc: 0.8680
Epoch 3/10
22500/22500 [==============================] - 1s - loss: 0.3159 - acc: 0.8648 - val_loss: 0.3198 - val_acc: 0.8724
Epoch 4/10
22500/22500 [==============================] - 1s - loss: 0.3123 - acc: 0.8679 - val_loss: 0.3104 - val_acc: 0.8720
Epoch 5/10
22500/22500 [==============================] - 1s - loss: 0.3081 - acc: 0.8700 - val_loss: 0.3086 - val_acc: 0.8724
Epoch 6/10
22500/22500 [==============================] - 1s - loss: 0.3053 - acc: 0.8718 - val_loss: 0.3091 - val_acc: 0.8728
Epoch 7/10
22500/22500 [==============================] - 1s - loss: 0.3028 - acc: 0.8723 - val_loss: 0.3059 - val_acc: 0.8744
Epoch 8/10
22500/22500 [==============================] - 1s - loss: 0.2999 - acc: 0.8734 - val_loss: 0.3091 - val_acc: 0.8724
Epoch 9/10
22500/22500 [==============================] - 1s - loss: 0.2966 - acc: 0.8754 - val_loss: 0.3025 - val_acc: 0.8764
Epoch 10/10
22500/22500 [==============================] - 1s - loss: 0.2954 - acc: 0.8767 - val_loss: 0.3019 - val_acc: 0.8752
"""

def trainW2V(sentences):
    """

    :param sentences:
    :return:
    """
    # ****** Set parameters and train the word2vec model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)

    # Set values for various parameters

    num_features = 300  #词向量维度
    min_word_count = 40 #词的最少次数,默认为5
    num_workers = 4     #并行的进程数
    context = 10        #窗口大小,默认为5
    downsampling = 1e-3 #


    print 'Training w2v model ...'

    model = Word2Vec(sentences,size=num_features,window=context,
                     min_count=min_word_count)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    model.save(model_name)

    #评价
    model.doesnt_match("man woman child kitchen".split())
    model.doesnt_match("france england germany berlin".split())
    model.doesnt_match("paris berlin london austria".split())
    model.most_similar("man")
    model.most_similar("queen")
    model.most_similar("awful")
    return model
def loadW2V():
    """

    :return:
    """
    model = Word2Vec.load("300features_40minwords_10context")
    print model.doesnt_match("man woman child kitchen".split())
    print model.most_similar("man")
    return model
def getSentences(train,test,unlabeled_train):


    print "Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["review"].size,
                                          test["review"].size, unlabeled_train["review"].size)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    # ****** Split the labeled and unlabeled training sets into clean sentences
    print 'Parsing sentences from training set...'
    for review in train['review']:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
    return sentences

def getAvgFeatureVecs(reviews, model, num_features=300):
    """
    # 给定一个由单词列表组成的评论, 为每句话
    #
    #
    #
    :return:二维数组
    """
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    index2word_set = set(model.index2word)
    for review in reviews:
        if counter % 1000 == 0:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,index2word_set,num_features)
        counter = counter + 1
    return reviewFeatureVecs

def makeFeatureVec(words, model,index2word_set, num_features=300):
    """

    :param words:
    :param model:
    :param num_features:
    :return:
    """
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    #将模型


    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    # print featureVec
    return featureVec

def getCleanReviews(reviews):
    """

    :param reviews:
    :return:
    """
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
    return clean_reviews

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,
                                  delimiter="\t", quoting=3)
    # sentences = getSentences(train,test,unlabeled_train)
    # w2vModel = trainW2V(sentences)
    w2vModel = loadW2V()
    print "Creating average feature vecs for training reviews"
    trainDataVecs = getAvgFeatureVecs(getCleanReviews(train), w2vModel, num_features=300)
    print "Creating average feature vecs for test reviews"

    testDataVecs = getAvgFeatureVecs(getCleanReviews(test), w2vModel, num_features=300)
    trainDataVecs = np.nan_to_num(trainDataVecs)
    testDataVecs = np.nan_to_num(testDataVecs)


    model = Sequential()
    model.add(Dense(256,input_shape=(300,),activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(trainDataVecs, train['sentiment'],
                        epochs=10, batch_size=32,
                        validation_split=0.1)

    res = model.predict(testDataVecs, batch_size=32)

    print 'res is', res.shape

    output = pd.DataFrame(data={"id": test["id"], "sentiment": res[:, 0]})
    output.to_csv("simple_mlp.csv", index=False, quoting=3)
    print "Wrote simple_mlp.csv"