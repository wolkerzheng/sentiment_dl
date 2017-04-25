#encoding=utf8

import pandas as pd

import  os

from nltk.corpus import stopwords

import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import KaggleWord2VecUtility
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation


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

def getFeatureVecs(reviews, model,word_maxlen=300, num_features=300):
    """
    # 给定一个由单词列表组成的评论, 为每句话
    #
    #
    #
    :return:二维数组
    """
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), word_maxlen), dtype="object")

    index2word_set = set(model.index2word)

    for review in reviews:
        if counter % 1000 == 0:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,index2word_set,num_features)

        counter = counter + 1
    return reviewFeatureVecs

def makeFeatureVec(words, model,index2word_set,word_maxlen=300, num_features=300):
    """

    :param words:
    :param model:
    :param num_features:
    :return:
    """
    featureVec = []
    nwords = 0
    #将模型

    for word in words:
        if word in index2word_set:
            featureVec.append(model[word])
            nwords = nwords + 1
    if nwords>=word_maxlen:
        return featureVec[0:word_maxlen]

    while nwords<word_maxlen:
        featureVec.append(np.zeros((num_features)))
        nwords += 1
    # featureVec = np.divide(featureVec,nwords)
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

def mainModel(w2vModel,trainDataVecs,testDataVecs,y_label):


    num_features = 300
    output_unit = 2
    batch_size = 40
    epoch = 10
    word_maxlen = 300
    num_class = 2
    # x_train =
    print 'building model ... '
    model = Sequential()
    #64个隐藏单元
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class))
    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    h = model.fit(trainDataVecs,y_label,
                  epochs = epoch,
                  batch_size=batch_size,
                  verbose=1,
                  validation_data=0.2)

    p = model.predict(testDataVecs,batch_size=batch_size,verbose=1)
    return p
    pass



if __name__ == '__main__':
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    w2vModel = loadW2V()
    word_maxlen = 300   #每句话最大的单词数
    trainDataVecs = getFeatureVecs(getCleanReviews(train), w2vModel,word_maxlen = 300 , num_features=300)
    print "Creating average feature vecs for test reviews"

    testDataVecs = getFeatureVecs(getCleanReviews(test), w2vModel,word_maxlen = 300 , num_features=300)
    trainDataVecs = np.nan_to_num(trainDataVecs)
    testDataVecs = np.nan_to_num(testDataVecs)
    res = mainModel(w2vModel,trainDataVecs,testDataVecs,train['sentiment'])

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)
    print "Wrote Word2Vec_AverageVectors.csv"