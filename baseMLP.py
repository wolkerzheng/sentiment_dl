#encoding=utf8
__author__='ZGD'
import pandas as pd
import re
from bs4 import BeautifulSoup
import  os
from nltk.corpus import stopwords

import nltk.data
import logging
import numpy as np
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import KaggleWord2VecUtility
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout,Activation,Embedding,Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def loadW2V():
    """

    :return:
    """
    model = Word2Vec.load("300features_40minwords_10context")
    # print model.doesnt_match("man woman child kitchen".split())
    # print model.most_similar("man")
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
        #一定要加encode('utf-8')，要不然报错！！！
        #python2对于字符编码问题要慎重
        review_text = BeautifulSoup(review, "lxml").get_text().encode('utf-8')
        review_text = re.sub(r"[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        text = " ".join(words)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        clean_reviews.append(text)
        # clean_reviews.append(KaggleWord2VecUtility.review_to_wordlist(review, remove_stopwords=True))
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
    # w2vModel = loadW2V()
    MAX_SEQUENCE_LENGTH = 100   #每句话最大的单词数
    MAX_NB_WORDS = 200000
    EMBDDING_DIM = 300
    batch_size = 128
    epoch = 10
    #获取文本列表【w1,w2,w3】
    train_text = getCleanReviews(train)
    test_text = getCleanReviews(test)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(train_text+test_text)
    #train_text格式为：["w1 w2 w3 w4','w2 w5 w6']
    x_train_sequences = tokenizer.texts_to_sequences(train_text)
    x_test_sequences = tokenizer.texts_to_sequences(test_text)
    print 'x_train_sequences', len(x_train_sequences)
    word_index = tokenizer.word_index
    print 'Found %s unique tokens' % len(word_index),type(word_index)

    data_1 = pad_sequences(x_train_sequences,maxlen=MAX_SEQUENCE_LENGTH)
    labels = train['sentiment']
    # print data_1.shape,labels.shape     #(25000, 100) (25000,)

    data_2 = pad_sequences(x_test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # test_ids = test["id"]

    # print '构造词向量矩阵'
    # # 取词向量或分词的最小，之所以加1,因为有一维代表改词没有在训练的词向量中的表示
    # nb_words = min(MAX_NB_WORDS,len(word_index))+1
    # embedding_matrix = np.zeros((nb_words,EMBDDING_DIM))
    # for word,i in word_index.items():
    #     if word in w2vModel.vocab:
    #         embedding_matrix[i] = w2vModel[word]
    # print embedding_matrix.shape    #(101226, 300)
    # print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    num_classes = 1
    #定义MLP结构


    # sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedding_layer = Embedding(nb_words,
    #                             EMBDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable=False)
    # # embedding输出：(None, MAX_SEQUENCE_LENGTH, EMBDDING_DIM)
    # embedded_sequences_1 = embedding_layer(sequence_1_input)
    # hidden_layer1 = Dense(512,activation='relu')(embedded_sequences_1)
    # hidden_layer1 = Dropout(0.25)(hidden_layer1)
    # pred = Dense(1,activation='sigmoid')(embedded_sequences_1)
    # model = Model(inputs=sequence_1_input,outputs=pred)



    model = Sequential()
    model.add(Dense(512,input_shape=(MAX_SEQUENCE_LENGTH,),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.25))
    #在第一层dense确定维数之后，就不需要为之后的Dense确定size了
    model.add(Dense(num_classes,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(data_1, labels,
                        epochs=epoch, batch_size=batch_size,
                        validation_split=0.1)
    res = model.predict(data_2, batch_size=batch_size)

    # trainDataVecs = getFeatureVecs(getCleanReviews(train), w2vModel,word_maxlen = 300 , num_features=300)
    # print "Creating average feature vecs for test reviews"
    #
    # testDataVecs = getFeatureVecs(getCleanReviews(test), w2vModel,word_maxlen = 300 , num_features=300)
    # trainDataVecs = np.nan_to_num(trainDataVecs)
    # testDataVecs = np.nan_to_num(testDataVecs)
    # res = mainModel(w2vModel,trainDataVecs,testDataVecs,train['sentiment'])

    output = pd.DataFrame(data={"id": test["id"], "sentiment": res})
    output.to_csv("Word2Vec_mlp.csv", index=False, quoting=3)
    print "Wrote Word2Vec_mlp.csv"