#encoding=utf8

from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
def review_to_wordlist(review,remove_stopwords=False):
    """
    :param review:
    :param remove_stopwords:
    :return: 单词词表
    """
    review_text = BeautifulSoup(review, "lxml").get_text()
    #去除非字母的字符

    review_text = re.sub(r"[^a-zA-Z]"," ",review_text)
    #小写
    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    """

    :param review:
    :param tokenizer:
    :param remove_stopwords:
    :return:
    """
    #用nltk,tokenize来将段落分句
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())
    sentences = []

    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    return  sentences