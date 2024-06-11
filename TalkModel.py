import nltk
import pickle
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt

class TalkClassifier(object):
    def __init__(self, trainData, method='tf-idf'):
        self.talks, self.labels = trainData['message'], trainData['label']
        self.method = method

    def train(self):
        self.calc_TF_and_IDF()
        if self.method == 'tf-idf':
            self.calc_TF_IDF()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word] + 1) / (self.depressive_words + \
                                                                           len(list(self.tf_depressive.keys())))
        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word] + 1) / (self.positive_words + \
                                                                       len(list(self.tf_positive.keys())))
        self.prob_depressive_talk, self.prob_positive_talk = self.depressive_talks / self.total_talks, self.positive_talks / self.total_talks

    def calc_TF_and_IDF(self):
        noOfMessages = self.talks.shape[0]
        self.depressive_talks, self.positive_talks = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_talks = self.depressive_talks + self.positive_talks
        self.depressive_words = 0
        self.positive_words = 0
        self.tf_depressive = dict()
        self.tf_positive = dict()
        self.idf_depressive = dict()
        self.idf_positive = dict()
        for i in range(noOfMessages):
            message_processed = process_message(self.talks.iloc[i])
            count = list()  # To keep track of whether the word has ocured in the message or not.
            # For IDF
            for word in message_processed:
                if self.labels.iloc[i]:
                    self.tf_depressive[word] = self.tf_depressive.get(word, 0) + 1
                    self.depressive_words += 1
                else:
                    self.tf_positive[word] = self.tf_positive.get(word, 0) + 1
                    self.positive_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels.iloc[i]:
                    self.idf_depressive[word] = self.idf_depressive.get(word, 0) + 1
                else:
                    self.idf_positive[word] = self.idf_positive.get(word, 0) + 1
            pickle_out = open("dataB.pickle","wb")
            pickle.dump(self.depressive_words,pickle_out)
            pickle.dump(self.positive_words,pickle_out)
            pickle_out.close()

    def calc_TF_IDF(self):
        self.prob_depressive = dict()
        self.prob_positive = dict()
        self.sum_tf_idf_depressive = 0
        self.sum_tf_idf_positive = 0
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.tf_depressive[word]) * log(
                (self.depressive_talks + self.positive_talks) \
                / (self.idf_depressive[word] + self.idf_positive.get(word, 0)))
            self.sum_tf_idf_depressive += self.prob_depressive[word]
        for word in self.tf_depressive:
            self.prob_depressive[word] = (self.prob_depressive[word] + 1) / (
                        self.sum_tf_idf_depressive + len(list(self.prob_depressive.keys())))

        for word in self.tf_positive:
            self.prob_positive[word] = (self.tf_positive[word]) * log((self.depressive_talks + self.positive_talks) \
                                                                      / (self.idf_depressive.get(word, 0) +
                                                                         self.idf_positive[word]))
            self.sum_tf_idf_positive += self.prob_positive[word]
        for word in self.tf_positive:
            self.prob_positive[word] = (self.prob_positive[word] + 1) / (
                        self.sum_tf_idf_positive + len(list(self.prob_positive.keys())))
        self.prob_depressive_talk, self.prob_positive_talk = self.depressive_talks / self.total_talks, self.positive_talks / self.total_talks

        pickle_out = open("dataA.pickle","wb")
        pickle.dump(self.prob_depressive,pickle_out)
        pickle.dump(self.sum_tf_idf_depressive,pickle_out)
        pickle.dump(self.prob_positive,pickle_out)
        pickle.dump(self.sum_tf_idf_positive,pickle_out)
        pickle.dump(self.prob_depressive_talk,pickle_out)
        pickle.dump(self.prob_positive_talk,pickle_out)
        pickle_out.close()

    def classify(self, processed_message,method):

        pickle_in = open("dataA.pickle","rb")
        prob_depressive = pickle.load(pickle_in)
        sum_tf_idf_depressive = pickle.load(pickle_in)
        prob_positive = pickle.load(pickle_in)
        sum_tf_idf_positive = pickle.load(pickle_in)
        prob_depressive_talk = pickle.load(pickle_in)
        prob_positive_talk = pickle.load(pickle_in)

        pickle_in = open("dataB.pickle","rb")
        depressive_words = pickle.load(pickle_in)
        positive_words = pickle.load(pickle_in)

        pDepressive, pPositive = 0, 0.

        for word in processed_message:
            if word in prob_depressive:
                pDepressive += log(prob_depressive[word])
            else:
                if method == 'tf-idf':
                    pDepressive -= log(sum_tf_idf_depressive + len(list(prob_depressive.keys())))
                else:
                    pDepressive -= log(depressive_words + len(list(prob_depressive.keys())))
            if word in prob_positive:
                pPositive += log(prob_positive[word])
            else:
                if method == 'tf-idf':
                    pPositive -= log(sum_tf_idf_positive + len(list(prob_positive.keys())))
                else:
                    pPositive -= log(positive_words + len(list(prob_positive.keys())))
            pDepressive += log(prob_depressive_talk)
            pPositive += log(prob_positive_talk)
        return pDepressive >= pPositive

    def predict(self, testData,method):
        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message,method))
        return result

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words