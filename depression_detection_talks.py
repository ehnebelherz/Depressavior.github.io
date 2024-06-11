import nltk
nltk.download('punkt')
from TalkModel import TalkClassifier,process_message
from math import log, sqrt
import pandas as pd
import numpy as np
import pickle

class DepressionDetection:

    """# Loading the Data"""
    def __init__(self):
        self.talks = pd.read_csv('dataset/talks.csv')
        self.talks.drop(['Unnamed: 0'], axis = 1, inplace = True)
        self.talks['label'].value_counts()
        self.talks.info()

        self.totalTalks = 8000 + 2314
        trainIndex, testIndex = list(), list()
        for i in range(self.talks.shape[0]):
            if np.random.uniform(0, 1) < 0.98:
                trainIndex += [i]
            else:
                testIndex += [i]

        self.trainData = self.talks.iloc[trainIndex]
        self.testData = self.talks.iloc[testIndex]
        self.trainData['label'].value_counts()
        self.testData['label'].value_counts()

    def classify(processed_message,method):

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
        if pDepressive >= pPositive:
            return 1
        else:
            return 0

    def metrics(self,labels, predictions):
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for i in range(len(labels)):
            true_pos += int(labels.iloc[i] == 1 and predictions[i] == 1)
            true_neg += int(labels.iloc[i] == 0 and predictions[i] == 0)
            false_pos += int(labels.iloc[i] == 0 and predictions[i] == 1)
            false_neg += int(labels.iloc[i] == 1 and predictions[i] == 0)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)


if __name__ == "__main__":

    obj = DepressionDetection()
    sc_tf_idf = TalkClassifier(obj.trainData, 'tf-idf')
    #sc_tf_idf.train()
    preds_tf_idf = sc_tf_idf.predict(obj.testData['message'],'tf-idf')
    obj.metrics(obj.testData['label'], preds_tf_idf)

    sc_bow = TalkClassifier(obj.trainData, 'bow')
    #sc_bow.train()
    preds_bow = sc_bow.predict(obj.testData['message'],'bow')
    obj.metrics(obj.testData['label'], preds_bow)

    """# Predictions with TF-IDF
    # Depressive Talks
    """
    pm = process_message('Extreme sadness, lack of energy, hopelessness')
    print(f"Extreme sadness, lack of energy, hopelessness : {sc_tf_idf.classify(pm,'tf-idf')}")
    """# Positive Talks"""
    pm = process_message('Loving how me and my lovely partner is talking about what we want.')
    print(f"Loving how me and my lovely partner is talking about what we want. : {sc_tf_idf.classify(pm,'tf-idf')}")



    """# Predictions with Bag-of-Words (BOW)
    # Depressive talks """
    pm = process_message('Hi hello depression and anxiety are the worst')
    sc_bow.classify(pm,'bow')
    """# Positive Talks"""
    pm = process_message('Loving how me and my lovely partner is talking about what we want.')
    sc_bow.classify(pm,'bow')

