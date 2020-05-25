import os
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from joblib import dump
from process import tokanize, clean, getVocabulary, getBOW
from utils import readTestingData, readTrainingData
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import json

DEBUG = False
def main():
    # read the data:
    print("reading data...")
    trainingData, trainingLabels = readTrainingData()
    testingData, testingLabels = readTestingData()
    print("done")

    #tokanize:
    print("tokanize...")
    trainingTokens = tokanize(trainingData)
    testingTokens = tokanize(testingData)  
    #print(trainingTokens[:5])
    print("done")

    # clean data
    print("cleaning the data...")
    trainingTokens = clean(trainingTokens)
    testingTokens = clean(testingTokens)
    print("done")

    # #write vocabulary 
    print("creating vocabulary")
    f = open("../data/vocab.json", "w")
    vocab = getVocabulary(trainingTokens)
    json_vocab = json.dumps(vocab)
    f.write(json_vocab)
    print(len(vocab))
    
    
    #read vocabulary
    # print("load vocabulary...")
    # f = open("../data/vocab.json", "r")
    # with open("../data/vocab.json") as vocab_file:
    #     vocab = json.load(vocab_file)
    # print(len(vocab))
    # print("done!")

    # create vector
    print("creating bag of words")
    words_vector = getBOW(trainingTokens, vocab)
    words_vector_test = getBOW(testingTokens, vocab)
    #print(words_vector[2])
    print("done")

    #create model

    #logistic regression:
    #clf = LogisticRegression(verbose = True, random_state=0)

    #decision tree:
    #clf = tree.DecisionTreeClassifier()

    #Multy-layer Perceptron:   
    clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(112, 10), random_state=1)

    #print("training the model...")
    clf.fit(words_vector, trainingLabels)
    #print("done")
    score = clf.score(words_vector_test, testingLabels)
    print("score: "+ str(score))
   
 
if __name__ == "__main__":
    main()
