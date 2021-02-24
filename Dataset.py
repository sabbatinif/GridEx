import pandas as pd
import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from pathlib import Path

class Dataset:
    def __init__(self, name, target, sep = ",", dec = ".", rem = [], testSize = 0.2):
        # dataset letto da file
        self.target = target
        self.__readCSV(name, sep, dec)
        self.__remove(rem)
        self.__splitAndScaleData(testSize)
        
    def __readCSV(self, name, sep, dec):
        # lettura dei dati
        self.name = name
        self.dataset = pd.read_csv("datasets/{}".format(name), sep = sep, decimal = dec, 
                                   encoding = "ISO-8859-1", engine='python')  
        
    def __remove(self, name):
        for n in name:
            self.dataset = self.dataset.drop(n, axis = 1)
        
    # metodo per dividere gli esempi in train set e test set    
    def __splitData(self, testSize):
        X = self.dataset.drop(self.target, axis = 1)
        y = self.dataset[self.target]
        return train_test_split(X, y, test_size = testSize)
    
    # metodo per dividere gli esempi e scalarli
    def __splitAndScaleData(self, testSize):
        Xtrain, Xtest, self.ytrain, self.ytest = self.__splitData(testSize)
        scaler = StandardScaler()
        scaler.fit(Xtrain)
        self.Xtrain = pd.DataFrame(scaler.transform(Xtrain), columns = Xtrain.columns)
        self.Xtest = pd.DataFrame(scaler.transform(Xtest), columns = Xtest.columns)
        Path("datasets/train/y").mkdir(parents = True, exist_ok = True)
        Path("datasets/test/y").mkdir(parents = True, exist_ok = True)
        Path("datasets/train/x").mkdir(parents = True, exist_ok = True)
        Path("datasets/test/x").mkdir(parents = True, exist_ok = True)
        dump(self.ytrain, "datasets/train/y/{}.joblib".format(self.name))
        dump(self.ytest, "datasets/test/y/{}.joblib".format(self.name))
        dump(self.Xtrain, "datasets/train/x/{}.joblib".format(self.name))
        dump(self.Xtest, "datasets/test/x/{}.joblib".format(self.name))
        
    def getSamples(self):
        return self.Xtrain, self.Xtest, self.ytrain, self.ytest
    
    def describe(self):
        return self.dataset.describe()