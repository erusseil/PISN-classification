import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

import pickle


def create_ml(training,save,binary=True):
    
    if binary == True :
        isnotPISN_train = training['target']!=994
        training.loc[isnotPISN_train,'target']=1
        
    X_train = training.loc[:,0:]
    y_train = training['target']

    model=RandomForestClassifier()
    forst = model.fit(X_train, y_train)

    pickle.dump(forst, open("%s.sav"%save, 'wb'))