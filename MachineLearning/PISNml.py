import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import IsolationForest

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
    
    
    
def create_if(training,nb_band,nb_param):#,save):
    
    iso = []
    
    for j in range (len(training)):
        
        for i in range (nb_band) :
            
            iso.append(np.array(training.iloc[j, 2+nb_param*i : 2 + nb_param*(i+1)]))

    
    clf = IsolationForest(random_state=0).fit(iso)
    
    scores = clf.decision_function(iso)
    df_score = np.reshape(scores,(len(training),nb_band))
    
    for i in range (nb_band) :
        
        training.insert(2+nb_param + (nb_param+1)*i, 'score'+str(i+2), df_score[:,i])
        
    return training