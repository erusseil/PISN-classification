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
    
    """Create a classification algorithm using random forest
    
    Parameters
    ----------
    training: pd.DataFrame
        Table containing all parameters for each object
    save: str
        location and name of the save
    binary: boolean
        If true, all non-PISN we be grouped in the target
        1. Default is True
    ----------

    """
    
    if binary == True :
        isnotPISN_train = training['target']!=994
        training.loc[isnotPISN_train,'target']=1
        
    X_train = training.loc[:,0:]
    y_train = training['target']

    model=RandomForestClassifier()
    forst = model.fit(X_train, y_train)

    pickle.dump(forst, open("%s.sav"%save, 'wb'))
    
    
    
def create_if(training,band_used,nb_param, ntrees):
    
    """Perform anomaly detection using isolation forest
    
    Parameters
    ----------
    training: pd.DataFrame
        Table containing all parameters for each object
    band_used: np.array
        Array of the passbands chosen
    ntrees: int
        Number of trees to use for the isolation forest
    ----------
        
    
    Returns
    ----------
    training: pd.Dataframe
        Original table with added anomaly score columns
    score_df: pd.Dataframe
        Additionnal dataframe containing all the scores on
        a single column
    ----------
    """
    
    print("go")
    # Define useful values
    width = np.shape(training)[1]
    print("width")
    total_band = int((width-2)/nb_param)
    print("total band = ",total_band)
    shift = 6 - total_band
    nb_band = len(band_used)
    
    
    
    
    iso = []
    
    for j in range (len(training)): 
        for i in band_used :
            iso.append(np.array(training.iloc[j, 2+nb_param*(i-shift) : 2 + nb_param*(i-shift+1)]))

    clf = IsolationForest(n_estimators = ntrees).fit(iso)
    
    scores = clf.decision_function(iso)
    df_score = np.reshape(scores,(len(training),nb_band))
    training2 = training.copy()
    
    for i in band_used :

        training2.insert(2+i-shift+nb_param*(i-shift+1), 'score'+str(i), df_score[:,i-shift])
        
    shape_score = {'score':[], 'target':[], 'object_id':[]}
    score_df = pd.DataFrame(data=shape_score)

    for i in band_used:
        score_nb = 'score'+str(i)
        score_df = pd.concat([score_df,training2.loc[:,[score_nb,'target','object_id']].rename(columns={score_nb: "score"})])
        
    return training2,score_df

