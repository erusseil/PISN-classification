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



def create_ml(training,save,binary=True,target=994,nb_tree=100):
    
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
    target: int
        Integer associated to the class to be find by the model
        Default is 994 (PISN)
    nb_tree: int
        Number of tree to use for the random forest
        Default is 100 
    ----------

    """
    
    if binary == True :
        isnotPISN_train = training['target']!=target
        training.loc[isnotPISN_train,'target']=1
        
    X_train = training.loc[:,0:]
    y_train = training['target']

    model=RandomForestClassifier(n_estimators=nb_tree)
    forst = model.fit(X_train, y_train)

    pickle.dump(forst, open("%s.sav"%save, 'wb'))
    
    
    
def create_if(data,band_used,nb_param, ntrees, split_band=0):
    
    """Perform anomaly detection using isolation forest
    
    Parameters
    ----------
    data: pd.DataFrame
        Table containing all parameters for each object
    band_used: np.array
        Array of the passbands chosen
    ntrees: int
        Number of trees to use for the isolation forest
    split_band: int
        If 2 performs n isolation forests in the n passbands. Each light curve is an object
        If 1 performs 1 isolation forest where each light curve is an object
        Else performs 1 isolation where each object is the collection of all it's lightcurves
        Default is 0
    ----------
         
    
    Returns
    ----------
    score_df: pd.Dataframe
        Dataframe containing all the scores on
        a single column
    ----------
    """

        
    # Define useful values
    width = np.shape(data)[1]
    nb_band = len(band_used)
  
    data2 = data.copy()

    clf = IsolationForest(n_estimators = ntrees).fit(data.iloc[:,2:])
    scores = clf.decision_function(data.iloc[:,2:])
    data2.insert(2+nb_param*nb_band, 'score', scores)
            
            
        # Second part : Create a df with all scores aligned
    
    if (split_band == 1) or (split_band == 2):
        shape_score = {'score':[], 'target':[], 'object_id':[]}
        score_df = pd.DataFrame(data=shape_score)  
        for i in band_used:
                score_nb = 'score'+str(i)
                score_df = pd.concat([score_df,data2.loc[:,[score_nb,'target','object_id']].rename(columns={score_nb: "score"})])
    else :
        score_df = pd.DataFrame({'score':data2['score'], 'target':data2['target'], 'object_id':data2['object_id']})

    return score_df