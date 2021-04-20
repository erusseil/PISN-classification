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
    
    
    
def create_if(data,band_used,nb_param, ntrees, split_band=False):
    
    """Perform anomaly detection using isolation forest
    
    Parameters
    ----------
    data: pd.DataFrame
        Table containing all parameters for each object
    band_used: np.array
        Array of the passbands chosen
    ntrees: int
        Number of trees to use for the isolation forest
    split_band: boolean
        If true perform isolation forest for each band
        independently. Default is False
    ----------
        
    
    Returns
    ----------
    data2: pd.Dataframe
        Copy of original table with added anomaly score columns
    score_df: pd.Dataframe
        Additionnal dataframe containing all the scores on
        a single column
    ----------
    """

        
    # Define useful values
    width = np.shape(data)[1]
    nb_band = len(band_used)
    shift = 6 - nb_band # Needs consecutive bands to work. Is equal to the minimal band
  
    data2 = data.copy()

    reshaped = np.array(data.iloc[:, 2 : 2 + nb_param])
    
    for i in range(nb_band-1) :
        reshaped = np.concatenate([reshaped,data.iloc[:, 2+nb_param*(i+1) : 2 + nb_param*(i+2)]])
       

    # First part : apply the isolation forest and add a column to a copy of the initial df    

    if split_band == True:
                      
        """ 
        iso = []
        for i in range (nb_band) :
            iso = data.iloc[:, 2+nb_param*(i) : 2 + nb_param*(i+1)]
            clf = IsolationForest(n_estimators = ntrees).fit(iso)
            data2.insert(i+2+nb_param*(i+1), 'score'+str(i+shift), clf.decision_function(iso))
                    
        """
               
        for i in range (nb_band) :
                 
            clf = IsolationForest(n_estimators = ntrees).fit(reshaped[i*len(data):(i+1)*len(data)])
            scores = clf.decision_function(reshaped[i*len(data):(i+1)*len(data)])
            data2.insert(i+2+nb_param*(i+1), 'score'+str(i+shift), scores)
            print('score'+str(i+shift)+" : OK")         
        
        
    else :

        clf = IsolationForest(n_estimators = ntrees).fit(reshaped)
        scores = clf.decision_function(reshaped)
        
        for i in range (nb_band) :
            single_band = scores[i*len(data):(i+1)*len(data)]
            data2.insert(i+2+nb_param*(i+1), 'score'+str(i+shift), single_band)
        print("All bands : OK")
            
            
    # Second part : Create a df with all scores aligned
    
    shape_score = {'score':[], 'target':[], 'object_id':[]}
    score_df = pd.DataFrame(data=shape_score)  
    
    for i in band_used:
            score_nb = 'score'+str(i)
            score_df = pd.concat([score_df,data2.loc[:,[score_nb,'target','object_id']].rename(columns={score_nb: "score"})])
            
    return data2,score_df