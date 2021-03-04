import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import os
import glob


def create(data,metadata,band_used,name,addPISN=True,dff=True,extra=True,Dbool=False,complete=True,mini=5,totrain=True):

'''
addPISN : add pair instability supernovae to the database
data : the light curve data frame
metadata : the corresponding meta data frame
band : array like of all the passband you want to keep (ex/ [0,1,2,3,4,5] is to keep them all)
name : name given to the saved .pkl file at the end
dff : only deep drilling field ?
extra : only extra galactic objects ?
Dbool : only detected boolean ?
complete : keep only objects that have a minimum of 'mini' points in each chosen passband. 
mini : minimum number of points in a passband (only the one chose in 'band') to be consider exploitable
totrain : are you creating a training data sample ? (include or not the target column)
'''
    
    #Add PISN extra data
    if addPISN==True:
        
        all_filenames = [i for i in glob.glob(f"PLAsTiCC_PISN/PISN/*{'.csv'}")]
        PISN = pd.concat([pd.read_csv(f) for f in all_filenames])
        PISN=PISN.drop("Unnamed: 0",axis=1)
        PISN['target']=994

    #Conditions on the deep drilling field and the redshift
    isDDF = metadata['ddf_bool']==1
    isExtra = metadata['hostgal_specz']>0
    

    #We filter the initial metadata
    if (dff==True):
        metadata = metadata.loc[isDDF]

    if (extra==True):
        metadata=metadata.loc[isExtra]

    # Keep only 2 columns before fusing
    if totrain==True:
        metadata=metadata.loc[:,['object_id','target']]
    else :
        meta_tofuse=metadata.loc[:,['object_id']]    
        
    # Then we fuse the metadata target column using the mutual ids 
    data_filtered = pd.merge(data, metadata, on="object_id")

    #We add the PISN data to obtain our training sample
    if addPISN==True:
        train=pd.concat([PISN,data_filtered])
    else:
        train=data_filtered

    
    #Filter the passband
        
    to_fuse=[]
    for i in band_used:
        to_fuse.append(train.loc[train['passband']==i])
        
    train=pd.concat(to_fuse)
        
    # Filter the detected boolean    
    if Dbool==True:
        train = train[train['detected_bool']==1]
        
    #List of all objects in the training sample
    objects = np.unique(train['object_id'])

    
    #For each object we normalize the mjd
    for i in objects:
        for j in band_used:
            object_mjd=train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']
            train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']= object_mjd-object_mjd.min()

    # Filter only complete objects
    if complete==True:

        objects_complet=[]
        for i in objects:
            a = train.loc[train['object_id']==i]
            bandOK=0
            for j in band_used:
                nb_pts=(a['passband']==j).sum()
                if nb_pts>=mini:
                    bandOK+=1

            if bandOK==len(band_used):
                objects_complet.append(i)

        isComplete=[]
        for i in range(len(train['object_id'])):
            isComplete.append(train.iloc[i]['object_id'] in objects_complet)

        train=train[isComplete]




    train.to_pickle("%s.pkl"%name)