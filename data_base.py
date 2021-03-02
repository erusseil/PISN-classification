import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import os
import glob


def create(data,metadata,band_used,name,dff=True,extra=True,Dbool=False,mini=5,complete=True,addPISN=True):


    ##Base data
    #data = pd.read_csv("plasticc_train_lightcurves.csv")
    ##Associated metadata
    #metadata = pd.read_csv("plasticc_train_metadata.csv")
    
    if addPISN==True:
        #PISN extra data
        all_filenames = [i for i in glob.glob(f"PLAsTiCC_PISN/PISN/*{'.csv'}")]
        PISN = pd.concat([pd.read_csv(f) for f in all_filenames])
        PISN=PISN.drop("Unnamed: 0",axis=1)
        PISN['target']=994

    #Conditions on the deep drilling field and the redshift
    isDDF = metadata['ddf_bool']==1
    isExtra = metadata['hostgal_specz']>0
    

    #We filter the initial metadata

    if (dff==True) & (extra==True):
        metadata2=metadata.loc[isDDF&isExtra]
    elif (dff==True) & (extra==False):
        metadata2=metadata.loc[isDDF]
    elif (dff==False) & (extra==True):
        metadata2=metadata.loc[isExtra]
    elif (dff==False) & (extra==False):
        metadata2=metadata

    meta_tofuse=metadata2.loc[:,['object_id','target']]

    # Then we fuse the metadata target column using the mutual ids 
    data_filtered = pd.merge(data, meta_tofuse, on="object_id")

    if addPISN==True:
        #We add the PISN data to obtain our training sample
        train=pd.concat([PISN,data_filtered])
    else:
        train=data_filtered

    
        #Filter the passband right away
    to_fuse=[]
    for i in band_used:
        to_fuse.append(train.loc[train['passband']==i])
    
    fused=to_fuse[0]
    for i in range (len(band_used)-1):
        fused=pd.concat([fused,to_fuse[i+1]])
    train=fused

        
    #List of all objects in the training sample
    objects = np.unique(train['object_id'])

    target_types = np.hstack([np.unique(train['target']), [999]])

    #For each object we normalize the mjd
    for i in objects:
        for j in band_used:
            object_mjd=train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']
            train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']= object_mjd-object_mjd.min()

    if Dbool==True:
        train = train[train['detected_bool']==1]


    if complete==True:
    #mini #Nombre minimum de points pour une passband

        objects = np.unique(train['object_id'])

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