import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import os
import glob


def create(data,metadata,band_used,name,PISNdf='',addPISN=True,dff=True,extra=True,Dbool=False,complete=True,mini=5,norm=True):
    

#addPISN : add pair instability supernovae to the database
#PISNfile : fused PISN data frame. If addPISN is false, you can ignore this argument
#data : the light curve data frame
#metadata : the corresponding meta data frame
#band : array like of all the passband you want to keep (ex/ [0,1,2,3,4,5] is to keep them all)
#name : name given to the saved .pkl file at the end
#dff : only deep drilling field ?
#extra : only extra galactic objects ?
#Dbool : only detected boolean ?
#complete : keep only objects that have a minimum of 'mini' points in each chosen passband. 
#mini : minimum number of points in a passband (only the one chose in 'band') to be consider exploitable
#norm : normalise mjd ?


    print('We start with  %s objects and %s mesures'%(len(np.unique(data['object_id'])),len(data)))
    
      
    #Conditions on the deep drilling field and the redshift
    isDDF = metadata['ddf_bool']==1
    isExtra = metadata['hostgal_specz']>0
    

    #We filter the initial metadata
    if (dff==True):
        metadata = metadata.loc[isDDF]

    if (extra==True):
        metadata=metadata.loc[isExtra]

    # Keep only 2 columns before fusing
    metadata=metadata.loc[:,['object_id','true_target']]
    metadata=metadata.rename(columns={"true_target": "target"})
        
    # Then we fuse the metadata target column using the mutual ids 
    train = pd.merge(data, metadata, on="object_id")

    print('After EXTRA-GALACTIC and DDF we have %s objects and %s mesures'%(len(np.unique(train['object_id'])),len(train)))
    
    #We add the PISN data to obtain our training sample
    if addPISN==True:
        train=pd.concat([PISNdf,train])
        print('After we add PISN we have %s objects and %s mesures'%(len(np.unique(train['object_id'])),len(train)))

    
    #Filter the passband
        
    to_fuse=[]
    for i in band_used:
        to_fuse.append(train.loc[train['passband']==i])
        
    train=pd.concat(to_fuse)
    print('After PASSBANDS we have %s objects and %s mesures'%(len(np.unique(train['object_id'])),len(train)))    
        
    # Filter the detected boolean    
    if Dbool==True:
        train = train[train['detected_bool']==1]
        print('After DDB we have %s objects and %s mesures'%(len(np.unique(train['object_id'])),len(train)))
        
    #List of all objects in the training sample
    objects = np.unique(train['object_id'])

    if norm ==True:
        #For each object we normalize the mjd
        start = timeit.default_timer()
        for i in objects:
            for j in band_used:
                object_mjd=train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']
                train.loc[(train['object_id']==i)&(train['passband']==j),'mjd']= object_mjd-object_mjd.min()

        stop = timeit.default_timer()
        print('Total time to normalise mjd %.1f sec'%(stop - start)) 


    # Filter only complete objects
    if complete==True:
        
        start = timeit.default_timer()

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
        stop = timeit.default_timer()
        print('Total time to check completness %.1f sec'%(stop - start)) 
        print('After COMPLETNESS we are left with %s objects and %s mesures'%(len(np.unique(train['object_id'])),len(train)))


    train.to_pickle("%s.pkl"%name)