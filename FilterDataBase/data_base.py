import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import os
import glob
from sklearn.model_selection import train_test_split


def create(data, metadata, band_used,
           name, PISNdf='', ratioPISN=-1,
           training=True, ddf=True,
           extra=True, Dbool=False, complete=True,
           mini=5, norm=True, half=True, metatest=''):
    """
    Construct training or test sample with required fraction of PISN.
    
    Parameters
    ----------
    PISNfile: pd.DataFrame
        Fused PISN data frame. Only used if ratioPISN != -2 
    data: pd.DataFrame 
        The light curve data from PLAsTiCC zenodo files.
    metadata: pd.DataFrame 
        The corresponding metadata from PLAsTiCC zenodo files.
    band_used: list
       List of all the passband you want to keep, using PLAsTiCC
       zenodo designations [0,1,2,3,4,5].
    name: str
        Name for output pickle file (.pkl is added automatically).
    ddf: bool (optional)
        If True, use only DDF objects. Default is True.
    extra: bool (optional) 
        If True, use only extra galactic objects. Default is True.
    Dbool: bool (optional) 
        If True, use only epochs with detected_bool == 1.
        Default is True. 
    complete: bool (optional)
        If  True, keep only objects that have a minimum of 
        'mini' points in each chosen passband. Default is True.   
    mini: int (optional)
        Minimum number of points required in a passband 
        so the objects is considered exploitable. Default is 5.
    norm: bool (True) 
        If True, normalise the 'mjd' by shifting to first observed
        epoch. Default is True.
    ratioPISN : float (optional)
        Fraction of PISN to be added to training 
        OR complementary fraction to be added to test (after we remove PISN).
        If -1, all PISN will be added to training sample and none 
        will be substracted from test. If -2, nothing will be done 
        (usable even if no PISN file are inputed). Default is -1.
    training: bool (optional)
        If True, add ratioPISN to the training sample, otherwise remove 
        all PISN and add 1-ratioPISN to the test sample. 
        Default is True.
    half: bool (optional)
        If True, keep only points before the peak. Default is True.
    metatest: pd.DataFrame (optional)
        metadata corresponding to the test sample from PLAsTiCC zenodo files.
        
        
    Returns
    -------
    
    """

    f = open("%s.txt"%name, "w") # Clear the previous print save
    f.close()
    f = open("%s.txt"%name, "a") # We will save the print in a txt file
    
    if training == True :
        print("\n CREATION OF THE TRAINING DATA BASE\n ", file=f)
        print("\n CREATION OF THE TRAINING DATA BASE\n ")
    else :
        print("\n CREATION OF THE TESTING DATA BASE\n ", file=f)
        print("\n CREATION OF THE TESTING DATA BASE\n ")
    

    print('We start with  %s objects and %s mesures'%(len(np.unique(data['object_id'])),len(data)), file=f)
    print('We start with  %s objects and %s mesures'%(len(np.unique(data['object_id'])),len(data)))
      
    
        
    #Conditions on the deep drilling field and the redshift
    isDDF = metadata['ddf_bool'] == 1
    isExtra = metadata['true_z'] > 0
    
    #We filter the initial metadata
    if (ddf == True):
        metadata = metadata.loc[isDDF]

    if (extra == True):
        metadata = metadata.loc[isExtra]

    peaklist=np.array(metadata['true_peakmjd'])
    print(peaklist)
    
    # Keep only 2 columns before fusing
    metadata = metadata.loc[:, ['object_id','true_target']]
    metadata = metadata.rename(columns={"true_target": "target"})
        
    # Then we fuse the metadata target column using the mutual ids 
    clean = pd.merge(data, metadata, on="object_id")

    print('\nAfter EXTRA-GALACTIC and DDF we have %s objects and %s mesures\n'%(len(np.unique(clean['object_id'])),len(clean)))
    print('\nAfter EXTRA-GALACTIC and DDF we have %s objects and %s mesures\n'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
    
    #We add the PISN data to obtain our training sample
    
    if ratioPISN == -1:             # if -1 we add all to training and let the testing as is
        if training == True:           
            clean = pd.concat([PISNdf, clean])
            
            print('After we add PISN we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
            print('After we add PISN we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
            print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n')
            print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n', file=f)
            
    elif (0 <= ratioPISN <= 1):       # if 0<ratioPISN<1 we add ratioPISN to training or 1-ratioPISN to testing
        
        obj_PISN = (np.unique(PISNdf['object_id']))
        
        if ratioPISN == 0:
            PISN_split = obj_PISN
        else :    
            PISN_split = clean_test_split(obj_PISN, test_size=ratioPISN, random_state=1)  
        
        if training==True:
            PISNdf_split = pd.DataFrame(data={'object_id': PISN_split[1]})
        else:
            clean = clean[clean['target'] != 994]
            PISNdf_split = pd.DataFrame(data = {'object_id': PISN_split[0]})
            
        PISNdf = pd.merge(PISNdf_split, PISNdf, on="object_id")
        clean = pd.concat([PISNdf,clean])

        print('After we add/remove PISN we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
        print('After we add/remove PISN we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n', file=f)
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n')
        
    elif (ratioPISN == -2): # Does nothing, ignore PISN
        print('PISN ignored\n', file=f)
        print('PISN ignored\n')
    else:
        print('ERROR RATIO PISN VALUE\n', file=f)
        print('ERROR RATIO PISN VALUE\n')
           
    #List of all objects in the clean sample
    objects = np.unique(clean['object_id'])
    
    # We take only points before true peak
    
    if half==True:
        start = timeit.default_timer()
        list01=np.array([])
        for i in range(len(objects)):
            print(i, end='\r')
            objdata=data[data['object_id']==objects[i]]
            list01 = np.append(list01,objdata['mjd']<peaklist[i])
            
        firsthalf=(list01 > 0).tolist()   #We have produced an array of 1 and 0, this converts it into boolean
        clean = clean[firsthalf]
        
        objects = np.unique(clean['object_id'])
        
        print('After TRUE_PEAK we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset', file=f)
        print('After TRUE_PEAK we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset')
        stop = timeit.default_timer()
        print('Total time to select points before true peak %.1f sec\n'%(stop - start), file=f)
        print('Total time to select points before true peak %.1f sec\n'%(stop - start)) 
    
    #Filter the passband
        
    to_fuse=[]
    for i in band_used:
        to_fuse.append(clean.loc[clean['passband']==i])
        
    clean=pd.concat(to_fuse)
    print('After PASSBANDS we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
    print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n', file=f)
    print('After PASSBANDS we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
    print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n')
        
    # Filter the detected boolean    
    if Dbool==True:
        clean = clean[clean['detected_bool']==1]
        print('After DDB we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n', file=f)
        print('After DDB we have %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset\n')
        
    objects = np.unique(clean['object_id'])

    if norm ==True:
        #For each object we normalize the mjd
        start = timeit.default_timer()
        print("Number of objects to normalize :",len(objects))
        
        for i in objects:
            print(np.where(objects==i)[0], end='\r')
            for j in band_used:
                object_mjd = clean.loc[(clean['object_id'] == i) & (clean['passband'] == j),'mjd']
                clean.loc[(clean['object_id'] == i) & (clean['passband'] == j),'mjd'] = object_mjd-object_mjd.min()

        stop = timeit.default_timer()
        print('Total time to normalise mjd %.1f sec\n'%(stop - start), file=f)
        print('Total time to normalise mjd %.1f sec\n'%(stop - start)) 


    # Filter only objects with the required minimum number of epochs per filter
    if complete==True:
        
        start = timeit.default_timer()

        objects_complet=[]
        print("Number of objects to check :",len(objects))
        for i in objects:
            print(np.where(objects==i)[0], end="\r")
            a = clean.loc[clean['object_id'] == i]
            bandOK = 0
            
            for j in band_used:
                nb_pts = (a['passband'] == j).sum()
                
                if nb_pts >= mini:
                    bandOK += 1

            if bandOK==len(band_used):
                objects_complet.append(i)

        isComplete=[]
        for i in range(len(clean['object_id'])):
            isComplete.append(clean.iloc[i]['object_id'] in objects_complet)

        clean=clean[isComplete]
        stop = timeit.default_timer()
        print('Total time to check completness %.1f sec\n'%(stop - start), file=f) 
        print('After COMPLETNESS we are left with %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)), file=f)
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset', file=f)
        print('Total time to check completness %.1f sec\n'%(stop - start)) 
        print('After COMPLETNESS we are left with %s objects and %s mesures'%(len(np.unique(clean['object_id'])),len(clean)))
        print('--> There are ',len(np.unique(clean.loc[clean['target']==994,'object_id'])),'PISN in the dataset')

    f.close()
    clean.to_pickle("%s.pkl"%name)
