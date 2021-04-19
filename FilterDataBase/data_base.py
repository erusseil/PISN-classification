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
           mini=5, mjd_tozero=True, half=True, 
           norm_flux=True, metatest=''):
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
        If True, use only DDF objects, else only non DDF.
        Default is True.
    extra: bool (optional) 
        If True, use only ext122618819ra galactic objects. Default is True.
    Dbool: bool (optional) 
        If True, use only epochs with detected_bool == 1.
        Default is True. 
    complete: bool (optional)
        If  True, keep only objects that have a minimum of 
        'mini' points in each chosen passband. Default is True.   
    mini: int (optional)
        Minimum number of points required in a passband 
        so the objects is considered exploitable. Default is 5.
    mjd_tozero: bool (optional) 
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
        Default is ''
    norm_flux: bool (optional) 
        If True, divide the 'flux' by the maximum flux of the considered
        passband. Default is True.
        
    Returns
    -------
    
    """
 
    f = open("%s.txt"%name, "w") # Clear the previous print save
    f.close()
    f = open("%s.txt"%name, "a") # We will save the print in a txt file
    
    if training == True :
        print("\n\n CREATION OF THE TRAINING DATA BASE\n ", file=f)
        print("\n\n CREATION OF THE TRAINING DATA BASE\n ")
    else :
        print("\n\n CREATION OF THE TESTING DATA BASE\n ", file=f)
        print("\n\n CREATION OF THE TESTING DATA BASE\n ")
        
    
    data = pd.merge(data, metadata.loc[:,['object_id','true_target']], on="object_id") 
    data = data.rename(columns={"true_target": "target"})
    
    PISN_nb = len(np.unique(data.loc[data['target']==994,'object_id']))

    print('We start with  %s objects and %s measurements'%(len(np.unique(data['object_id'])),len(data)), file=f)
    print('We start with  %s objects and %s measurements'%(len(np.unique(data['object_id'])),len(data)))
    print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
    print('--> There are ',PISN_nb,'PISN in the dataset\n')
    
    metadata = metadata[np.in1d(metadata['object_id'],data['object_id'])] #In metadata, keep only objects that exist in data
      
    #------------------------------------------------------------------------------------------------------------------
    
    # Add the PISN data to obtain our training sample
    
    #------------------------------------------------------------------------------------------------------------------
    
    if ratioPISN == -1:             # if -1 we add all to training and let the testing as is
        if training == True:           
            data = pd.concat([PISNdf, data],ignore_index=True)   #We fuse the original dataset with PISN dataset
            
            PISN_nb = len(np.unique(data.loc[data['target']==994,'object_id']))
            objects = np.unique(data['object_id'])
            
            print('After we add PISN we have %s objects and %s measurements'%(len(objects),len(data)))
            print('After we add PISN we have %s objects and %s measurements'%(len(objects),len(data)), file=f)
            print('--> There are ',PISN_nb,'PISN in the dataset\n')
            print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
            
        metaPISN = metatest[np.in1d(metatest['object_id'],PISNdf['object_id'])]
       
        
    elif (0 < ratioPISN <= 1):       # if 0<ratioPISN<1 we add ratioPISN to training or 1-ratioPISN to testing
        
        obj_PISN = (np.unique(PISNdf['object_id']))
        PISN_split = train_test_split(obj_PISN, test_size=ratioPISN, random_state=1)  

        if training==True:
            PISNdf_split = pd.DataFrame(data={'object_id': PISN_split[1]})
            
            
        else:
            data = data[data['target'] != 994]
            PISNdf_split = pd.DataFrame(data = {'object_id': PISN_split[0]})
            
         
        PISNdf = PISNdf[np.in1d(PISNdf['object_id'],PISNdf_split['object_id'])] #Keep only the random subsample
        metaPISN = metatest[np.in1d(metatest['object_id'],PISNdf['object_id'])] # We get metadata for the added PISN
        
        data = pd.concat([PISNdf,data],ignore_index=True)  #We fuse the original dataset with PISN dataset
        
        PISN_nb = len(np.unique(data.loc[data['target']==994,'object_id']))
        objects = np.unique(data['object_id'])

        print('After we add/remove PISN we have %s objects and %s measurements'%(len(objects),len(data)))
        print('After we add/remove PISN we have %s objects and %s measurements'%(len(objects),len(data)), file=f)
        print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
        print('--> There are ',PISN_nb,'PISN in the dataset\n')
        
    elif (ratioPISN == -2): # Does nothing, ignore PISN
        print('PISN ignored\n', file=f)
        print('PISN ignored\n')
    else:
        print('ERROR RATIO PISN VALUE\n', file=f)
        print('ERROR RATIO PISN VALUE\n')
  

    if ratioPISN!=-2:
        
        #Let's keep only the 5 intersting columns for the metadata
        
        metadata = metadata.loc[:, ['object_id','true_target','ddf_bool','true_z','true_peakmjd']]
        metaPISN = metaPISN.loc[:, ['object_id','true_target','ddf_bool','true_z','true_peakmjd']]
        metadata = pd.concat([metaPISN, metadata],ignore_index=True)   #We fuse the original metadata with the PISN metadata
        
    #------------------------------------------------------------------------------------------------------------------
            
    #Filter on the deep drilling field and the redshift
    
    #------------------------------------------------------------------------------------------------------------------
    
    isDDF = metadata['ddf_bool'] == 1
    isnotDDF = metadata['ddf_bool'] == 0
    isExtra = metadata['true_z'] > 0
    isnotExtra = metadata['true_z'] == 0
    
    
    #We filter the initial metadata
    
    #Start with DDF objects :
    if (ddf == True):
        metadata = metadata.loc[isDDF]
    else :
        metadata = metadata.loc[isnotDDF]
    
    #Then extragalactic objects :
    if (extra == True):
        metadata = metadata.loc[isExtra]
    else :
        metadata = metadata.loc[isnotExtra]   
        
    # Then we keep only objects that exist in the metadata
    clean = data[np.in1d(data['object_id'],metadata['object_id'])]
    objects = np.unique(clean['object_id'])
    PISN_nb = len(np.unique(clean.loc[clean['target']==994,'object_id']))
    
    print('After EXTRA-GALACTIC and DDF we have %s objects and %s measurements'%(len(objects),len(clean)))
    print('After EXTRA-GALACTIC and DDF we have %s objects and %s measurements'%(len(objects),len(clean)), file=f)
    
    print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
    print('--> There are ',PISN_nb,'PISN in the dataset\n')
           
    #List of all objects/peaks in the clean sample
    objects = np.unique(clean['object_id'])
    
    #------------------------------------------------------------------------------------------------------------------
    
    # Take only points before true peak
  
    #------------------------------------------------------------------------------------------------------------------
    
    # We get all the peak values
    peaklist=metadata.loc[:,['true_peakmjd','object_id']]
 
    if half==True:
        start = timeit.default_timer()
        clean = pd.merge(clean, peaklist, on="object_id")
        clean['true_peakmjd'] = clean['mjd'] - clean['true_peakmjd']
        clean = clean[clean['true_peakmjd'] < 0]
        clean = clean.drop(['true_peakmjd'], axis=1)     
 
        objects = np.unique(clean['object_id'])
        PISN_nb = len(np.unique(clean.loc[clean['target']==994,'object_id']))
        
        print('After TRUE_PEAK we have %s objects and %s measurements'%(len(objects),len(clean)), file=f)
        print('--> There are ',PISN_nb,'PISN in the dataset', file=f)
        print('After TRUE_PEAK we have %s objects and %s measurements'%(len(objects),len(clean)))
        print('--> There are ',PISN_nb,'PISN in the dataset')
        stop = timeit.default_timer()
        print('Total time to select points before true peak %.1f sec\n'%(stop - start), file=f)
        print('Total time to select points before true peak %.1f sec\n'%(stop - start)) 
        
    #------------------------------------------------------------------------------------------------------------------
    
    #Filter the passband
     
    #------------------------------------------------------------------------------------------------------------------
    
    to_fuse=[]
    for i in band_used:
        to_fuse.append(clean.loc[clean['passband']==i])
        
    clean = pd.concat(to_fuse)
    objects = np.unique(clean['object_id'])
    PISN_nb = len(np.unique(clean.loc[clean['target']==994,'object_id']))
    
    print('After PASSBANDS we have %s objects and %s measurements'%(len(objects),len(clean)), file=f)
    print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
    print('After PASSBANDS we have %s objects and %s measurements'%(len(objects),len(clean)))
    print('--> There are ',PISN_nb,'PISN in the dataset\n')
    
    #------------------------------------------------------------------------------------------------------------------
    
    # Filter the detected boolean   
    
    #------------------------------------------------------------------------------------------------------------------
    
    if Dbool==True:
        
        clean = clean[clean['detected_bool']==1]
        objects = np.unique(clean['object_id'])
        PISN_nb=len(np.unique(clean.loc[clean['target']==994,'object_id']))
        
        print('After DDB we have %s objects and %s measurements'%(len(objects),len(clean)), file=f)
        print('--> There are ',PISN_nb,'PISN in the dataset\n', file=f)
        print('After DDB we have %s objects and %s measurements'%(len(objects),len(clean)))
        print('--> There are ',PISN_nb,'PISN in the dataset\n')
        
    #------------------------------------------------------------------------------------------------------------------    
    
    # Translate the mjd
    
    #------------------------------------------------------------------------------------------------------------------ 
    
    objects = np.unique(clean['object_id'])
    
    if mjd_tozero == True:
        
        start = timeit.default_timer()

        mintable = clean.pivot_table(index="passband", columns="object_id", values="mjd",aggfunc='min')
        mindf = pd.DataFrame(data=mintable.unstack())
        clean = pd.merge(mindf, clean, on=["object_id","passband"])
        clean['mjd'] = clean['mjd']-clean[0]
        clean = clean.drop([0],axis=1)
    
        stop = timeit.default_timer()
        print('Total time to translate mjd %.1f sec\n'%(stop - start), file=f)
        print('Total time to translate mjd %.1f sec\n'%(stop - start)) 
        
    #------------------------------------------------------------------------------------------------------------------    
    
    # Normalise the flux
    
    #------------------------------------------------------------------------------------------------------------------ 
    
    objects = np.unique(clean['object_id'])
    
    if norm_flux == True:
        
        start = timeit.default_timer()

        maxtable = clean.pivot_table(index="passband", columns="object_id", values="flux",aggfunc='max')
        maxdf = pd.DataFrame(data=maxtable.unstack())
        clean = pd.merge(maxdf, clean, on=["object_id","passband"])
        clean['flux'] = clean['flux']/clean[0]
        clean['flux_err'] = clean['flux_err'] / clean[0]
        clean = clean.drop([0],axis=1)
    
        stop = timeit.default_timer()
        print('Total time to normalise flux %.1f sec\n'%(stop - start), file=f)
        print('Total time to normalise flux %.1f sec\n'%(stop - start)) 
        
        maxdf.to_pickle("%s_maxdf.pkl"%name)
    
    #------------------------------------------------------------------------------------------------------------------    
    
    # Filter only objects with the required minimum number of epochs per passband
    
    #------------------------------------------------------------------------------------------------------------------
    
    if complete==True:
     
        start = timeit.default_timer()

        # Create a table describing how many points exist for each bands and each object   
        counttable = clean.pivot_table(index="passband", columns="object_id", values="mjd",aggfunc=lambda x: len(x))

        # Create a table describing how many bands are complete for each object
        df_validband = pd.DataFrame(data={'nb_valid' : (counttable>=mini).sum()})

        clean = pd.merge(df_validband,clean, on=["object_id"])
        clean = clean[clean['nb_valid']==len(band_used)]
        clean = clean.drop(['nb_valid'],axis=1)
    
        stop = timeit.default_timer()
        
        objects = np.unique(clean['object_id'])
        PISN_nb=len(np.unique(clean.loc[clean['target']==994,'object_id']))
        
        print('Total time to check completness %.1f sec\n'%(stop - start), file=f) 
        print('After COMPLETNESS we are left with %s objects and %s measurements'%(len(objects),len(clean)), file=f)
        print('--> There are ',PISN_nb,'PISN in the dataset', file=f)
        print('Total time to check completness %.1f sec\n'%(stop - start)) 
        print('After COMPLETNESS we are left with %s objects and %s measurements'%(len(objects),len(clean)))
        print('--> There are ',PISN_nb,'PISN in the dataset')

    f.close()
    clean.to_pickle("%s.pkl"%name)

        
        
        
        
