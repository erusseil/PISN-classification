import numpy as np
import pandas as pd
import random
from numpy import random


def split(database,mini,size=365,band_used=[0,1,2,3,4,5]):
    """
    Divide each object randomly into two smaller lightcurve.
    Then keep only objects with at least one passband with more
    than a minimum number of detected_bool=1 points
    Effectively doubling the number of objects in a data base

    Parameters
    ----------
    database: pd.DataFrame 
        The light curve data from PLAsTiCC zenodo files.
    size: int64
        The mjd size of the created sub-objects. Default is 365 
    band_used: list
       List of all the passband you want to keep, using PLAsTiCC
       zenodo designations [0,1,2,3,4,5].
       Default uses all 6 bands : [0,1,2,3,4,5]
    mini: int (optional)
        Minimum number of points required in a passband 
        so the objects is considered exploitable

    Returns
    -------
    final_df: pd.DataFrame 
        Create df with twice the amount of objects.

    """
    # We create a data frame containing only values that can be the split point 
    interval = ((database['mjd'].min()+size) <= (database['mjd'])) & ((database['mjd']) <= (database['mjd'].max()-size))
    allowed_values = database[interval]
    
    # We random a float between 0 and 1. We multiply this number by the number of allowed points for each object.
    # We obtain a position that we will use to determine the corresponding mjd for each object
    df_count = allowed_values.pivot_table(index="object_id", values="mjd",aggfunc=['count'])
    arr_rand = random.rand(len(df_count))
    df_count['nb_points'] = df_count['count']
    df_count['rand_mult'] = arr_rand
    df_count['rand_pos'] =  np.round(df_count['rand_mult']*(df_count['nb_points']))
    df_count['cum_pos'] = df_count['nb_points'].cumsum()+df_count['rand_pos']-df_count['nb_points']-1
    
    
    centers = allowed_values.iloc[df_count['cum_pos']].loc[:,['mjd','object_id']]
    centers = centers.rename(columns={"mjd": "mjd_split"})
    data_tosplit = pd.merge(database, centers, on=["object_id"])
    lower_mjd = data_tosplit[((data_tosplit['mjd']<=data_tosplit['mjd_split'])&(data_tosplit['mjd']>=data_tosplit['mjd_split']-size))].copy()
    upper_mjd = data_tosplit[((data_tosplit['mjd']>data_tosplit['mjd_split'])&(data_tosplit['mjd']<=data_tosplit['mjd_split']+size))].copy()

    
    # We add 1 or 2 at the end of object_id to distinguish them
    
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id'].astype('string');
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id']+'1';
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id'].astype('int');

    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id'].astype('string');
    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id']+'2';
    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id'].astype('int');
    
    
    #Fuse both datasets and get rid of the extra column
    
    new_df = pd.concat([lower_mjd,upper_mjd],ignore_index=True)
    new_df = new_df.drop('mjd_split',axis = 1)
    
    # Look only at the passbands we are interested in
    
    to_fuse=[]
    for i in band_used:
        to_fuse.append(new_df.loc[new_df['passband']==i])

    final_df = pd.concat(to_fuse)
    
    # Count the number of passband with at least "mini" points for each object
    
    count_bool = final_df[final_df['detected_bool']==1].pivot_table(index="passband", columns="object_id", values="mjd",aggfunc=lambda x: len(x))
    df_validband = pd.DataFrame(data={'nb_valid' : (count_bool>=mini).sum()})

    # Keep objects with at least 1 passband that satisfies the previous condition
    
    final_df = pd.merge(df_validband,final_df, on=["object_id"])
    final_df = final_df[final_df['nb_valid']!=0]
    final_df = final_df.drop(['nb_valid'],axis=1)

    return final_df


def split_meta(meta):
    
    """
    Double each object in metadata file and modify their ID

    Parameters
    ----------
    meta: pd.DataFrame 
        The metadata from PLAsTiCC zenodo files.


    Returns
    -------
    final_df: pd.DataFrame 
        Create df with twice the amount of objects.

    """
    
    lower_data = meta.copy()
    upper_data = meta.copy()
    
    lower_data.loc[:,'object_id'] = lower_data.loc[:,'object_id'].astype('string');
    lower_data.loc[:,'object_id'] = lower_data.loc[:,'object_id']+'1';
    lower_data.loc[:,'object_id'] = lower_data.loc[:,'object_id'].astype('int');

    upper_data.loc[:,'object_id'] = upper_data.loc[:,'object_id'].astype('string');
    upper_data.loc[:,'object_id'] = upper_data.loc[:,'object_id']+'2';
    upper_data.loc[:,'object_id'] = upper_data.loc[:,'object_id'].astype('int');
    
    final_df = pd.concat([lower_data,upper_data],ignore_index=True)
    final_df = final_df.drop(columns='Unnamed: 0')
    
    
    return final_df
    