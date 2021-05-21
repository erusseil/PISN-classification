import numpy as np
import pandas as pd
import random
from numpy import random


def split(database,size=365):
     """
    Divide each object randomly into two smaller lightcurve. 
    Effectively doubling the number of objects in a data base
     
    Parameters
    ----------
    database: pd.DataFrame 
        The light curve data from PLAsTiCC zenodo files.
    size: int64
        The mjd size of the created sub-objects. Default is 365 

        
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
    lower_mjd = data_tosplit[((data_tosplit['mjd']<=data_tosplit['mjd_split'])&(data_tosplit['mjd']>=data_tosplit['mjd_split']-size))]
    upper_mjd = data_tosplit[((data_tosplit['mjd']>data_tosplit['mjd_split'])&(data_tosplit['mjd']<=data_tosplit['mjd_split']+size))]

    # We add 1 or 2 at the end of object_id to distinguish them
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id'].astype('string');
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id']+'1';
    lower_mjd.loc[:,'object_id'] = lower_mjd.loc[:,'object_id'].astype('int');

    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id'].astype('string');
    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id']+'2';
    upper_mjd.loc[:,'object_id'] = upper_mjd.loc[:,'object_id'].astype('int');
    
    final_df = pd.concat([lower_mjd,upper_mjd],ignore_index=True)
    final_df = final_df.drop('mjd_split',axis = 1)

    return final_df