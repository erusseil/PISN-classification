import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
from scipy.optimize import least_squares
from models import *


def parametrise(clean, nb_param, band_used, guess, err, save, checkpoint='', begin=0,
               cost=True, maxi=True, nb_points=True, nb_DB=True, peaktable=''):
     
    """Find best fit parameters for the polynomial model.
    
    Parameters
    ----------
    clean: pd.DataFrame
        Lightcurves dataframe to parametrize.
    nb_param: int 
        Number of parameters in your model.
    band_used: list
       List of all the passband you want to keep, using PLAsTiCC
       zenodo designations [0,1,2,3,4,5].
    guess: np.array
        Initial guess for parameters values. 
        [1, 0, 1, 30, -5] is good for Bazin.
    err: func 
        The err function associated with your model
    checkpoint : str 
        The table is saved each time a ligne is calculated, 
        if a problem occured you can put the partially filled 
        table as a check point. With the right 'begin' it 
        avoids recalculating from start.
    begin : int
        Start table construction at this line 
    save: str
        location and name of the save
    cost: bool (optional) 
        If True, add mean value the cost function
        as a parameter for each passband. Default is True.
    maxi: bool (optional) 
        If True, add maximum peak flux value
        as a parameter for each passband. Default is True.
    nb_points: bool (optional) 
        If True, add the number of points used for the fit
        as a parameter for each passband. Default is True.
    nb_DB: bool (optional) 
        If True, add the number of points with DB = True used for the fit
        for each passband. Default is True.
    peaktable: pd.DataFrame
        Two columns data frame : Maximum flux of the lightcurves 
        before normalisation along with the corresponding ids
        If maxi is false, no need to provide a table.
        Defaut is ''.

    """
    np.seterr(all='ignore')
    
    
    if maxi == True:     #Let's add the maximum flux as a column if needed
        clean = pd.merge(peaktable, clean, on=["object_id","passband"])
       
     
    #Get the number of passband used
    nb_passband = len(band_used)
    
    # Get ids of the objects
    objects = np.unique(clean['object_id'])
    
    # Get targets of the object
    target_list = np.array(clean.pivot_table(columns="object_id", values="target"))[0]
    
    
    #####################################################################################
    
    # COUNT THE NUMBER OF EXTRA PARAMETER TO ADD   
    
    ##################################################################################### 
    
    extra_param = 0
    
    if cost == True :
        extra_param += 1
        
    if maxi == True :
        extra_param += 1    
        
    if nb_points == True :
        extra_param += 1    
        
    if nb_DB == True :
        extra_param += 1    
     
    #####################################################################################
    
    # INITIALISE PARAMETER TABLE    
    
    #####################################################################################
    
    
    # We initialise a table of correct size but filled with 0    
    if checkpoint == '' :               

        df = {'object_id': objects, 'target': target_list}
        
        table = pd.DataFrame(data=df)    
        for i in range(nb_passband*(nb_param+extra_param)):
            table[i]=0
            
    # Or we start from checkpoint if provided
    else:
        table = checkpoint
        
    #####################################################################################
    
    # CONSTRUCT TABLE
    
    #####################################################################################  
    
    start = timeit.default_timer()
        
    print("Number of objects to parametrize : ",len(objects)-begin,"\n")
    
    for j in range(begin,len(objects)):

        print(j, end="\r")
        
        ligne=[]
        ide=objects[j]

        for i in band_used:
            
            obj = clean.loc[(clean['object_id'] == ide) & (clean['passband'] == i)]
            flux = np.array(obj['flux'])
            time = np.array(obj['mjd'])

            if err == errfunc_bazin:
                t0 = time[np.argmax(flux)]
                a0 = flux.max()
                guess[2], guess[0] = t0, a0

            fit = least_squares(err, guess, args=(time, flux))
            ligne = np.append(ligne,fit.x)

            
            # Add extra parameters 
            
            if cost==True:
                ligne = np.append(ligne,fit.cost)
        
            if nb_points==True:
                ligne = np.append(ligne,len(time))
                
            if maxi == True:
                band_max = obj.loc[:,0].iloc[0]
                ligne = np.append(ligne,band_max)
                
            if nb_DB == True :
                ligne = np.append(ligne,obj['detected_bool'].sum())
                    
        table.loc[j,0:] = np.array(ligne).flatten()
        table.to_pickle(save) 
        
    stop = timeit.default_timer()
    
    print('Total time of the parametrisation %.1f sec'%(stop - start)) 
    
    return table                      
                  
    table.to_pickle(save)          
    table    
        
        
  