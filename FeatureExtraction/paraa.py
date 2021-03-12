import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
from scipy.optimize import least_squares


def fit_scipy(time, flx, err_model,guess):
    """
    Find best-fit parameters using scipy.least_squares.
    Parameters
    ----------
    time: np.array
        exploratory variable (time of observation)
    flx: np.array
        response variable (measured flux)
    err_model:  func 
        The err function associated with your model
    guess: np.array
        Initial guess for parameters values. 
        [1, 0, 1, 30, -5] is good for Bazin.
     
    Returns
    -------
    output : list of float
        best fit parameter values
    """
    
    flux = np.asarray(flx)
    result = least_squares(err_model, guess, args=(time, flux))
    
    return result.x


def get_param(train, ide, err_model, guess, band_used):
    """
    Creates an array of parameters for each object.
    
    If the model used is bazin it automaticaly chooses t0 to be 
    t_max and A to be flux_max if there are not at least 2 points, 
    it puts NaN instead of the parameters
    
    Parameters
    ----------
    train: pd.DataFrame
        Lightcurves dataframe to parametrize.
    ide: int
        'Object_id' of the object to parametrize 
    err_model: func 
        The err function associated with your model
    guess: np.array
        Initial guess for parameters values. 
        [1, 0, 1, 30, -5] is good for Bazin.
    band_used: list
       List of all the passband you want to keep, using PLAsTiCC
       zenodo designations [0,1,2,3,4,5].
        
    Returns
    -------
    array of parameter values.
    """
    
    ligne=[]
    error=[]
    
    for i in range(len(guess)):
        error.append(float("nan"))
            
    for i in band_used:
        obj = train.loc[(train['object_id'] ==  ide) & (train['passband'] == i)]
        flux = np.array(obj['flux'])
        time = np.array(obj['mjd'])
        
        if err_model == errfunc_bazin:
            t0 = time[np.argmax(flux)]
            a0 = flux.max()
            guess[2], guess[0] = t0, a0
        
        if len(time)<=1:
            ligne.append(np.array(error))
        else:
            ligne.append(fit_scipy(time, flux,err_model,guess))
        
    return np.array(ligne).flatten()


def create_table(objects,train,err_model,guess,table,begin,band_used,save):
    """Create the data frame containing all the parameters.
    
    Parameters
    ----------
    objects: list
        List of objects ids.
    train: pd.DataFrame
        Lightcurves dataframe to parametrize.
    err_model: func
        The err function associated with your model.
    guess: np.array
        Initial guess for parameters values. 
        [1, 0, 1, 30, -5] is good for Bazin.
    table: pd.DataFrame
        Correctly sized table to which we write the parameters
    begin: int
        Start table construction at this line 
    band_used: list
       List of all the passband you want to keep, using PLAsTiCC
       zenodo designations [0,1,2,3,4,5].
    save: str
        location and name of the save
        
    Returns
    -------
    table: XX
        XXXX
    """
    
    start = timeit.default_timer()
        
    print("Number of objects to parametrize : ",len(objects)-begin,"\n")
    for i in range(begin,len(objects)):

        print(i, end="\r")
        
        table.loc[i,0:] = get_param(train, objects[i], err_model, 
                                    guess, band_used)
        table.to_pickle(save) 
        
    stop = timeit.default_timer()
    
    print('Total time of the parametrisation %.1f sec'%(stop - start)) 
    
    return table


def bazin(time, a, b, t0, tfall, trise):
    """
    Parametric light curve function proposed by Bazin et al., 2009.
    
    Parameters
    ----------
    time : np.array
        exploratory variable (time of observation)
    a: float
        Normalization parameter
    b: float
        Shift parameter
    t0: float
        Time of maximum
    tfall: float
        Characteristic decline time
    trise: float
        Characteristic raise time
        
    Returns
    -------
    array_like
        response variable (flux)
    """

    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    
    return a * X + b
    

def errfunc_bazin(params,time, flux):
    """
    Absolute difference between theoretical and measured flux.
    
    Parameters
    ----------
    params : list of float
        light curve parameters: (a, b, t0, tfall, trise)
    time : array_like
        exploratory variable (time of observation)
    flux : array_like
        response variable (measured flux)
        
    Returns
    -------
    diff : float
        absolute difference between theoretical and observed flux
    """

    return abs(flux - bazin(time,*params))


def poly(time,a,b,c):
    """Polynomial quadractic function of degree.
    
    Parameters
    ----------
    time: float
       Independent variable.
    a: float
    	Quadratic parameter.
    b: float
       Linear parameter.
    c: float
        intercept
        
    Returns
    -------
    float
        Value of function at time.       
    """
    
    return c + b * time + a * time ** 2


def errfunc_poly(params, time, flux):
    """Absolute difference between polynomial fit and measured flux.
    
    Parameters
    -----------
    params: np.array
        List of polynomial fit parameters.
    time: float
        Time of measured point.
    flux: float
        Measured value of flux.
        
    Returns
    -------
    float
        Absolute differnece between polynomial fit and measured flux.    
    """

    return abs(flux - poly(time,*params))
     

def parametrise(train, nb_param, band_used, guess, err, save, 
                checkpoint='', begin=0):
    """Find best fit parameters for the polynomial model.
    
    Parameters
    ----------
    train: pd.DataFrame
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
    """
    
    objects = np.unique(train['object_id'])

    target_list=[]
    for i in objects:
        target_list.append(train.loc[train['object_id']==i,'target'].min())

    # We initialise a table of correct size but filled with 0    
    if begin == 0 :
        nb_passband = len(band_used)           

        df = {'object_id': objects, 'target': target_list}
        
        table = pd.DataFrame(data=df)    
        for i in range(nb_passband*nb_param):
            table[i]=0
            
    # Or we start from checkpoint if provided
    else:
        table=checkpoint


    ## Update the table
                                                     
    np.seterr(all='ignore')

    create_table(objects, train, err, guess, 
                 table, begin, band_used, save)                       
                  
    table.to_pickle(save)          
    table
