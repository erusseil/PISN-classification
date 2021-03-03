import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timeit
from scipy.optimize import least_squares

def fit_scipy(time, flx, err_model,guess):
    flux = np.asarray(flx)
    result = least_squares(err_model, guess, args=(time, flux))
    return result.x

def get_param(train,ide,err_model,guess,band_used):
    ligne=[]
    error=[]
    
    for i in range(len(guess)):
        error.append(float("nan"))
            
    for i in band_used:#range(len(guess)):
        obj = train.loc[(train['object_id']==ide) & (train['passband']==i)]
        flux=np.array(obj['flux'])
        time=np.array(obj['mjd'])
        
        if err_model==errfunc_bazin:
            t0=time[np.argmax(flux)]
            a0=flux.max()
            guess[2],guess[0]=t0,a0
        
        if len(time)<=1:
            ligne.append(np.array(error))
        else:
            ligne.append(fit_scipy(time, flux,err_model,guess))
        
    return np.array(ligne).flatten()

# This function will create the data frame containing all the parameters

def create_table(objects,train,err_model,guess,feature,begin,band_used):
    
    start = timeit.default_timer()
    
    for i in range(begin,len(objects)):
        print(i,'/',len(objects), end="\r")
        feature.loc[i,0:]=get_param(train,objects[i],err_model,guess,band_used)

    stop = timeit.default_timer()
    print('Total time of the parametrisation %.1f sec'%(stop - start)) 
    
    return feature


    
    # We define the polynomial model

def poly(time,a,b,c,d,e):
    return c+b*time+a*time**2+d*time**3+e*time**4

def errfunc_poly(params,time, flux):
    return abs(flux - poly(time,*params))


        # We define the Bazin model
def bazin(time, a, b, t0, tfall, trise):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return a * X + b

def errfunc_bazin(params,time, flux):
    return abs(flux - bazin(time,*params))

        # We define the small polynomial model

def tiny_poly(time,a,b,c):
    return c+b*time+a*time**2

def errfunc_tiny_poly(params,time, flux):
    return abs(flux - tiny_poly(time,*params))
     




def parametrise(data,nb_param,band_used,guess,err,begin,save):
    

    
    
    ## On importe l'essentiel
    train=pd.read_pickle("%s.pkl"%data)    # <------
    objects = np.unique(train['object_id'])

    target_list=[]
    for i in objects:
        target_list.append(train.loc[train['object_id']==i,'target'].min())

        # ############################# R E S E T #################################
    #On initialise un tableau de 0
    
    # NOMBRE DE PARAM EST UN ENTIER
    # BAND USED EST UN ARRAY [0,1,2,3]
    # GUESS EST UN ARRAY # [10, 0, 1, 30, -5] good for bazin
    
    nb_passband=len(band_used)           

    df = {'object_id': objects,'target':target_list}
    table = pd.DataFrame(data=df)    
    for i in range(nb_passband*nb_param):
        table[i]=0



        ############### CALCULATE ALL THE PARAMETERS FOR A GIVEN MODEL ###################
    ## Update notre tableau de 0
                                                        # EDIT
    np.seterr(all='ignore')

    create_table(objects,train,err,guess,table,begin,band_used)                        
    table.to_pickle("Models/%s.pkl"%save)               # <------
    table
