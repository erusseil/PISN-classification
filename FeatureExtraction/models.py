import numpy as np

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

def poly3(time,a,b,c,d):
    """Polynomial cubic function of degree.
    
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
    
    return d + c * time + b * time ** 2 + a * time ** 3


def errfunc_poly3(params, time, flux):
    """Absolute difference between poly3 fit and measured flux.
    
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

    return abs(flux - poly3(time,*params))
     