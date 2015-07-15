import sys
import numpy as np
import pdrmodel, pdrplot, pdrdata

def fit_pixel(obs, mask=None, nmod=5e4, maxmod=1e4,
              grid=None, pdr=None, priors=None):
    """Fit data for a single pixel.
    """
    if mask is not None:
        # Mask lines to ignore in the fits. 1 = fit, 0 = not fit.
        # Order is CII, OI63, OI145
        obs['line_mask'] = np.array(mask, dtype=np.bool)

    #---------------
    # Build the model grid
    #---------------
    if grid is None:
        grid = pdrmodel.PDRGrid()
    if pdr is None:
        pdr = pdrmodel.PDRModel()
    theta, parnames = pdrmodel.sample_priors(nmod, priors)

    #----------------
    # Calculate the model probabilities (and model predicted intensities)
    #----------------
    nm = int(nmod)
    lnprob = np.zeros(nm)
    model_lines = np.zeros([nm, len(obs['line_intensity'])]) 
    model_fir, model_gstar = np.zeros(nm), np.zeros(nm)
#    blob = []
    i = 0
    # Split to conserve memory
    while (i*maxmod <= nmod):
        s1, s2 = int((i)*maxmod), int(np.min( [(i+1)*maxmod-1,nmod] ))
        chunk_theta = [theta[0][s1:s2], theta[1][s1:s2], theta[2][s1:s2]]
        lnp, chblob = pdr.lnprob(chunk_theta, obs=obs, grid=grid)
        lnprob[s1:s2] = lnp
#        blob += blob
        model_lines[s1:s2,:] = chblob[0]
        model_fir[s1:s2] = chblob[1]
        model_gstar = chblob[2]
        i += 1

    return theta, lnprob, [model_lines, model_fir, model_gstar]


if __name__ == '__main__':

    # Data
    region = 'Hubble X'    
    filename = (#"/Users/carlson/Desktop/NGC6822/FitsFiles/HubbleX/Ratios/"
                "../observations/HX_pixelvalues.txt")
                #"HX_pixelvalues_lown.txt")
    allobs = pdrdata.load_obs_fromtxt(filename)
    npix = len(allobs)
    npix = 1 

    # Model
    grid =  pdrmodel.PDRGrid()
    pdr = pdrmodel.PDRModel()
    priors = {'parnames':['logn', 'logGo', 'fill']}
    priors['logn'] = {'min': 1, 'max':4.5, 'transform':None}
    priors['logGo'] = {'min': 0.5, 'max':4, 'transform':None}
    priors['fill'] = {'min':0.0, 'max':1.0, 'transform':None}

    # Loop over pixels
    for ipix in xrange(npix):
        
        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]
        
        #Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdrmodel.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'transform':None}

        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=2e4,
                                     grid=grid, pdr=pdr, priors=priors)
        lnp[lnp == 0.] = lnp.min() #HACK
        
        pdrplot.triangle(theta, lnp, obs, n_per_bin=100)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin=100, fontsize=18)
        pdrplot.line_prediction(theta, lnp, obs, blob[0])
