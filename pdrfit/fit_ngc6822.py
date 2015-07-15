import sys
import numpy as np
import pdrmodel, pdrplot, pdrdata
from pdrfit import fit_pixel

if __name__ == '__main__':

    # Set up Data
    region = 'Hubble X'    
    filename = (#"/Users/carlson/Desktop/NGC6822/FitsFiles/HubbleX/Ratios/"
                "../observations/HX_pixelvalues.txt")
                #"HX_pixelvalues_lown.txt")
    allobs = pdrdata.load_obs_fromtxt(filename)
    npix = len(allobs)
    npix = 1 

    # Set up Model
    grid = pdrmodel.PDRGrid()
    pdr = pdrmodel.PDRModel()
    priors = {'parnames':['logn', 'logGo', 'fill']}
    priors['logn'] = {'min': 1, 'max':4.5, 'transform':None}
    priors['logGo'] = {'min': 0.5, 'max':4, 'transform':None}
    priors['fill'] = {'min':0.0, 'max':1.0, 'transform':None}

    # Loop over pixels
    for ipix in xrange(npix):
        
        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]
        
        # Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdrmodel.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'transform':None}
        # Do the fit
        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=2e4,
                                     grid=grid, pdr=pdr, priors=priors)

        # Make plots
        pdrplot.triangle(theta, lnp, obs, n_per_bin=100)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin=100, fontsize=18)
        pdrplot.line_prediction(theta[0], lnp, obs, blob[0],
                                line_index=1)
        
        v = pdrdata.pixel_values(table, theta, lnp, obs,
                                 blob=blob,
                                 blob_desc=pdr.blob_description(grid),
                                 theta_names=priors['parnames'])
                            
