import sys
import numpy as np
import pdrfit, pdrplot, pdrdata

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
        grid = pdrfit.PDRGrid()
    if pdr is None:
        pdr = pdrfit.PDRModel()
    theta, parnames = pdrfit.sample_priors(nmod, priors)

    #----------------
    # Calculate the model probabilities (and model predicted intensities)
    #----------------
    nm = int(nmod)
    lnprob = np.zeros(nm),
#    model_lines = np.zeros([nm, len(obs['line_intensity'])]) 
#    model_fir, model_gstar = np.zeros(nm), np.zeros(nm)
    blob = []
    i = 0
    # Split to conserve memory
    while (i*maxmod <= nmod):
        s1, s2 = int((i)*maxmod), int(np.min( [(i+1)*maxmod-1,nmod] ))
        chunk_theta = [theta[0][s1:s2], theta[1][s1:s2], theta[2][s1:s2]]
        lnp, chblob = pdr.lnprob(chunk_theta, obs=obs, grid=grid)
        lnprob[s1:s2] = lnp
        blob.append(chblob)
#        model_lines[s1:s2,:] = chblob[0]
#        model_fir[s1:s2] = chblob[1]
#        model_gstar = chblob[2]
        i += 1

    return theta, lnprob, blob #, [model_lines, model_fir, model_gstar]


if __name__ == '__main__':

    # Data
    region = 'Hubble X'    
    filename = ("/Users/carlson/Desktop/NGC6822/FitsFiles/HubbleX/Ratios/"
                "HX_pixelvalues_lown.txt")
    allobs = pdrdata.load_obs_fromtxt(filename)
    npix = len(allobs)

    # Model
    grid =  pdrfit.PDRGrid()
    pdr = pdrfit.PDRModel()
    priors = {'parnames':['logn', 'logGo', 'fill']}
    priors['logn'] = {'min': 1, 'max':4.5, 'transform':None}
    priors['logGo'] = {'min': 0.5, 'max':4, 'transform':None}
    priors['fill'] = {'min':0.0, 'max':1.0, 'transform':None}

    # Loop over pixels
    for ipix in xrange(npix):
        
        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]
        
        #Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdrfit.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'transform':None}

        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=1e5,
                                     grid=grid, pdr=pdr, priors=priors)
        lnp[lnp == 0.] = lnp.min() #HACK
        
        pdrplot.triangle(theta, lnp, obs, n_per_bin=100)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin=100, fontsize=18)
        pdrplot.line_prediction(theta, lnp, obs, blob[0])

        
        # Get point estimates based on the cell in the 3-d (n, Go, fill) histogram of
        #  probability with the maximum probability density.  upper and lower give the
        #  edges of the cell in each dimension (not actual uncertainty estimates).
        #  This could be very noisy, especially if n_per_bin is low....
        point, up, lo, cell_lines, chibest_cell, pcell = pdrplot.point_estimates(theta, lnp, obs, pred[0], point_type = 'max_den', n_per_bin = 20.)
        p50, p84, p16, marg_lines, _, _ = pdrplot.point_estimates(theta, lnp, obs, pred[0],
                                                                  point_type = 'marginalized')

        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*point))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*lo))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*up))
        outfile.write('{0:6.3f} {1:6.3e} '.format(chibest_cell, pcell))
        outfile.write('{0:6.3e} {1:6.3e} '.format(cell_lines[1], cell_lines[0]))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*p50))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*p16))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*p84))
        outfile.write('\n') #newline
        
    outfile.close()
