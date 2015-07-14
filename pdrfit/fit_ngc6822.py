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
    theta, parnames = fitpix.sample_priors(nmod, priors)

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

    region = 'Hubble X'    
    filename = '/Users/carlson/Desktop/NGC6822/FitsFiles/HubbleX/Ratios/HX_pixelvalues_lown.txt'
    allobs = fitpix.load_obs_fromtxt(filename)
    npix = len(allobs)
    
    outfile = open('HubbleXpoint_estimates_lown_lowfill_ciiweight.dat','w')
    outfile.write('x y chi_best, logn_best logG_best fill_best OI_best ' \
                  'logn_av logG_av fill_av ' \
                  'logn_lo logG_lo fill_lo ' \
                  'logn_hi logG_hi fill_hi ' \
                  'chibest_cell Pcell OI_cell CII_cell ' \
                  'logn_p50 logG_p50 fill_p50 ' \
                  'logn_p16 logG_p16 fill_p16 ' \
                  'logn_p84 logG_p84 fill_p84 \n')

    grid = pdr = None
    priors = {}
    priors['n'] = {'min': 1, 'max':4.5, 'scale':'log'}
    priors['Go'] = {'min': 0.5, 'max':4, 'scale':'log'}

    for ipix in xrange(npix):
        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]
        #Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdrfit.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'scale':'linear'}

        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=1e5,
                                     grid=grid, pdr=pdr)
        lnp[lnp == 0.] = lnp.min() #HACK
        grid, pdr = mod[0], mod[1]
        
        pdrplot.triangle(theta, lnp, obs, n_per_bin = 100.)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin = 100., fontsize=18)
        pdrplot.line_prediction(theta, lnp, obs, pred[0])

        #write the pixel number
        outfile.write('{0:2.0f} {1:2.0f} '.format(obs['pixel'][0],obs['pixel'][1]))
                      
        # Get point estimates based on best fitting model
        bestfit_theta, _, _, best_lines, chi_best, _ = pdrplot.point_estimates(theta, lnp, obs, pred[0], point_type = 'best_fit')
        outfile.write('{0:6.3f} {2:6.3f} {3:6.3f} {4:6.3f} {1:6.3e} '.format(chi_best, best_lines[1], *bestfit_theta))
        
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
