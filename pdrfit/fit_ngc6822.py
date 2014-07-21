import sys
import numpy as np
import pdrfit, pdrplot

def fit_pixel(pixel_values, region, mask = [1,0,0], nmod = 5e4,
              grid = None, pdr = None):

###    Run by typing: python fit_ngc6822.py
    
#######
# Load Observations
########
    dat = pixel_values
    obs = {}
    obs['region'] = region
    obs['pixel'] = dat[0:2]
    obs['line_intensity'] = np.array([dat[5], dat[7], dat[9]]) #W/m^2/sr
    obs['FIR'] = dat[3]  #W/m^2/sr
    obs['Gstar'] = dat[2] #Habings
    obs['line_unc'] = np.sqrt(np.array([dat[6], dat[8], dat[10]])**2 + (0.1 * obs['line_intensity'])**2)
    obs['FIR_unc'] = dat[4]
    obs['Gstar_unc'] = obs['Gstar'] * 0.5

# Mask lines to ignore in the fits.  1 = fit, 0 = not fit.  Order is CII, OI63, OI145
#    obs['line_mask'] = np.array(mask)
    obs['line_mask']= dat[ -3 : ]  # LYNN
    
#######
# Build the model grid
######
    if grid is None:
        grid = pdrfit.load_kauffman()
    if pdr is None:
        pdr = pdrfit.PDRModel()
    
    priors = {}
    priors['n'] = {'min': 1, 'max':6, 'scale':'log'}
    priors['Go'] = {'min': 0.5, 'max':4.5, 'scale':'log'}
    fill_try = obs['FIR'] / (obs['Gstar'] * pdrfit.GtoI) #first guess at filling factor
    priors['fill'] = {'min': fill_try/4., 'max':1.0, 'scale':'linear'}
    
    fill_try = obs['FIR'] / (obs['Gstar'] * pdrfit.GtoI)
#use random sampling of the parameter space.  Could also set up here to use grids
    n = np.random.uniform(priors['n']['min'], priors['n']['max'], int(nmod))
    Go = np.random.uniform(priors['Go']['min'], priors['Go']['max'], int(nmod))
    fill = np.random.uniform(priors['fill']['min'], priors['fill']['max'], int(nmod))
    theta = [n, Go, fill]

#######
# Calculate the model probabilities (and model predicted intensities)
######

#lnprob, blob = pdr.lnprob(theta, obs = obs, grid =grid)
#predicted_lines = blob[0]

    lnprob, predicted_lines = np.zeros(int(nmod)), np.zeros([int(nmod), len(obs['line_intensity'])]) 
    predicted_fir, predicted_gstar = np.zeros(int(nmod)), np.zeros(int(nmod))
    maxmod = 1e4
    i = 0
    #split to conserve memory
    while (i*maxmod <= nmod):
        s1, s2 = int((i)*maxmod), int(np.min( [(i+1)*maxmod-1,nmod] ))
        l, b = pdr.lnprob([theta[0][s1:s2], theta[1][s1:s2], theta[2][s1:s2]], obs = obs, grid =grid)
        lnprob[s1:s2] = l
        predicted_lines[s1:s2,:] = b[0]
        predicted_fir[s1:s2] = b[1]
        predicted_gstar = b[2]
        i += 1

    return theta, lnprob, obs, [predicted_lines, predicted_fir, predicted_gstar], [grid, pdr]

######
# Plot results
######

if __name__ == '__main__':

    #    pixel_values = [9, 7, 1372.8338, 4.8852239e-05, 7.32784e-06,
    #                    1.27997e-07, 4.22917e-10, 7.47584e-08, 5.60757e-10, 7.13582e-09, 2.57272e-10, 1, 1, 1]
                    #    region = 'Hubble V'
                    #    theta, lnprob, obs, predictions  = fit_pixel(pixel_values, region, mask = [1,0,0], nmod = 5e4)
                    #   theta, lnprob, obs, predictions  = fit_pixel(pixel_values, region, nmod = 5e4) # LYNN
                    #    pdrplot.triangle(theta, lnprob, obs)
                    #    pdrplot.line_prediction(theta, lnprob, obs, predictions[0])
    
                    #    sys.exit()
    
    dat = np.loadtxt('../observations/HV_pixelvalues.txt', skiprows = 2)
    region = 'Hubble V'
    npix = dat.shape[0]
    
    outfile = open('ngc6822_point_estimates.dat','w')
    outfile.write('x y chi_best, logn_best logG_best fill_best OI_best ' \
                  'logn_av logG_av fill_av ' \
                  'logn_lo logG_lo fill_lo ' \
                  'logn_hi logG_hi fill_hi ' \
                  'chibest_cell Pcell OI_cell CII_cell \n')

    grid = pdr = None
    for ipix in xrange(2):
        print('pixel #{0} of {1}'.format(ipix, npix))
        pixel_values = dat[ipix,:].tolist() 
        #        theta, lnp, obs, pred = fit_pixel(pixel_values, region, mask = [1,1,0], nmod = 5e4)
        theta, lnp, obs, pred, mod = fit_pixel(pixel_values, region, nmod = 5e4, grid = grid, pdr =pdr) # LYNN
        lnp[lnp == 0.] = lnp.min() #HACK
        grid, pdr = mod[0], mod[1]
        
        pdrplot.triangle(theta, lnp, obs, n_per_bin = 100.)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin = 100.)
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
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*point))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*lo))
        outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} '.format(*up))
        outfile.write('{0:6.3f} {1:6.3e} '.format(chibest_cell, pcell))
        outfile.write('{0:6.3e} {1:6.3e} '.format(cell_lines[1], cell_lines[0]))
        outfile.write('\n') #newline
        
    outfile.close()
