import sys
import numpy as np
import pdrmodel, pdrplot, pdrdata
from pdrfit import fit_pixel


def pixel_values(theta, lnp, obs, blob=None,
                 blob_desc=None, theta_names=None):

    lines, line_names = blob[0], blob_desc[0]
    qnames = theta_names + line_names
    allq = theta + [lines[:,i] for i in range(len(line_names))]

    # Set up output
    cols = [('x', np.int), ('y', np.int), ('region', 'S50'),
            ('chi_bestcell', np.float), ('chi_bestfit', np.float)]
    ptypes = ['bestcell', 'p16', 'median', 'p84', 'bestfit']
    for name in qnames:
        for ptype in ptypes:
            cols += [('{}_{}'.format(name, ptype), np.float)]
    result = np.zeros(1, dtype=np.dtype(cols))

    # bestfit
    bestfit, bestchi = pdrplot.best_fit(lnp, allq)
    # cell-based
    bcell = pdrplot.best_cell
    bestcell, _, _, cellchi, _ = bcell(lnp, theta, quantities=allq,
                                       nmod_per_bin=30.)
    # marginalized percentiles
    ptiles = pdrplot.marginalized_percentiles
    med, hi, lo = ptiles(lnp, allq, percentiles=[0.16, 0.5, 0.84])

    # Fill output
    result['x'] = obs['x']
    result['y'] = obs['y']
    result['region'] = obs['region']
    result['chi_bestfit'] = bestchi
    result['chi_bestcell'] = cellchi
    for i, name in enumerate(qnames):
        result['{}_bestfit'.format(name)] = bestfit[i]
        result['{}_bestcell'.format(name)] = bestcell[i]
        result['{}_median'.format(name)] = med[i]
        result['{}_p16'.format(name)] = lo[i]
        result['{}_p84'.format(name)] = hi[i]

    return result


if __name__ == '__main__':

    # Set up Data
    region = 'Hubble X'    
    filename = (#"/Users/carlson/Desktop/NGC6822/FitsFiles/HubbleX/Ratios/"
                "../observations/HX_pixelvalues.txt")
                #"HX_pixelvalues_lown.txt")
    allobs = pdrdata.load_obs_fromtxt(filename, region=region, gstar_jitter=0.5,
                                      line_jitter=[0.1, 0.5, 0.5],)
    npix = len(allobs)
    npix = 3

    # Set up Model
    grid = pdrmodel.PDRGrid()
    pdr = pdrmodel.PDRModel()
    priors = {'parnames':['logn', 'logGo', 'fill']}
    priors['logn'] = {'min': 1, 'max':4.5, 'transform':None}
    priors['logGo'] = {'min': 0.5, 'max':4, 'transform':None}
    priors['fill'] = {'min':0.0, 'max':1.0, 'transform':None}
    nmod = 1e3

    # Loop over pixels
    values = []
    for ipix in xrange(npix):

        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]

        # Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdrmodel.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'transform':None}
        # Do the fit
        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=nmod,
                                     grid=grid, pdr=pdr, priors=priors)

        # Make plots
        pdrplot.triangle(theta, lnp, obs, n_per_bin=100)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin=100, fontsize=18)
        pdrplot.line_prediction(theta[0], lnp, obs, blob[0],
                                line_index=1)

        pixvalues = pixel_values(theta, lnp, obs, blob=blob,
                                 blob_desc=pdr.blob_description(grid),
                                 theta_names=priors['parnames'])
        values.append(pixvalues)

    regiondat = np.squeeze(np.array(values))
