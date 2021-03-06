import sys
import numpy as np
from pdrfit import io as pdrdata
from pdrfit import pdrplot
from pdrfit import fit_pixel, PDRGrid, PDRModel

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits 

def fit_region(datafile, region='', nmod=1e3,
               npb=100, plotdir='./results/'):
    
    # Set up Data
    allobs = pdrdata.load_obs_fromtxt(datafile, region=region,
                                      gstar_jitter=0.5,
                                      line_jitter=[0.1, 0.5, 0.5],)
    npix = len(allobs)
    
    # Set up Model
    grid = PDRGrid()
    pdr = PDRModel()
    priors = {'parnames':['logn', 'logGo', 'fill']}
    priors['logn'] = {'min': 1, 'max':4.5, 'transform':None}
    priors['logGo'] = {'min': 0.5, 'max':4, 'transform':None}
    priors['fill'] = {'min':0.0, 'max':1.0, 'transform':None}

    # Loop over pixels
    values = []
    for ipix in xrange(npix):

        print('pixel #{0} of {1}'.format(ipix, npix))
        obs = allobs[ipix]

        # Modify the filling factor prior
        fill_try = obs['FIR'] / (obs['Gstar'] * pdr.GtoI) 
        priors['fill'] = {'min': fill_try/3.,
                          'max': np.min([3.*fill_try, 1.0]),
                          'transform':None}
        # Do the fit
        theta, lnp, blob = fit_pixel(obs, mask=None, nmod=nmod,
                                     grid=grid, pdr=pdr, priors=priors)

        # Make plots
        pdrplot.triangle(theta, lnp, obs, n_per_bin=npb*2,
                         plotdir=plotdir)
        pdrplot.plot_one(theta, lnp, obs, n_per_bin=npb*2,
                         plotdir=plotdir, fontsize=18)
        pdrplot.line_prediction(theta[0], lnp, obs, blob[0],
                                line_index=1, plotdir=plotdir)

        # point estimates and statistics
        pixvalues = pixel_values(theta, lnp, obs, blob=blob, npb=npb,
                                 blob_desc=pdr.blob_description(grid),
                                 theta_names=priors['parnames'])
        values.append(pixvalues)

    regiondat = np.squeeze(np.array(values))
    return regiondat

def pixel_values(theta, lnp, obs, blob=None, npb=100,
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
                                       nmod_per_bin=npb)
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

    region = 'Hubble X'    
    filename = ("observations/HX_pixelvalues.txt")
    dat = fit_region(filename, region=region, nmod=5e4,
                     npb=50, plotdir='./plots/')

    outroot = ''.join(region.split())+'.pdrfit'
    # Write a fits ascii table
    bintab = pyfits.TableHDU(data=dat)
    bintab.writeto(outroot + '.fits', clobber=True)
    # Wite a csv file
    pdrdata.write_dict(dat, outname=outroot+'.csv', csv=True)
