import numpy as np

filename = '../observations/HX_pixelvalues.txt'


def load_obs_fromtxt(filename, gstar_jitter=0.5, region='',
                     line_jitter=[0.1, 0.5, 0.5], line_mask=None):

    cols = [('x', np.int), ('y', np.int), ('region', 'S50'),
            ('line_intensity', np.float, 3), ('line_unc', np.float, 3),
            ('line_mask', np.bool, 3),
            ('FIR', np.float), ('FIR_unc', np.float),
            ('Gstar', np.float), ('Gstar_unc', np.float)]

    dat = np.loadtxt(filename, skiprows=2)
    npix = dat.shape[0]
    obs = np.zeros(npix, dtype=np.dtype(cols))

    obs['x'] = dat[:,0]
    obs['y'] = dat[:,1]
    obs['region'] = region
    obs['line_intensity'] = np.array([dat[:,5], dat[:,7], dat[:,9]]).T  # W/m^2/sr
    obs['FIR'][:] = dat[:,3]   # W/m^2/sr
    obs['FIR_unc'] = dat[:,4]
    obs['Gstar'] = dat[:,2]  # Habings
    obs['Gstar_unc'] = obs['Gstar'] * gstar_jitter
    
    obs['line_unc'] = np.array([dat[:,6], dat[:,8], dat[:,10]]).T
    extra_variance = (np.array(line_jitter) * obs['line_intensity'])**2
    obs['line_unc'] = np.sqrt(obs['line_unc']**2 + extra_variance)

    # Mask lines to ignore in the fits.  1 = fit, 0 = not fit.
    # Order is CII, OI63, OI145
    if line_mask is None:
        obs['line_mask'] = (dat[:, -3:])  # LYNN
    else:
        obs['line_mask'][:] = line_mask

    return obs


def load_observations_fromfits(files, fields):
    return obs_structured_array


def fit_pixel(obs, model, priors=None):
    return theta, lnprob


def write_text(outfilename='HubbleXpoint_estimates_lown_lowfill_ciiweight.dat'):
    outfile = open(outfilename,'w')
    outfile.write('x y chi_best, logn_best logG_best fill_best OI_best ' 
                  'logn_av logG_av fill_av ' 
                  'logn_lo logG_lo fill_lo ' 
                  'logn_hi logG_hi fill_hi ' 
                  'chibest_cell Pcell OI_cell CII_cell ' 
                  'logn_p50 logG_p50 fill_p50 ' 
                  'logn_p16 logG_p16 fill_p16 ' 
                  'logn_p84 logG_p84 fill_p84 \n')
    return outfile


def write_line(outfile, theta, lnp, blob, obs):

    # Write the pixel number
    outfile.write('{0:2.0f} {1:2.0f} '.format(obs['x'],obs['y']))

    # Get and write point estimates based on best fitting model
    best_theta, chi = pdrdata.best_fit(lnprob, theta)
    best_lines, chi = pdrdata.best_fit(lnprob, blob[0].T)
    vals = [chi] + best_theta + [best_lines[line_index]]
    outfile.write('{0:6.3f} {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3e} '.format(*vals))



    
def pixel_results(theta, lnp, obs,
                  blob=None, blob_desc=None,
                  theta_names=None):

    cols = [('x', np.int), ('y', np.int), ('region', 'S50'),
            ('chi_bestcell', np.float), ('chi_best', np.float)]
    
    lines, line_names = blob[0], blob_desc[0]
    qnames = theta_names + line_names
    q = theta + lines

    
    #for ptype in ['bestcell', 'p16', 'median', 'p84', 'bestfit'
    #cols = cols + zip

    #q_cell, q_low, q_high, 
