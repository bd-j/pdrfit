import numpy as np
import pdrmodel

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
    lnprob = np.zeros(nm) #- np.infty
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
        
    lnprob[lnprob == 0.] = lnprob[lnprob != 0.].min() #HACK
    return theta, lnprob, [model_lines, model_fir, model_gstar]
