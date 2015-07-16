import numpy as np
import matplotlib.pyplot as pl

def triangle(theta, lnprob, obs, n_per_bin=100.,
             plotdir='./' ):
    """Make a corner plot of the binned n-dimensional posteriror
    probability space.
    """
    labels = ['log n', 'log Go', 'fill fraction']
    nmod = len(lnprob)
    norm = (np.exp(lnprob)).sum()

    K = 3
    dim = (K + 0.5) * 3.0
    fig, axes = pl.subplots(K, K, figsize=(dim*1.1, dim))
    for i in xrange(K):
        ax = axes[i,i]
        hist, x = np.histogram(theta[i], weights=np.exp(lnprob)/norm,
                               bins=int(np.sqrt(nmod/n_per_bin)))
        ax.step(x[:-1], hist, where='post', color='k')
        ax.tick_params(axis='y', labelleft='off', labelright='on')
        if i == 0:
            ax.set_title
        for j in xrange(K):
            ax = axes[i,j]
            if j > i:
                ax.set_visible(False)
                ax.set_frame_on(False)
                continue
            elif j == i:
                if i == (K-1):
                    ax.set_xlabel(labels[j])
                else:
                    ax.set_xticklabels([])
                continue

            hist, x, y = np.histogram2d(theta[j], theta[i], weights=np.exp(lnprob)/norm,
                                        bins=int(np.sqrt(nmod/n_per_bin)))
            ax.imshow(hist.T, interpolation='nearest', origin='low',
                      extent=[x[0], x[-1], y[0], y[-1]], aspect='auto',
                      cmap=pl.cm.coolwarm)

            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])
            if i == (K-1):
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
        vals = [obs['region'], obs['x'], obs['y'], obs['line_mask']]
        pl.suptitle('{0}, pixel ({1},{2}), mask = {3}'.format(*vals))
        vals = [''.join(obs['region'].split()), obs['x'], obs['y']]
        fig.savefig(plotdir+'triangle_{0}_x{1:02.0f}_y{2:02.0f}.pdf'.format(*vals))
        pl.close(fig)

def plot_one(theta, lnprob, obs, n_per_bin=100., fontsize=18,
             axwidth=2, plotdir='./'):
    """Separately plot each panel of the corner plot.
    """
    pl.rc('axes', linewidth=axwidth)
    labels = ['log n', 'log G$_o$', 'fill fraction']
    sl = ['n','g','f']
    K = 3
    nmod = len(lnprob)
    norm = (np.exp(lnprob)).sum()
    for i in xrange(K):
        for j in xrange(K):
            if j > i:
                continue
            elif j == i:
                pl.figure()
                pl.clf()
                hist, x = np.histogram(theta[i], weights=np.exp(lnprob)/norm,
                                       bins=int(np.sqrt(nmod/n_per_bin)))
                pl.step(x[:-1], hist, where='post', color='k', linewidth=axwidth)
                pl.xlabel(labels[i], fontsize=fontsize)
                vals = [sl[i], ''.join(obs['region'].split()), obs['x'],obs['y']]
                fnstring = plotdir+'lnphist_{0}_{1}_x{2:02.0f}_y{3:02.0f}.pdf'
                pl.savefig(fnstring.format(*vals))
                pl.close()
            else:
                pl.figure()
                pl.clf()
                hist, x, y = np.histogram2d(theta[j], theta[i],
                                            weights=np.exp(lnprob)/norm,
                                            bins=int(np.sqrt(nmod/n_per_bin)))
                pl.imshow(hist.T, interpolation='nearest', origin='low',
                          extent=[x[0], x[-1], y[0], y[-1]], aspect='auto',
                          cmap=pl.cm.coolwarm)
                pl.colorbar()
                pl.ylabel(labels[i], fontsize=fontsize)
                pl.xlabel(labels[j], fontsize=fontsize)
                pl.tick_params(axis='both', width=axwidth * 1.5, length=5)
                
                vals = [sl[i], sl[j], obs['x'], obs['y'], ''.join(obs['region'].split())]
                fnstring = plotdir+'lnp2d_{0}_vs_{1}_{4}_x{2:02.0f}_y{3:02.0f}.pdf'
                pl.savefig(fnstring.format(*vals))
                pl.close()
                
def line_prediction(theta, lnprob, obs, predicted_lines,
                    line_index=1, plotdir='./'):
    """Plot the ratio of the predicted line intensity to the observed
    line intensity as a function of theta, color coded by
    ln(probability).

    :param theta:
        The parameter against which to plot the model/obs
        ratio. Array-like of shape (nmod,).

    :param line_index: (default: 1)
        The zero-based index of the line for which you want to plot
        predictions.  Order is 'CII158', 'OI63', 'OI145'
    """
    line_name = ['CII158', 'OI63', 'OI145']
    if obs['line_intensity'][line_index] == 0 :
        return
    pp = lnprob.copy()
    pp[pp < lnprob.max()-10] = lnprob.max() - 10
    b = pp > (lnprob.max()-10)  # These are the reasonable fits
    mod_obs = predicted_lines[b,line_index] / obs['line_intensity'][line_index]

    fig, ax = pl.subplots()
    ax.scatter(theta[b], mod_obs,
               c=lnprob[b], linewidths=0)
    ax.set_xlabel('log n')
    ax.set_ylabel('{0} model/obs'.format(line_name[line_index]))
    ax.set_yscale('log')
    ax.set_ylim(0.5,100)
    
    vals = [obs['region'], obs['x'], obs['y'], obs['line_mask']]
    ax.set_title('{0}, pixel ({1},{2}), mask={3}'.format(*vals))
    vals = [''.join(obs['region'].split()), obs['x'], obs['y'], line_name[line_index]]
    fnstring = plotdir+'linepred{3}_{0}_x{1:02.0f}_y{2:02.0f}.pdf'
    fig.savefig(fnstring.format(*vals))
    pl.close(fig)

def point_estimates(theta, lnprob, quantities=None,
                    point_type='best_fit', **kwargs):
    """
    :param lnprob:
        Array like of shape (nmod)

    :param theta:
        Array like, of shape (ntheta, nmod)

    :param quantities:
        Array like, of shape (nquantities, nmod)
    """
    if quantities is not None:
        allq = theta + quantities
    else:
        allq = theta
        
    if point_type == 'best_fit':
        point, chi_best = best_fit(lnprob, allq, **kwargs)
        upper = lower = len(point) * [None]
        pcell = None
        
    if point_type == 'max_den':
        out = best_cell(lnprob, theta, quantities=quantitites,
                        **kwargs)
        point, upper, lower, chi_best, pcell = out
        
    if point_type == 'marginalized':
        out = marginalized_percentiles(lnprob, allq, **kwargs)
        point, upper, lower = out
        chi_best = -2.0 * np.max(lnprob)
        pcell = None

    return point, upper, lower, chi_best, pcell

def best_fit(lnprob, quantities, **extras):
    """Find the chi**2 and the quantities corresponding to the best
    fit.
    
    :param lnprob:
        Array like of shape (nmod) giving ln(probability)

    :param theta:
        Array like, of shape (ntheta, nmod)

    :param quantities:
        Array like, of shape (nquantities, nmod)
    """
    ind_best = np.argmax(lnprob)
    point = [quantity[ind_best] for quantity in quantities]
    return point, -2 * lnprob[ind_best]

def marginalized_percentiles(lnprob, quantities,
                             percentiles=[0.16, 0.5, 0.84],
                             **extras):
    """ Estimate the percentiles of the marginalized pdfs for the
    quantities.
    
    :param lnprob:
        Array like of shape (nmod) giving ln(probability)

    :param quantities:
        Array like, of shape (nquantities, nmod)
    """
    point, upper, lower = [],[],[]
    for q in quantities:
        pctiles, pmax = cdf_moment(q, lnprob, percentiles)
        lower.append(pctiles[0])
        upper.append(pctiles[2])
        point.append(pctiles[1])
    return point, upper, lower

def cdf_moment(inpar, inlnprob, percentiles=[0.16, 0.5, 0.84]):
    """Obtain specified percentiles of the CDF.
    """
    good = np.isfinite(inpar) & np.isfinite(inlnprob)
    par, lnprob = inpar[good], inlnprob[good]
    order = np.argsort(par)
    cdf = np.cumsum(np.exp(lnprob[order])) / np.sum(np.exp(lnprob))
    ind_ptiles= np.searchsorted(cdf, percentiles)
    ind_max=np.argmax(lnprob)

    return par[order[ind_ptiles]], par[ind_max]

def best_cell(lnprob, theta_in, nmod_per_bin=20.,
              quantities=None, **extras):
    """Find the highest probability cell in the n-dimensional
    probablity space, and report the probability weighted average
    parameters in this cell.  The space is divided regularly. I think
    this amounts to a smoothing of the posterior space by a square
    kernel.
    
    :param lnprob:
        Array like of shape (nmod)

    :param theta_in:
        Array like, of shape (ntheta, nmod)

    :param quantities:
        Array like, of shape (nquantities, nmod)
    """
    theta = np.array(theta_in).T
    nmod, ntheta = theta.shape
    assert len(lnprob) == nmod
    
    norm = (np.exp(lnprob)).sum()
    nbins = int((nmod/nmod_per_bin)**(1./ntheta))

    H, edges = np.histogramdd(theta, bins=nbins,
                              weights=np.exp(lnprob)/norm)
    ind_best = np.unravel_index(np.argmax(H), H.shape)
    lower = [e[ind_best[i]] for i,e in enumerate(edges)]
    upper = [e[ind_best[i] + 1] for i,e in enumerate(edges)]
    #point = ((np.array(lower) + np.array(upper))/2.).tolist()

    # Total normalized probability within the cell
    pcell = np.exp(H[ind_best]) / (np.exp(H)).sum()

    # Find the models within the cell
    inds = np.ones(nmod, dtype = bool)
    for i in range(ntheta):
        inds = inds & ((theta[:,i] < upper[i]) & (theta[:,i] >= lower[i]))
    chi_best = -2.0 * lnprob[inds].max()
    
    # Weighted average within the cell
    nquantities = len(quantities)
    point = [np.sum(np.exp(lnprob[inds]) * quantities[i][inds]) /
             np.exp(lnprob[inds]).sum() for i in range(nquantities)]
    return point, upper, lower, chi_best, pcell    
