import numpy as np
import matplotlib.pyplot as pl

def triangle(theta, lnprob, obs, n_per_bin = 100.):
    labels = ['log n', 'log Go', 'fill fraction']
    nmod = len(lnprob)
    norm = (np.exp(lnprob)).sum()
    
    K = 3
    dim = (K + 0.5) * 3.0
    fig, axes = pl.subplots(K, K, figsize=(dim*1.1, dim))
    for i in xrange(K):
        ax = axes[i,i]
        hist, x = np.histogram(theta[i], weights = np.exp(lnprob)/norm, bins = int(np.sqrt(nmod/n_per_bin)))
        ax.step(x[:-1],hist, where ='post', color = 'k')
#        ax.hist(theta[i], weights = np.exp(lnprob)/norm, bins = int(nmod/n_per_bin),color = 'k', histtype = 'step')
        ax.tick_params(axis ='y', labelleft = 'off', labelright = 'on')
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
            
            hist, x, y = np.histogram2d(theta[j], theta[i], weights = np.exp(lnprob)/norm,
                                        bins = int(np.sqrt(nmod/n_per_bin)))
            ax.imshow(hist.T, interpolation = 'nearest', origin = 'low',
                      extent=[x[0], x[-1], y[0], y[-1]], aspect ='auto')
            #
            #pl.colorbar()


            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])
            if i == (K-1):
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])

        pl.suptitle('{0}, pixel ({1},{2}), mask = {3}'.format(obs['region'], obs['pixel'][0],obs['pixel'][1], obs['line_mask']))
        fig.savefig('results/triangle_{0}_x{1:02.0f}_y{2:02.0f}.pdf'.format(''.join(obs['region'].split()), obs['pixel'][0],obs['pixel'][1]))
        pl.close(fig)


def plot_one(theta, lnprob, obs, n_per_bin = 100., fontsize =18, axwidth = 2):
    pl.rc('axes', linewidth=axwidth)
    labels = ['log n', 'log Go', 'fill fraction']
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
                hist, x = np.histogram(theta[i], weights = np.exp(lnprob)/norm, bins = int(np.sqrt(nmod/n_per_bin)))
                pl.step(x[:-1],hist, where ='post', color = 'k', linewidth = axwidth)
                pl.xlabel(labels[i], fontsize = fontsize)
                pl.savefig('results/lnphist_{0}_{1}_x{2:02.0f}_y{3:02.0f}.pdf'.format(sl[i], ''.join(obs['region'].split()),
                                                                                     obs['pixel'][0],obs['pixel'][1]))
                pl.close()
            else:
                pl.figure()
                pl.clf()
                hist, x, y = np.histogram2d(theta[j], theta[i], weights = np.exp(lnprob)/norm,
                                        bins = int(np.sqrt(nmod/n_per_bin)))
                pl.imshow(hist.T, interpolation = 'nearest', origin = 'low',
                          extent=[x[0], x[-1], y[0], y[-1]], aspect ='auto', cmap = pl.cm.coolwarm)
                pl.colorbar()
                pl.ylabel(labels[i], fontsize = fontsize)
                pl.xlabel(labels[j], fontsize = fontsize)
                pl.tick_params(axis = 'both', width = axwidth * 1.5, length = 5)
                pl.savefig('results/lnp2d_{0}_vs_{1}_{4}_x{2:02.0f}_y{3:02.0f}.pdf'.format(sl[i], sl[j],
                                                                                            obs['pixel'][0],obs['pixel'][1],
                                                                                            ''.join(obs['region'].split())))
                pl.close()
                
def line_prediction(theta, lnprob, obs, predicted_lines, line_index = 1):
    line_name = ['CII158', 'OI63', 'OI145']
    if obs['line_intensity'][line_index] == 0 :
        return
    pp = lnprob.copy()
    pp[pp < lnprob.max()-10] = lnprob.max()-10
    b = (pp > lnprob.max()-10)  # these are the reasonable fits
    pl.figure()
    pl.clf()
    pl.scatter(theta[0][b], predicted_lines[b,line_index]/obs['line_intensity'][line_index],c = lnprob[b], linewidths = 0)
    pl.xlabel('log n')
    pl.ylabel('{0} model/obs'.format(line_name[line_index]))
    pl.yscale('log')
    pl.ylim(0.5,100)
    #pl.colorbar()
    pl.title('{0}, pixel ({1},{2}), mask = {3}'.format(obs['region'], obs['pixel'][0],obs['pixel'][1], obs['line_mask']))
    pl.savefig('results/linepred{3}_{0}_x{1:02.0f}_y{2:02.0f}.pdf'.format(''.join(obs['region'].split()), obs['pixel'][0],obs['pixel'][1], line_name[line_index]))
    pl.close()


def point_estimates(theta, lnprob, obs, predicted_lines,
                    point_type = 'best_fit', n_per_bin = 30):

    nmod = len(lnprob)
    ntheta = len(theta)
    nlines = predicted_lines.shape[1]

    #best fit
    if point_type == 'best_fit':
        ind_best = np.argmax(lnprob)
        point = [t[ind_best] for t in theta]
        lower = ntheta * [0.]
        upper = ntheta * [0.]
        lines = [predicted_lines[ind_best, i] for i in range(nlines)]
        chi_best = -2.0 * lnprob.max()
        pcell = None
        
    #cell with maximum probability density
    if point_type == 'max_den':
        norm = (np.exp(lnprob)).sum()
        nbins = int((nmod/n_per_bin)**(1./ntheta))
        ta = np.array(theta).T
        H, edges = np.histogramdd(ta, bins = nbins, weights = np.exp(lnprob)/norm)
        ind_best = np.unravel_index(np.argmax(H), H.shape)
        lower = [e[ind_best[i]] for i,e in enumerate(edges)]
        upper = [e[ind_best[i] + 1] for i,e in enumerate(edges)]
        point = ((np.array(lower) + np.array(upper))/2.).tolist()
        
        pcell = np.exp(H[ind_best]) / (np.exp(H)).sum()
        inds = np.ones(nmod, dtype = bool)
        for i in range(ntheta):
            inds = inds & ((theta[i] < upper[i]) & (theta[i] >= lower[i]))
        chi_best = -2.0 * lnprob[inds].max()
        #weighted average line predictions
        lines = [ np.sum(np.exp(lnprob[inds]) * predicted_lines[inds, i]) / np.exp(lnprob[inds]).sum() 
                  for i in range(nlines)]
        
    #marginalized medians and other percentiles
    if point_type == 'marginalized':
        print('not implemented')

    return point, upper, lower, lines, chi_best, pcell
    
