# FIR = 1.6e-6*Go/(2*!pi)  relates Go in the file to FIR in units of W m-2 sr-1
# FIR = 1.6e-3*Go/(2*!pi)  relates Go in the file to FIR in units of erg s-1 cm-2 sr-1
# G(FIR)=FIR*2*!pi/(1.6e-6)     ; Go calculated from FIR - with FIR in W m-2 sr-1  and Go in Habings
# Go(stellar) map in Habings.  1 Habing=1.6x10-3 erg s-1 cm-2 =1.6x10-6 W m-2
#    When comparing to model Go, divide by SQRT(2) for a nod to line of sight geometry
# MODELS -
# OI file has columns
# index(n value), log(n), index(g value), log(Go), [OI]63um, [OI]145um
# CP file has
# index(n value), log(n), index(g value), log(Go), [CII]158um
# The [OI]63um, [OI]145um, and [CII] are in erg s-1 cm-2  and should be divided by 2pi to get sr-1

import modelgrid
import numpy as np
from pkg_resources import resource_filename

GtoI = 1.6e-6/(2*np.pi)


class PDRModel(object):
    """Class to produce generative models of PDR emission lines and IR
    luminosities, and to determine the likelihood of a given set of
    observations.
    """

    GtoI = GtoI
    
    def __init__(self):
        pass

    def model(self, logn, logGo, fill, grid=None):
        """Produce observable quantities given physical parameters for
        a model.

        :param logn:
            The log of the gas volume density, in cm^-3.  Scalar or
            ndarray.

        :param logGo:
            The log of the interstellar radiation field intensity, in
            Habings.  Scalar or ndarray of same shape as n.

        :param fill:
            The filling factor of the dense line (and IR emitting) gas
            and dust. Scalar or ndarray of same shape as n.

        :param grid: optional
            A pre-computed PDRGrid.  If not given, then it is assumed
            that the `grid` attribute exists and contains the model grid
            to use.

        :returns lines:
            The line intensities of the models.  ndarray of shape ?

        :returns FIR:
            The FIR luminosity in solar luminosities, ndarray of shape ?

        :returns Gstar:
            The stellar radiation field intensity, in Habings.
        """

        if grid is None:
            grid = self.grid
        Gstar = 10**logGo * np.sqrt(2)
        FIR = 10**logGo * fill * GtoI
        lines = grid.lines(logn, logGo, interpolation='dt')
        lines *= fill[:, None] / (2 * np.pi) * 1e-3
        return lines, FIR, Gstar

    def lnprob(self, theta, obs=None, grid=None):
        """Compute the likelihood of a model or of a grid of models.

        :param theta:
            The physical parameters of the models for which you wish
            to obtain likelihoods.  3-element list or tuple, that
            unpacks to the logn, logGo, and fill values of the models.

        :param obs:
            A dictionary containing the observed values and
            uncertainties thereon.

        :param grid:
            A pre-computed PDRGrid.  If not given, then it is assumed
            that the `grid` attribute exists and contains the model grid
            to use.

        :returns lnprob:
            The likelihood of each of the models.

        :returns blob:
            A list of each of the predicted observables for the models
        """
        logn, logGo, fill = theta
        lines, FIR, Gstar = self.model(logn, logGo, fill, grid=grid)
        lnprob = -0.5 * ((lines - obs['line_intensity'][None, :])**2 /
                         obs['line_unc'][None, :]**2 *
                         obs['line_mask']).sum(axis=-1)
        lnprob += -0.5 * ((FIR - obs['FIR'])**2 / obs['FIR_unc']**2)
        lnprob += -0.5 * ((Gstar - obs['Gstar'])**2 / obs['Gstar_unc']**2)
        return lnprob, [lines, FIR, Gstar]

    def blob_description(self, grid=None):
        """A description of the objects passed out of the lnprob
        method.
        """
        return [grid.line_names, 'FIR', 'Gstar'] 

class PDRGrid(modelgrid.ModelLibrary):
    """Subclass the ModelLibrary to store and procduce line
    intensities from the precomputed Kauffman 2001 models.
    """

    def __init__(self):
        self.files = ['data/CPwMeudonHol.txt', 'data/OPwMeudonHol.txt']
        try:
            self.files = [resource_filename('pdrfit', f) for f in self.files]
        except:
            pass
        
        self.load_kauffman(skiplines=2)

    def load_kauffman(self, skiplines=2):
        """Load the kauffman 2001 PDR models.
        """
        self.wavelength = np.array([158., 63., 145.])
        self.line_names = ['CII158', 'OI63', 'OI145']
        n, g, cii = np.loadtxt(self.files[0], usecols=(1,3,4),
                               skiprows=skiplines, unpack=True)
        n, g, oi63, oi145 = np.loadtxt(self.files[1], usecols=(1,3,4,5),
                                       skiprows=skiplines, unpack=True)

        dt = np.dtype([('logn', '<f8'), ('logGo', '<f8')])
        self.pars = np.zeros(len(n), dtype=dt)
        self.pars['logn'] = n
        self.pars['logGo'] = g
        # ndarray of shape (len(n), len(names))
        self.intensity = np.array([cii, oi63, oi145]).T

    def lines(self, logn, logGo, interpolation='dt'):
        """Use the Delauynay triangulation interpolation techniques of
        the ModelLibrary class to obtain FIR line intensities
        interpolated from the Kaufman 01 grids.

        :param logn:
            The log of the gas volume density, in
            r'$cm^{{-3}}$'. Scalar or array-like.

        :param logGo:
            The log of the interstellar radiation field intensity, in
            Habings.  Scalar or array-like with shape matching n

        :returns intensities:
            The line intensities for the given model parameters, in
            units of ?.  ndarray of shape ?
        """
        parnames = ['logn', 'logGo']
        target_points = np.vstack([np.atleast_1d(logn), np.atleast_1d(logGo)]).T
        inds, weights = self.model_weights(target_points, parnames=parnames,
                                           itype=interpolation)
        intensities = ((weights * (self.intensity[inds].transpose(2,0,1))).sum(axis=2)).T
        return intensities

def sample_priors(nmod, prior_desc):
    pnames = prior_desc['parnames']
    theta = len(pnames) * [[]]
    for i, par in enumerate(pnames):
        mini, maxi = prior_desc[par]['min'], prior_desc[par]['max']
        theta[i] = np.random.uniform(mini, maxi, int(nmod))
    return theta, pnames
