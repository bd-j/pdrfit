#FIR = 1.6e-6*Go/(2*!pi)  relates Go in the file to FIR in units of W m-2 sr-1
#FIR = 1.6e-3*Go/(2*!pi)  relates Go in the file to FIR in units of erg s-1 cm-2 sr-1
#G(FIR)=FIR*2*!pi/(1.6e-6)     ; Go calculated from FIR - with FIR in W m-2 sr-1  and Go in Habings
#Go(stellar) map in Habings.  1 Habing=1.6x10-3 erg s-1 cm-2 =1.6x10-6 W m-2
#    When comparing to model Go, divide by SQRT(2) for a nod to line of sight geometry 
#MODELS -
#OI file has columns
#index(n value), log(n), index(g value), log(Go), [OI]63um, [OI]145um
#CP file has
#index(n value), log(n), index(g value), log(Go), [CII]158um
#The [OI]63um, [OI]145um, and [CII] are in erg s-1 cm-2  and should be divided by 2pi to get sr-1


import modelgrid
import numpy as np

GtoI = 1.6e-6/(2*np.pi)

class PDRModel(object):
    
    def __init__(self):
        pass
        
    def model(self, n, Go, fill, grid = None):
        if grid is None:
            grid = self.grid
        Gstar = 10**Go * np.sqrt(2)
        FIR = 10**Go * fill * GtoI
        lines = grid.lines(n, Go, interpolation = 'dt') * fill[:,None] / (2 *np.pi) * 1e-3
        return lines, FIR, Gstar
          
    def lnprob(self, theta, obs = None, grid = None):
        n, Go, fill = theta
        lines, FIR, Gstar = self.model(n, Go, fill, grid = grid)
        lnprob = -0.5 * ((lines - obs['line_intensity'][None,:])**2 /
                         obs['line_unc'][None,:]**2 *
                         obs['line_mask']).sum(axis = -1)
        lnprob +=  -0.5 * ((FIR -obs['FIR'])**2 / obs['FIR_unc']**2)
        lnprob +=  -0.5 * ((Gstar -obs['Gstar'])**2 / obs['Gstar_unc']**2)
        return lnprob, [lines, FIR, Gstar]
        
class PDRGrid(modelgrid.ModelLibrary):

    pars = 0.
    intensity = 0.
    
    def __init__(self):
        pass
        
    def lines(self, n, Go, interpolation = 'dt'):
        parnames = ['n', 'Go']
        target_points = np.vstack([np.atleast_1d(n),np.atleast_1d(Go)]).T
        inds, weights = self.model_weights(target_points, parnames = parnames, itype=interpolation)
        intensities  = (( weights* (self.intensity[inds].transpose(2,0,1)) ).sum(axis=2)).T
        return intensities
            
def load_kauffman():
    files = ['data/CP_Meudon_interpol.txt', 'data/OI_Meudon_interpol.txt']
    files = ['data/CPwMeudonHol.txt', 'data/OPwMeudonHol.txt']
    skiplines = 2
    names = ['CII158','OI63', 'OI145']
    waves = np.array([158.,63.,145.])
    grid = PDRGrid()
    grid.wavelength = waves
    grid.line_names = names
    
    n, g, cii = np.loadtxt(files[0], usecols = (1,3,4), skiprows = skiplines, unpack = True)
    grid.pars = np.zeros(len(n), dtype = np.dtype([('n','<f8'), ('Go', '<f8')]))
    grid.pars['n'] = n
    grid.pars['Go'] = g
    grid.intensity = np.zeros([len(n), len(names)])
    grid.intensity[:,0] = cii
    n, g, oi63, oi145 = np.loadtxt(files[1], usecols = (1,3,4,5), skiprows = skiplines, unpack = True)
    grid.intensity[:,1] = oi63
    grid.intensity[:,2] = oi145
    return grid
    
def read_kauffman(filename, as_struct = False):
    """Read the output of the interpolated Kaufman PDR models for the
    line intensity expected at a given density and G_o.

    :param filename:
        name of the file containing the model predictions
    :returns n:
        density values
    :returns Go:
        radiation field values
    :returns I:
        line intensity (or FIR luminosity)
    """
    tmp = open(filename, 'rb')
    while len(tmp.readline().split()) < 5:
        skiprows += 1
    tmp.close()

    if as_struct:
        f8 = '<f8'
        dt = np.dtype([('n',f8), ('Go', f8), ('I', f8)])
        data = np.loadtxt(filename, usecols = (1,3,4), dtype = dt, skiprows = skiprows)
        return data
    else:
        n, Go, I = np.loadtxt(filename, usecols = (1,3,4), skiprows = skiprows)
        return n, Go, I
