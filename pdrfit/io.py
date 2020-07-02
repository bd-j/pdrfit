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

    obs['x'] = dat[:, 0]
    obs['y'] = dat[:, 1]
    obs['region'] = region
    lines = np.array([dat[:, 5], dat[:, 7], dat[:, 9]]).T  # W/m^2/sr
    obs['line_intensity'] = lines
    obs['FIR'][:] = dat[:, 3]   # W/m^2/sr
    obs['FIR_unc'] = dat[:, 4]
    obs['Gstar'] = dat[:, 2]  # Habings
    obs['Gstar_unc'] = obs['Gstar'] * gstar_jitter

    obs['line_unc'] = np.array([dat[:, 6], dat[:, 8], dat[:, 10]]).T
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


def write_dict(incat, outname='out.dat', csv=False, transpose=False):
    """Write a dictionary or Numpy structured array to an ascii file,
    either space delimited or CSV.

    :param incat:
       The catalog to write out, either a dictionary or a numpy
       structured array.  If a dictionary, it is assumed that the
       length of the value in each key:value pair is the same.

    :param outname:
       String giving the name of of the to which the output will be
       written.

    :param csv: If ``True`` the output file is CSV.  If ``False`` the
       output file is space delimited.
    """
    try:
        # input is a dictionary
        colnames = incat.keys()
    except AttributeError:
        # input is a numpy structured array
        colnames = incat.dtype.names

    # Transpose arrays
    if transpose:
        for col in colnames:
            incat[col] = list(np.array(incat[col]).T)

    ncol = len(colnames)
    nrow = len(incat[colnames[0]])
    # Assert each column has the same number of rows and build the
    # header
    hdr = []
    for c in colnames:
        assert len(incat[c]) == nrow
        hdr += np.size(incat[c][0]) * ['{}'.format(c)]

    # Open file and write header
    out = open(outname, 'w')
    out.write('# ')
    if csv:
        out.write(','.join(hdr))
        formatstring = (len(hdr)-1) * '{},' + '{}\n'
    else:
        out.write(' '.join(hdr))
        formatstring = len(hdr) * '{} ' + '\n'
    out.write('\n')

    for irow in range(nrow):
        vals = []
        for col in colnames:
            print(col, incat[col][irow])
            vals += list(np.atleast_1d(incat[col][irow]))
        out.write(formatstring.format(*vals))
    out.close()
