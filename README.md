pdrfit
-

Modules for fitting FIR line ratios to infer physical conditions

Requirements:

1. numpy
2. scipy.spatial (for Delaunay triangulation)
3. matplotlib
4. astropy or pyfits
5. (optional)  scikit-learn (for kdtrees)


Usage:
- See `demo/fit_ngc6822.py`

- Because of how the fitting is done, errors should never be set
  to 0. even if the line value is zero and/or the line is masked.
  This will result in dividing by zero and a NAN for the whole
  calculation.

Description:
- `demo/fit_ngc6822` an example of how to fit the data and store and
  produce output for a given region

- `pdrfit/pdrfit.py` Has a method to fit data for a single pixel

- `pdrfit/pdrmodel.py` has classes for reading and storing holding
Kaufmann model grids and for generating model predictions of
observables given physical conditions. 

- `pdrfit/io.fits` Methods for reading observational data into numpy
  structured arrays

- `pdrfit/pdrplot.py` methods to produce pretty plots and to produce
  point estimates from posterior grids

- `pdrfit/modelgrid.py` generic classes for storing and interpolating
  model grids.  Typically should be left alone.
