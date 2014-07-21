pdrfit
-

Modules for fitting FIR line ratios to infer physical conditions

Requirements:

1. numpy
2. scipy.spatial (for Delaunay triangulation)
3. scikit-learn (for kdtrees)
4. matplotlib
5. astropy or pyfits


Use:
-To fit all pixels remove `sys.exit()` from `fit_ngc6822.py`, adjust filenames
  and rules for masking, and type `python fit_ngc6822.py` at the command line
  (in the same directory as the codes)
  BETTER: enter pylab by typing      `ipython --pylab`
  Then type `%run fit_ngc6822.py`

- Because of how the fitting is done, errors should never be set to 0. even if the line value is zero
  and/or the line is masked.  This will result in dividing by zero and
  a NAN for the whole calculation.

Description:
- `fit_ngc6822` an example of how to fit the data for a given pixel.

- `pdrfit.py` has classes for holding kaufmann model grids and for generating
  model predictions of observables given physical conditions. Also includes a method
  for reading Kaufmann predition files.

- `modelgrid.py` generic classes for storing and interpolating model grids

- `pdrplot.py` methods to produce pretty plots.

