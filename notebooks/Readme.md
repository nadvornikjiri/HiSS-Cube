# Notebook contents
The IPython notebooks in this folder are documentation for separate HiSS-Cube areas.

1. `SDSS_cube.ipynb` - Overall documentation of the preprocessing phase of HiSS-Cube. Essentially describes in detail what happens in `hisscube.py create`.
2. `h5web.ipynb` - Showcase of the H5Web viewer with the created HiSS-Cube file. Run the previous notebook first to create the file or create it yourself.
3. `ml_testing.ipynb` - Shows how to work with the 3D cube (image, spectral axes, with temporal axis aggregated), created in the `ml-cube` subcommand of `hisscube.py update`.
4. `SNR tests.ipynb` - Shows how to import the Star formation rate data into HiSS-Cube and work with it in combination with the 3D cube from `ml-cube` step.


# Star formation rate use case
The data shown in the `SNR tests.ipynb` can be used to estimate the Star Formation Rate on combined spectra and images by HiSS-Cube. We want to train the Star Formation Rate parameter prediction based on the SDSS Spectra and images combination (3D cube from the `ml-cube` step, see `ml_testing.ipynb` notebook), but to be able to predict Star Formation Rate on image data only as a result.

The envisioned machine learning algorithm would work as follows:
1. Train neural network to predict spectra based on images:
  1. Input vector of images cutouts part of the 3D `ml-cube`
  2. Train output vectors of:
     1. Images (autoencoding part)
     2. Spectra (prediction part, trained on the spectral data included in the `ml-cube`   
2. Fix the weights.
3. Train additional output vector Star Formation Rate.


The resulting network would then predict Images, Spectra, and Star formation rate vectors based on Image data only.


