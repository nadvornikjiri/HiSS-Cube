# SDSS Uncertainty Image+Spectral Cube construction
In this notebook we show the methods in detail how do we construct the uncertainty cube from SDSS Spectra and images. The data used are DR14 images and spectra from Stripe 82.


```python
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from hisscube.utils.photometry import Photometry
from hisscube.utils.config import Config



from astropy.time import Time
from pathlib import Path
from matplotlib.colors import LogNorm
from importlib import reload
import h5py
import fitsio
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import timeit
#%matplotlib notebookQQ


photometry = Photometry(Config())

test_image = "../data/raw/images/301/4797/1/frame-g-004797-1-0019.fits.bz2"    
test_spectrum = "../data/raw/spectra/blue_star.fits"

test_images_small = "frame-*-002820-3-0122.fits.bz2"
test_spectra_small = "*.fits"
test_images = "*.fits.bz2"
test_spectra = "*.fits"

img_header, img_data, file = photometry.read_fits_file(test_image)
spec_data, spec_header = photometry.read_spectrum(test_spectrum)
#print(img_header)
t = Time(img_header["DATE-OBS"], format='isot', scale='tai')

font = {'weight' : 'bold',
        'size'   : 24}

plt.rc('font', **font)

```

# Extracting uncertainty for image measurements.
In this step we extract the uncertainty for an SDSS image and plot all the intermediate by-products below. Steps involved:
1. Reading sky background from the FITS extension.
2. Interpolating the sky background to the image resolution.
3. Reading calibration vector and extending it to whole image (1D calibration vector works because of the way how SDSS continuously reads out the image).
4. Subtracting sky background and image calibration from the calibrated image, thus producing the Image before calibration in Detector Data Numbers.
5. Showing how we can calculate number of electrons from the data numbers - we don't need these for our error calculation.
6. Calculating Errors in the Data Numbers
7. Converting the Data Number errors to Nanomaggies (this is what we store in the uncertainty cube)
8. Showing the error to signal ratio - this plot demonstrates what we would expect - for brighter areas of the image we are more certain about the measurements than for dark areas.


```python
from numpy import inf
from matplotlib import colors

def draw_plots(img, img_err, allsky, simg, cimg, dn, nelec, dn_err, err_to_signal):
    plt.figure(figsize=(30,20))
    plt.imshow(allsky, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Sky background small")

    plt.figure(figsize=(30,20))
    plt.imshow(simg, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Sky background")

    plt.figure(figsize=(30,20))
    plt.imshow(img, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Calibrated image")

    plt.figure(figsize=(30,20))
    plt.imshow(dn, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Image before calibration in Data Numbers")

    plt.figure(figsize=(30,20))
    plt.imshow(nelec, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Number of photo-electrons")

    plt.figure(figsize=(30,20))
    plt.imshow(dn_err, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Errors in the Data Numbers")

    plt.figure(figsize=(30,20))
    plt.imshow(img_err, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Errors in nanomaggies")
    
    
    plt.figure(figsize=(30,20))
    plt.imshow(err_to_signal, cmap='afmhot', norm=LogNorm())
    plt.colorbar()
    plt.title("Uncertainty in signal - Uncertainty to signal ratio")
    

def get_image_with_errors(fitsPath):
    with fitsio.FITS(fitsPath) as f:
        start_time = timeit.default_timer()
        fits_header = f[0].read_header()
        img = f[0].read()
        x_size = fits_header["NAXIS1"]
        y_size = fits_header["NAXIS2"]
        camcol = fits_header['CAMCOL']
        run = fits_header['run']
        band = fits_header['filter']    
        allsky = f[2]['allsky'].read()[0]
        xinterp = f[2]['xinterp'].read()[0]
        yinterp = f[2]['yinterp'].read()[0]
        
        elapsed1 = timeit.default_timer() - start_time
        
        #print(elapsed1)


        gain = float(photometry.get_ccd_gain(camcol, run, band))
        darkVariance = float(photometry.get_dark_variance(camcol, run, band))
        elapsed2 = timeit.default_timer() - start_time
        
        #print(elapsed2)
        
        grid_x, grid_y = np.meshgrid(xinterp, yinterp, copy=False)
        elapsed3 = timeit.default_timer() - start_time
        
        #print(elapsed3)
        #interpolating sky background from 256x196 small image included in SDSS
        simg = ndimage.map_coordinates(allsky, (grid_y, grid_x), order = 1, mode="nearest")
        
        elapsed4 = timeit.default_timer() - start_time
        #print(elapsed4)

        calib = f[1].read()

        cimg = np.tile(calib, (y_size, 1)) #calibration image constructed from calibration vector

        dn = img / cimg + simg #data numbers detected originally by the detector 
        

        nelec = dn*gain #number of electrons

        dn_err = np.sqrt(dn/gain + darkVariance) #errors in data numbers (sigma)

        img_err= dn_err*cimg  #errors in nanomaggies (sigma)
        
        img_phys_flux = img

        err_to_signal = img_err/ img
        err_to_signal[err_to_signal == inf] = 0
        elapsed5 = timeit.default_timer() - start_time
        #print(elapsed5)
        
        

        
        return img, img_err, allsky, simg, cimg, dn, nelec, dn_err, err_to_signal
    
    
img, img_err, allsky, simg, cimg, dn, nelec, dn_err, err_to_signal = get_image_with_errors(test_image)

draw_plots(img, img_err, allsky, simg, cimg, dn, nelec, dn_err, err_to_signal)
```

    /tmp/ipykernel_33574/331323260.py:97: RuntimeWarning: divide by zero encountered in divide
      err_to_signal = img_err/ img



    
![png](SDSS_cube_files/SDSS_cube_3_1.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_2.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_3.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_4.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_5.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_6.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_7.png)
    



    
![png](SDSS_cube_files/SDSS_cube_3_8.png)
    


# Extracting spectrum errors
For the spectrum the SDSS has made life much more easy for us as the inverse variance of the measurement is already stored in the fits files. We just need to convert both to nano-maggies to be able to compare it with image photometry, as it is in '1E-17 erg/cm^2/s/Ang' units.


```python
def plot_spec(fits_path):
    with fitsio.FITS(fits_path) as hdul:
        data = hdul[1].read()
        flux = data["flux"] * 1e-17
        sigma =  np.sqrt(np.divide(1,data["ivar"])) * 1e-17
        wl = np.power(10, data["loglam"])
        print(data["loglam"])
        plt.figure(figsize=(30,20))
        ax = plt.axes(xlabel="Wavelength [Angstrem]", ylabel="Flux [ erg/cm^2/s/Ang]")
        ax.plot(wl, flux)
        ax.fill_between(wl, flux - sigma, flux + sigma, color="orange")
        return wl, flux, sigma

spec_wl, spec_flux, spec_sigma = plot_spec(test_spectrum)
```

    [3.5804 3.5805 3.5806 ... 3.9643 3.9644 3.9645]



    
![png](SDSS_cube_files/SDSS_cube_5_1.png)
    


# Getting Filter transmission curves
We are not able to compare directly spectroscopic measurements with photometric ones. However, we can approximate this by applying the filter transmission curves to the spectrum and thus simulate values that would be measured to photon counts reduced by filter transmission function.

Below we show how the transmission curve looks like throught the whole spectrograph range. We take the transmission curves for individual filters and take the maximum transmission ratio for each wavelength, which essentially gets rid of the filter overlaps. We can do this because the image coordinates are the mid_points of those filters which are no in the overlapping regions anyway, so there will be no ambiguity to which filter should be applied in those regions.


```python
u = photometry.transmission_curves["u"]
g = photometry.transmission_curves["g"]
r = photometry.transmission_curves["r"]
i = photometry.transmission_curves["i"]
z = photometry.transmission_curves["z"]

merged_transmission_curve = photometry.merged_transmission_curve

wl, band_ratio = zip(*list(merged_transmission_curve.items()))
band, ratio = zip(*list(band_ratio))
u_wl, u_ratio = zip(*list(u.items()))
g_wl, g_ratio = zip(*list(g.items()))
r_wl, r_ratio = zip(*list(r.items()))
i_wl, i_ratio = zip(*list(i.items()))
z_wl, z_ratio = zip(*list(z.items()))

plt.rc('font', **font)

plt.figure(figsize=(30,20))
ax_u = plt.axes(xlabel="Wavelength [Angstrem]", ylabel="Transmission ratio")
ax_u = ax_u.plot(u_wl, u_ratio, color="blue")
ax_g = plt.plot(g_wl, g_ratio, color="green")
ax_r = plt.plot(r_wl, r_ratio, color="red")
ax_i = plt.plot(i_wl, i_ratio, color="#BF0000")
ax_z = plt.plot(z_wl, z_ratio, color="grey")
plt.title("Individual transmission curves for UGRIZ filters.")

plt.figure(figsize=(30,20))
ax_merged = plt.axes(xlabel="Wavelength [Angstrem]", ylabel="Transmission ratio")
ax_merged = plt.plot(wl, ratio)
plt.title("Merged transmission curve.")

print()
```

    



    
![png](SDSS_cube_files/SDSS_cube_7_1.png)
    



    
![png](SDSS_cube_files/SDSS_cube_7_2.png)
    


# Applied transmission curve
This is how the spectrum looks like when the photometric transmission curve is applied.
Procedure:
1. Multiply magnitude in each wavelength with the transmission curve ratio in that wavelength.


```python
transmission_ratio, zero_point, softening = photometry.get_photometry_params(spec_flux, spec_wl)

photometric_observed_spectrum_flux = spec_flux * transmission_ratio
photometric_observed_spectrum_flux_sigma =  spec_sigma * transmission_ratio

plt.figure(figsize=(30,20))
ax1 = plt.axes(xlabel="Wavelength [Angstrem]", ylabel="Flux [erg/cm^2/s/Ang]")
ax1.plot(spec_wl, photometric_observed_spectrum_flux)
ax1.fill_between(spec_wl, 
                photometric_observed_spectrum_flux - photometric_observed_spectrum_flux_sigma, 
               photometric_observed_spectrum_flux + photometric_observed_spectrum_flux_sigma, 
              color="orange")
plt.title("Spectrum with applied photometric transmissions")
print()
```

    



    
![png](SDSS_cube_files/SDSS_cube_9_1.png)
    



```python
spec_zoom_cnt = 2

def plot_cube(res_cube, trans_txt, y_unit):
    for spec in res_cube:
        wl = spec["wl"]
        flux = spec["flux_mean"]
        flux[flux == 0 ] = 'nan'
        flux_sigma = spec["flux_sigma"]
        flux_sigma[flux_sigma == 0 ] = 'nan'
        plt.figure(figsize=(30,20))
        ax = plt.axes(xlabel="Wavelength [Angstrem]", ylabel=y_unit)
        ax.plot(wl, flux)
        ax.fill_between(wl, 
                        flux - flux_sigma, 
                        flux + flux_sigma, 
                        color="orange")
        plt.title("%s, resolution: %s" %(trans_txt, spec["zoom_idx"]))
        
        bins = range(0,len(wl))
        
        plt.figure(figsize=(30,20))
        ax = plt.axes(xlabel="Bin index", ylabel=y_unit)
        ax.plot(bins, flux)
        ax.fill_between(bins, 
                        flux - flux_sigma, 
                        flux + flux_sigma, 
                        color="orange")
        plt.title("%s, bins, resolution: %s" %(trans_txt, spec["zoom_idx"]))

fits_header, spec_cube = photometry.get_multiple_resolution_spectrum(test_spectrum, spec_zoom_cnt,
                                                                     apply_rebin=False, rebin_min=0, rebin_max=0,
                                                                     rebin_samples=0, apply_transmission=False)
fits_header, spec_cube_binned = photometry.get_multiple_resolution_spectrum(test_spectrum, spec_zoom_cnt,
                                                                            apply_rebin=True, rebin_min=3600, rebin_max=10400,
                                                                            rebin_samples=4620, apply_transmission=False)
plot_cube(spec_cube, "Spectrum", "Flux density [erg/cm^2/s/Ang]")
plot_cube(spec_cube_binned, "Spectrum binned", 
          "Flux density [erg/cm^2/s/Ang])")

```


    
![png](SDSS_cube_files/SDSS_cube_10_0.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_1.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_2.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_3.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_4.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_5.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_6.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_7.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_8.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_9.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_10.png)
    



    
![png](SDSS_cube_files/SDSS_cube_10_11.png)
    


# Constructing multiple resolution spectrum
Here we show how the multiple resolution spectrum is constructed, with the same algorithm as above:
1. In Fluxes without photometric transmission applied
2. In Magnitudes without photometric transmission applied
3. In Magnitudes with photomoetric transmission applied

For each of these, we construct the lower resolutions down to the MIN_RES setting (Each lower resolution is half of the upper one, meaning we get spectral resolutions e.g. (4620, 2310, 1155, etc.)


```python
spec_zoom_cnt = 4

def plot_cube(res_cube, trans_txt, y_unit):
    for spec in res_cube:
        wl = spec["wl"]
        flux = spec["flux_mean"]
        flux_sigma = spec["flux_sigma"]
        plt.figure(figsize=(30,20))
        ax = plt.axes(xlabel="Wavelength [Angstrem]", ylabel=y_unit)
        ax.plot(wl, flux)
        ax.fill_between(wl, 
                        flux - flux_sigma, 
                        flux + flux_sigma, 
                        color="orange")
        plt.title("%s, resolution: %s" %(trans_txt, spec["zoom_idx"]))

fits_header, spec_cube = photometry.get_multiple_resolution_spectrum(test_spectrum,
                                                                     spec_zoom_cnt,
                                                                     apply_transmission=False)
fits_header, spec_cube_with_transmission = photometry.get_multiple_resolution_spectrum(test_spectrum,
                                                                                       spec_zoom_cnt,
                                                                                       apply_transmission=True)
plot_cube(spec_cube, "Spectrum", "Flux [erg/cm^2/s/Ang]")
plot_cube(spec_cube_with_transmission, "Spectrum, transmission curve applied", 
          "Flux [erg/cm^2/s/Ang])")      
```


    
![png](SDSS_cube_files/SDSS_cube_12_0.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_1.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_2.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_3.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_4.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_5.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_6.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_7.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_8.png)
    



    
![png](SDSS_cube_files/SDSS_cube_12_9.png)
    


# In this cell we show how we construct the lower resolutions for images.
We use bilinear interpolation to construct the lower resolutions, both for image measurements and their sigmas, while for each lower resolution, the sigma is halved


```python
img_zoom_cnt = 4


def plot_image_cube(res_cube):
    for img in res_cube:
        plt.figure(figsize=(30,20))
        plt.imshow(img["flux_mean"], cmap='afmhot',  norm=LogNorm())
        plt.colorbar()
        plt.title("Image flux density for res %s" %(str(img["zoom_idx"])))
        #plt.figure(figsize=(30,20))
        #plt.imshow(img["flux_sigma"], cmap='gray')
        #plt.colorbar()
        #plt.title("Image variance for res %s" %(str(img["res"])))
       
        

fits_header, image_cube = photometry.get_multiple_resolution_image(test_image, img_zoom_cnt)

plot_image_cube(image_cube)



        
        
        
```


    
![png](SDSS_cube_files/SDSS_cube_14_0.png)
    



    
![png](SDSS_cube_files/SDSS_cube_14_1.png)
    



    
![png](SDSS_cube_files/SDSS_cube_14_2.png)
    



    
![png](SDSS_cube_files/SDSS_cube_14_3.png)
    



    
![png](SDSS_cube_files/SDSS_cube_14_4.png)
    


# Constructing database in HDF5 file
In this step we load the images and spectra, with their preprocessing applied above, into HDF5 file. Effectively we combine independently measured spectra and images into one sparse spectral cube that contains all the datapoints with comparable values.

Steps involved:
1. Create indexing structure to enable spatial, temporal, spectral and resolution-wise searching of the datasets.
2. Preprocess the images and spectra to construct lower resolutions and uncertainties.


```python
import warnings
from astropy.utils.exceptions import AstropyWarning

h5_output_path = "../data/processed/galaxy_small.h5"
input_folder = "../data/raw/galaxy_small"

warnings.simplefilter('ignore', category=AstropyWarning)
command = "python %s/hisscube.py %s %s create" % (module_path, input_folder, h5_output_path)
print("Running command: %s" % command)
!mpiexec -n 8 $command
```

    Running command: python /home/caucau/SDSSCube/hisscube.py ../data/raw/galaxy_small ../data/processed/galaxy_small.h5 create
    
    Image headers: 185it [00:00, 1155.87it/s]
    
    Spectra headers: 11it [00:00, 271.28it/s]
    
    Image metadata: 100%|██████████| 185/185 [00:00<00:00, 250.21it/s]
    
    Spectra metadata: 100%|██████████| 11/11 [00:00<00:00, 479.95it/s]
    
    Image data:   0%|          | 0/185 [00:00<?, ?it/s][A^C

