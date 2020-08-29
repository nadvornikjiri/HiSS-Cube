import csv
import fitsio
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from pathlib import Path
from astropy.io import ascii
from scipy import interpolate
from scipy import ndimage
import cv2
import math

import numpy as np


class CubeUtils:

    def __init__(self, filter_curve_path, ccd_gain_path, ccd_dark_var_path):
        self.ccd_gain_config = self.read_config(ccd_gain_path)
        self.ccd_dark_variance_config = self.read_config(ccd_dark_var_path)

        self.filter_midpoints = {
            "u": 3551,
            "g": 4686,
            "r": 6166,
            "i": 7480,
            "z": 8932
        }

        self.transmission_params = {
            "u": {
                "z_point": 8.423e-9,
                "b_soft": 0.00000000014
            },
            "g": {
                "z_point": 5.055e-9,
                "b_soft": 0.00000000009
            },
            "r": {
                "z_point": 2.904e-9,
                "b_soft": 0.00000000012
            },
            "i": {
                "z_point": 1.967e-9,
                "b_soft": 0.00000000018
            },
            "z": {
                "z_point": 1.375e-9,
                "b_soft": 0.00000000074
            },
        }

        self.transmission_curves = {
            "u": dict(np.array(ascii.read("%s/SLOAN_SDSS.u.dat" % filter_curve_path)).tolist()),
            "g": dict(np.array(ascii.read("%s/SLOAN_SDSS.g.dat" % filter_curve_path)).tolist()),
            "r": dict(np.array(ascii.read("%s/SLOAN_SDSS.r.dat" % filter_curve_path)).tolist()),
            "i": dict(np.array(ascii.read("%s/SLOAN_SDSS.i.dat" % filter_curve_path)).tolist()),
            "z": dict(np.array(ascii.read("%s/SLOAN_SDSS.z.dat" % filter_curve_path)).tolist())
        }

        self.merged_transmission_curve = self.merge_transmission_curves_max(self.transmission_curves)

    def read_config(self, config_path):
        with open(config_path) as tsv_config:
            config = []
            reader = csv.DictReader(tsv_config, dialect='excel-tab')
            for row in reader:
                config.append(row)
        return config

    def get_ccd_gain(self, camcol, run, band):
        return self.get_config(self.ccd_gain_config, camcol, run, band)

    def get_dark_variance(self, camcol, run, band):
        return self.get_config(self.ccd_dark_variance_config, camcol, run, band)

    def get_config(self, config_obj, camcol, run, band):
        config_obj = list(filter(
            lambda r: r['camcol'] == str(camcol) and eval(str(run) + r['run']),
            config_obj))[0]  # there will be only one row matching the criteria
        return config_obj[band]

    def readFITSFiles(self, ):
        pathlist = Path("./data").glob('**/*.fits*')
        for path in pathlist:
            self.read_fits_file(path)

    def read_fits_file(self, filename):
        with fitsio.FITS(filename) as f:
            header = f[0].read_header()
            data = f[0].read()
            return [header, data, f]

    def read_spectrum(self, filename):
        with fitsio.FITS(filename) as hdul:
            data = hdul[1].read()
            fits_header = hdul[0].read_header()
            return data, fits_header

    # concats the transmission curves and where they overlap, takes maximum
    def merge_transmission_curves_max(self, *dicts):
        merged = {}
        for transmission in dicts:  # `dicts` is a tuple storing the input dictionaries
            for band, d in transmission.items():
                for key in d:
                    if key not in merged or d[key] > merged[key][1]:
                        merged[key] = [band, d[key]]
        return merged

    def get_multiple_resolution_spectrum(self, path, min_res, apply_transmission=True):
        multiple_resolution_cube = []
        data, fits_header = self.read_spectrum(path)

        wl_orig_res = np.power(10, data["loglam"])
        flux_mean_orig_res = data["flux"] * 1e-17
        with np.errstate(divide='ignore'):
            flux_sigma_orig_res = np.sqrt(np.divide(1, data["ivar"])) * 1e-17

        band, transmission_ratio, wl_trans = self.get_transmission_ratio(wl_orig_res)

        if apply_transmission:
            flux_mean_orig_res, flux_sigma_orig_res = self._get_filtered_spectrum(flux_mean_orig_res,
                                                                                  flux_sigma_orig_res,
                                                                                  transmission_ratio)

        multiple_resolution_cube.append({"res": len(wl_orig_res),
                                         "wl": wl_orig_res,
                                         "flux_mean": flux_mean_orig_res,
                                         "flux_sigma": flux_sigma_orig_res})
        self._append_lower_resolution_1D(multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                         wl_orig_res,
                                         min_res)
        return fits_header, multiple_resolution_cube

    def get_multiple_resolution_image(self, path, min_res):
        multiple_resolution_cube = []
        fits_header, img_orig_res_flux, img_orig_res_flux_sigma = self._get_image_with_errors(path)

        x_orig_res = img_orig_res_flux.shape[1]
        y_orig_res = img_orig_res_flux.shape[0]

        img_orig_res_flux = img_orig_res_flux * 3e-5 * 3.631e-6 / (self.filter_midpoints[fits_header["FILTER"]] ** 2)

        multiple_resolution_cube.append({"res": (x_orig_res, y_orig_res),
                                         "flux_mean": img_orig_res_flux,
                                         "flux_sigma": img_orig_res_flux_sigma})
        self._append_lower_resolution_2D(multiple_resolution_cube, img_orig_res_flux, img_orig_res_flux_sigma,
                                         min_res)
        return fits_header, multiple_resolution_cube

    def _get_image_with_errors(self, fitsPath):
        with fitsio.FITS(fitsPath) as f:
            fits_header = f[0].read_header()
            img = f[0].read()
            x_size = fits_header["NAXIS1"]
            y_size = fits_header["NAXIS2"]
            camcol = fits_header['CAMCOL']
            run = fits_header['RUN']
            band = fits_header['FILTER']
            allsky = f[2]['allsky'].read()[0]
            xinterp = f[2]['xinterp'].read()[0]
            yinterp = f[2]['yinterp'].read()[0]
            gain = float(self.get_ccd_gain(camcol, run, band))
            dark_variance = float(self.get_dark_variance(camcol, run, band))

            grid_x, grid_y = np.meshgrid(xinterp, yinterp)
            orig_x, orig_y = np.meshgrid(np.arange(256), np.arange(192))
            orig_coords = np.dstack((orig_y.ravel(), orig_x.ravel()))[0]

            simg = interpolate.griddata(orig_coords, allsky.ravel(), (grid_y, grid_x), method='nearest')
            calib = f[1].read()
            cimg = np.tile(calib, (y_size, 1))  # calibration image
            dn = img / cimg + simg  # data numbers
            dn_err = np.sqrt(dn / gain + dark_variance)  # data number errors
            img_err = dn_err * cimg  # image errors in nanomaggies
            return fits_header, img, img_err  # return calibrated image with errors in nanomaggies

    def _get_filtered_spectrum(self, flux, flux_sigma, transmission_ratio):
        # calibrating via filter curves
        photometric_observed_spectrum_flux = flux * transmission_ratio
        photometric_observed_spectrum_flux_sigma = flux_sigma * transmission_ratio
        return photometric_observed_spectrum_flux, photometric_observed_spectrum_flux_sigma

    def get_photometry_params(self, flux, wl):
        band, transmission_ratio, wl_trans = self.get_transmission_ratio(wl)
        band_wl_limits = {}
        for wavelength, band in zip(wl_trans, band):
            if band not in band_wl_limits or band_wl_limits[band] < wavelength:
                band_wl_limits[band] = wavelength
        zero_points = np.empty(flux.shape)
        softenings = np.empty(flux.shape)
        for i, wavelength in np.ndenumerate(wl):
            for band, band_limit in band_wl_limits.items():
                if wavelength < band_limit:
                    zero_points[i[0]] = self.transmission_params[band]["z_point"]
                    softenings[i[0]] = self.transmission_params[band]["b_soft"]
                    break
        return transmission_ratio, zero_points, softenings

    def get_transmission_ratio(self, wl):
        wl_trans, band_ratio = zip(*list(self.merged_transmission_curve.items()))
        band, ratio = zip(*list(band_ratio))
        transmission_ratio = np.interp(wl,
                                       wl_trans,
                                       ratio)
        return band, transmission_ratio, wl_trans

    def _append_lower_resolution_1D(self, multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                    wl_orig_res,
                                    min_res):
        # smoothing the curve with gaussian kernel to simulate observation in lower resolution (assuming gaussian-distributed errors)
        gauss_kernel = Gaussian1DKernel(stddev=2)
        smoothed_flux_mean_orig_res = convolve(flux_mean_orig_res, gauss_kernel)
        # producing lower resolution
        wl_lower_res = np.linspace(wl_orig_res[0],  # from minimum wavelength
                                   wl_orig_res[-1],  # to maximum wavelength
                                   int(wl_orig_res.size / 2))  # every second coordinate.
        flux_lower_res = np.interp(wl_lower_res,  # interpolation takes every second coordinate
                                   wl_orig_res,
                                   smoothed_flux_mean_orig_res)
        flux_sigma_every_second_point = np.interp(wl_lower_res,  # interpolation takes every second coordinate
                                                  wl_orig_res,
                                                  flux_sigma_orig_res)
        flux_sigma_lower_res = np.divide(flux_sigma_every_second_point,
                                         2)  # resampling to every second coordinate in 1D divides the variance by 2

        multiple_resolution_cube.append({"res": len(wl_lower_res),
                                         "wl": wl_lower_res,
                                         "flux_mean": flux_lower_res,
                                         "flux_sigma": flux_sigma_lower_res})

        if not len(wl_lower_res) / 2 < min_res:
            self._append_lower_resolution_1D(multiple_resolution_cube, flux_lower_res, flux_sigma_lower_res,
                                             wl_lower_res,
                                             min_res)

    def _append_lower_resolution_2D(self, multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                    min_res):
        # producing lower resolution
        flux_lower_res = cv2.resize(flux_mean_orig_res, dsize=(int(flux_mean_orig_res.shape[1] / 2),
                                                               int(flux_mean_orig_res.shape[0] / 2)),
                                    interpolation=cv2.INTER_CUBIC)
        flux_sigma_every_second_point = cv2.resize(flux_sigma_orig_res,
                                                   dsize=(int(flux_mean_orig_res.shape[1] / 2),
                                                          int(flux_mean_orig_res.shape[0] / 2)),
                                                   interpolation=cv2.INTER_CUBIC)
        flux_sigma_lower_res = np.divide(flux_sigma_every_second_point,
                                         4)  # resampling to every second coordinate in 2D divides the variance by 2x2
        x_lower_res = flux_lower_res.shape[1]
        y_lower_res = flux_lower_res.shape[0]

        multiple_resolution_cube.append({"res": (flux_lower_res.shape[1], flux_lower_res.shape[0]),
                                         "flux_mean": flux_lower_res,
                                         "flux_sigma": flux_sigma_lower_res})

        if not (x_lower_res / 2 < min_res or y_lower_res / 2 < min_res):
            self._append_lower_resolution_2D(multiple_resolution_cube, flux_lower_res, flux_sigma_lower_res,
                                             min_res)
