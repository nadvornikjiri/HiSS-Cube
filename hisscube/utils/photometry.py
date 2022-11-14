import csv
import pathlib

import cv2
import numpy as np
import spectres as spectres
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.io import ascii
from astropy.io import fits
from scipy import ndimage


class Photometry:

    def __init__(self, fits_mem_map=True):
        """
        Initializes the parameters of SDSS photometry from the ccd_gain and ccd_dark_var files. The transmission curves
        are initialized directly here.
        Parameters
        ----------
        """
        lib_path = pathlib.Path(__file__).parent.absolute()
        filter_curve_path = "%s/../../config/SDSS_Bands" % lib_path
        ccd_gain_path = "%s/../../config/ccd_gain.tsv" % lib_path
        ccd_dark_var_path = "%s/../../config/ccd_dark_variance.tsv" % lib_path
        self.ccd_gain_config = self._read_config(ccd_gain_path)
        self.ccd_dark_variance_config = self._read_config(ccd_dark_var_path)
        self.fits_mem_map = fits_mem_map

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

        self.merged_transmission_curve = self._merge_transmission_curves_max(self.transmission_curves)

    def get_image_wl(self, metadata):
        return self.filter_midpoints[metadata["FILTER"]]

    def get_midpoints(self):
        for val in self.filter_midpoints.values():
            yield val

    def get_multiple_resolution_spectrum(self, path, spec_zoom_cnt, apply_rebin=False, rebin_min=0, rebin_max=0,
                                         rebin_samples=0, apply_transmission=True):
        """
        Constructs lower resolutions for a spectrum, optionally with applied transmission curve to simulate
        observation through photometry filter. Can be called recursively.

        Parameters
        ----------
        rebin_samples
        rebin_min           Minimum wavelength of the to which the spectra will be rebinned.
        rebin_max           Maximum wavelength of the to which the spectra will be rebinned.
        apply_rebin         Flag specifying whether rebinning will be applied.
        path                String
        spec_zoom_cnt       int
        apply_transmission  Bool

        Returns             (Dictionary, numpy array)
        -------

        """
        multiple_resolution_cube = []
        data, fits_header = self.read_spectrum(path)

        wl_orig_res = np.power(10, data["loglam"])
        flux_mean_orig_res = data["flux"] * 1e-17
        with np.errstate(divide='ignore'):
            flux_sigma_orig_res = np.sqrt(np.divide(1, data["ivar"])) * 1e-17

        band, transmission_ratio, wl_trans = self._get_transmission_ratio(wl_orig_res)

        if apply_transmission:
            flux_mean_orig_res, flux_sigma_orig_res = self._get_filtered_spectrum(flux_mean_orig_res,
                                                                                  flux_sigma_orig_res,
                                                                                  transmission_ratio)
        if apply_rebin:
            wl_orig_res, flux_mean_orig_res, flux_sigma_orig_res = self._get_rebinned_spectrum(wl_orig_res,
                                                                                               flux_sigma_orig_res,
                                                                                               flux_mean_orig_res,
                                                                                               rebin_min,
                                                                                               rebin_max, rebin_samples)

        multiple_resolution_cube.append({"zoom_idx": len(wl_orig_res),
                                         "wl": wl_orig_res,
                                         "flux_mean": flux_mean_orig_res,
                                         "flux_sigma": flux_sigma_orig_res})
        if spec_zoom_cnt > 0:
            spec_zoom_cnt -= 1
            self._append_lower_resolution_1d(multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                             wl_orig_res,
                                             spec_zoom_cnt)
        return fits_header, multiple_resolution_cube

    def get_multiple_resolution_image(self, path, img_zoom_cnt):
        """
        Constructs multiple resolutions for the input image. Can be called recursively.

        Parameters
        ----------
        path    String
        min_res int

        Returns (int, numpy array)
        -------

        """
        multiple_resolution_cube = []
        fits_header, img_orig_res_flux, img_orig_res_flux_sigma = self._get_image_with_errors(path)

        x_orig_res = img_orig_res_flux.shape[1]
        y_orig_res = img_orig_res_flux.shape[0]

        img_orig_res_flux = self.mag_to_flux(fits_header, img_orig_res_flux)
        img_orig_res_flux_sigma = self.mag_to_flux(fits_header, img_orig_res_flux_sigma)

        multiple_resolution_cube.append({"zoom_idx": (x_orig_res, y_orig_res),
                                         "flux_mean": img_orig_res_flux,
                                         "flux_sigma": img_orig_res_flux_sigma})
        if img_zoom_cnt > 0:
            img_zoom_cnt -= 1
            self._append_lower_resolution_2d(multiple_resolution_cube, img_orig_res_flux, img_orig_res_flux_sigma,
                                             img_zoom_cnt)
        return fits_header, multiple_resolution_cube

    def mag_to_flux(self, fits_header, img_mag):
        return img_mag * 3.631e-6 * 2.99792458e-5 / (self.filter_midpoints[fits_header["FILTER"]] ** 2)

    def get_photometry_params(self, flux, wl):
        band, transmission_ratio, wl_trans = self._get_transmission_ratio(wl)
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

    def _get_image_with_errors(self, fits_path):
        """
        Calculates image uncertainties, see the
        https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html for the algorith
        description and returns them along with image header and original data.

        Parameters
        ----------
        fitsPath    String

        Returns     (Dictionary, numpy array, numpy array)
        -------

        """
        with fits.open(fits_path, memmap=self.fits_mem_map) as f:
            fits_header = f[0].header
            img = f[0].data
            y_size = fits_header["NAXIS2"]
            camcol = fits_header['CAMCOL']
            run = fits_header['RUN']
            band = fits_header['FILTER']
            allsky = f[2].data.field('allsky')[0]
            xinterp = f[2].data.field('xinterp')[0]
            yinterp = f[2].data.field('yinterp')[0]
            gain = float(self.get_ccd_gain(camcol, run, band))
            dark_variance = float(self.get_dark_variance(camcol, run, band))

            grid_x, grid_y = np.meshgrid(xinterp, yinterp, copy=False)
            simg = ndimage.map_coordinates(allsky, (grid_y, grid_x), order=1, mode="nearest")
            calib = f[1].data
            cimg = np.tile(calib, (y_size, 1))  # calibration image
            dn = img / cimg + simg  # data numbers
            dn_err = np.sqrt(dn / gain + dark_variance)  # data number errors
            img_err = dn_err * cimg  # image errors in nanomaggies
            return fits_header, np.ascontiguousarray(img), np.ascontiguousarray(
                img_err)  # return calibrated image with errors in nanomaggies

    def get_ccd_gain(self, camcol, run, band):
        return self._get_config(self.ccd_gain_config, camcol, run, band)

    def get_dark_variance(self, camcol, run, band):
        return self._get_config(self.ccd_dark_variance_config, camcol, run, band)

    def read_fits_file(self, filename):
        with fits.open(filename, memmap=self.fits_mem_map) as f:
            header = f[0].header
            data = f[0].data
            return [header, data, f]

    def read_spectrum(self, filename):
        with fits.open(filename, memmap=self.fits_mem_map) as hdul:
            fits_header = hdul[0].header
            data = hdul[1].data
            return data, fits_header

    def _get_transmission_ratio(self, wl):
        wl_trans, band_ratio = zip(*list(self.merged_transmission_curve.items()))
        band, ratio = zip(*list(band_ratio))
        transmission_ratio = np.interp(wl,
                                       wl_trans,
                                       ratio)
        return band, transmission_ratio, wl_trans

    def _append_lower_resolution_1d(self, multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                    wl_orig_res,
                                    spec_zoom_cnt):
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

        multiple_resolution_cube.append({"zoom_idx": len(wl_lower_res),
                                         "wl": wl_lower_res,
                                         "flux_mean": flux_lower_res,
                                         "flux_sigma": flux_sigma_lower_res})

        if spec_zoom_cnt > 0:
            spec_zoom_cnt -= 1
            self._append_lower_resolution_1d(multiple_resolution_cube, flux_lower_res, flux_sigma_lower_res,
                                             wl_lower_res,
                                             spec_zoom_cnt)

    def _append_lower_resolution_2d(self, multiple_resolution_cube, flux_mean_orig_res, flux_sigma_orig_res,
                                    res_zoom):
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

        multiple_resolution_cube.append({"zoom_idx": (flux_lower_res.shape[1], flux_lower_res.shape[0]),
                                         "flux_mean": flux_lower_res,
                                         "flux_sigma": flux_sigma_lower_res})

        if res_zoom > 0:
            res_zoom -= 1
            self._append_lower_resolution_2d(multiple_resolution_cube, flux_lower_res, flux_sigma_lower_res,
                                             res_zoom)

    @staticmethod
    def _get_filtered_spectrum(flux, flux_sigma, transmission_ratio):  # calibrating via filter curves
        photometric_observed_spectrum_flux = flux * transmission_ratio
        photometric_observed_spectrum_flux_sigma = flux_sigma * transmission_ratio
        return photometric_observed_spectrum_flux, photometric_observed_spectrum_flux_sigma

    @staticmethod
    def _read_config(config_path):
        with open(config_path) as tsv_config:
            config = []
            reader = csv.DictReader(tsv_config, dialect='excel-tab')
            for row in reader:
                config.append(row)
        return config

    @staticmethod
    def _merge_transmission_curves_max(*dicts):
        """
        Concats the transmission curves and where they overlap, takes maximum of both curves.
        Parameters
        ----------
        dicts   *[Dictionary]

        Returns Dictionary
        -------

        """
        merged = {}
        for transmission in dicts:  # `dicts` is a tuple storing the input dictionaries
            for band, d in transmission.items():
                for key in d:
                    if key not in merged or d[key] > merged[key][1]:
                        merged[key] = [band, d[key]]
        return merged

    @staticmethod
    def _get_rebinned_spectrum(orig_wavs, flux_sigma_orig_res, flux_mean_orig_res, rebin_min, rebin_max,
                               rebin_samples):
        new_wavs = np.linspace(rebin_min, rebin_max, rebin_samples)
        rebinned_flux, rebinned_sigma = spectres.spectres(new_wavs, orig_wavs, flux_mean_orig_res, flux_sigma_orig_res,
                                                          verbose=False)

        return new_wavs, rebinned_flux, rebinned_sigma

    @staticmethod
    def _get_config(config_obj, camcol, run, band):
        config_obj = list(filter(
            lambda r: r['camcol'] == str(camcol) and eval(str(run) + r['run']),
            config_obj))[0]  # there will be only one row matching the criteria
        return config_obj[band]
