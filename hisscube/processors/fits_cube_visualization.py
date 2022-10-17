from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.time import Time

from hisscube.processors.cube_visualization import VisualizationProcessor
from hisscube.utils import astrometry
from hisscube.utils.astrometry import NoCoverageFoundError, is_cutout_whole


class FITSProcessor(VisualizationProcessor):

    def __init__(self, config, photometry, spectra_path, image_path, spectra_regex="*.fits", image_regex="*.fits*"):
        super().__init__(config)
        self.photometry = photometry
        self.spectra_path = spectra_path
        self.spectra_regex = spectra_regex
        self.image_regex = image_regex
        self.image_path = image_path
        self.image_metadata = None

    def get_spectral_cube_from_orig_for_res(self, res_idx=0):
        self.spectral_cube = np.empty((self.config.INIT_ARRAY_SIZE,), dtype=self.array_type)
        self.output_res = res_idx
        self.get_spectral_cube_from_orig()
        truncated_cube = self.spectral_cube[:self.output_counter]
        self.spectral_cube = truncated_cube
        return self.spectral_cube

    def get_spectral_cube_from_orig(self):
        for spectrum_path in Path(self.spectra_path).rglob(self.spectra_regex):
            print("writing: %s" % spectrum_path)
            with fits.open(spectrum_path, memmap=True) as hdul:
                self.read_spectral_dataset(spectrum_path, hdul)
        return

    def read_spectral_dataset(self, spectrum_path, spectrum_hdul):
        self.metadata = spectrum_hdul[0].header
        spectrum_fits_name = str(spectrum_path).split('/')[-1]
        spectrum_5d = self.get_pixels_from_spectrum(spectrum_hdul[0].header, spectrum_hdul, spectrum_fits_name)
        self._resize_output_if_necessary(spectrum_5d)
        self.spectral_cube[self.output_counter:self.output_counter + spectrum_5d.shape[0]] = np.reshape(spectrum_5d,
                                                                                                        spectrum_5d.shape)
        self.output_counter += spectrum_5d.shape[0]

        for image_path in Path(self.image_path).rglob(self.image_regex):
            try:
                cutout_region, image_header = self.get_region_from_fits(spectrum_hdul[0].header, image_path)
            except NoCoverageFoundError:
                continue
            print("reading region %s dataset: %s" % (cutout_region.shape, image_path))
            try:
                image_5d = self.get_pixels_from_image_cutout(self.metadata, image_header, self.output_res,
                                                             cutout_region, spectrum_path, image_path)
                self._resize_output_if_necessary(image_5d)
                self.spectral_cube[self.output_counter:self.output_counter + image_5d.shape[0]] = image_5d
                self.output_counter += image_5d.shape[0]
            except ValueError as e:
                self.logger.error("Could not process region for %s, message: %s" % (spectrum_path, str(e)))

    def _resize_output_if_necessary(self, spectrum_5d):
        if self.output_counter + spectrum_5d.shape[0] > self.spectral_cube.shape[0]:
            self.spectral_cube.resize((self.spectral_cube.shape[0] * 2, 1), refcheck=False)

    def get_pixels_from_spectrum(self, spectrum_header, spectrum_hdul, spectrum_fits_name):
        res = int(len(spectrum_hdul[1].data))
        ra, dec = spectrum_header["PLUG_RA"], spectrum_header["PLUG_DEC"]
        spectrum_part = self.get_spectral_data(spectrum_hdul)
        return self._get_table_pixels_from_spectrum_generic(dec, ra, res, spectrum_fits_name, spectrum_part)

    def get_pixels_from_image_cutout(self, orig_spectrum_header, orig_image_header, res_idx, image_region,
                                     spectrum_path, image_path):
        image_region = np.dstack((image_region, np.zeros(image_region.shape)))
        time_attr = orig_image_header["DATE-OBS"]
        try:
            time = Time(time_attr, format='isot', scale='tai').mjd
        except ValueError:
            time = Time(datetime.strptime(time_attr, "%d/%m/%y")).mjd
        wl = str(self.photometry.filter_midpoints[orig_image_header["filter"]])

        w = wcs.WCS(orig_image_header)
        image_size = np.array((orig_image_header["NAXIS2"], orig_image_header["NAXIS1"]))
        cutout_bounds = astrometry.process_cutout_bounds(w, image_size, orig_spectrum_header,
                                                         self.config.IMAGE_CUTOUT_SIZE)
        return self._get_table_image_pixels_from_cutout_bounds(cutout_bounds, image_path, image_region, spectrum_path,
                                                               time,
                                                               w, wl)

    def get_region_from_fits(self, spectrum_header, image_path):
        with fits.open(image_path, memmap=True) as image_hdul:
            image_header = image_hdul[0].header
            w = wcs.WCS(image_header)
            image_size = np.array((image_header["NAXIS2"], image_header["NAXIS1"]))
            cutout_bounds = astrometry.process_cutout_bounds(w, image_size, spectrum_header,
                                                             self.config.IMAGE_CUTOUT_SIZE)
            image_data = image_hdul[0].data
            if not is_cutout_whole(cutout_bounds, image_data):
                raise NoCoverageFoundError
            return image_data[cutout_bounds[0][1][1]:cutout_bounds[1][1][1],
                   cutout_bounds[1][0][0]:cutout_bounds[1][1][0]], image_header

    def get_spectral_data(self, spectrum_hdul):
        data = spectrum_hdul[1].data

        wl_orig_res = np.power(10, data["loglam"])
        flux_mean_orig_res = data["flux"] * 1e-17
        with np.errstate(divide='ignore'):
            flux_sigma_orig_res = np.sqrt(np.divide(1, data["ivar"])) * 1e-17

        return np.dstack((wl_orig_res, flux_mean_orig_res, flux_sigma_orig_res))[0]
