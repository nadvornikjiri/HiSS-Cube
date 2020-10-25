import logging
from datetime import datetime

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import wcs
from astropy.io.votable import from_table, writeto
from astropy.table import QTable

from scripts import astrometry, SDSSCubeHandler
from astropy.time import Time
import os
from scripts import SDSSCubeHandler as h5
from scripts.astrometry import get_optimized_wcs


class SDSSCubeReader(h5.SDSSCubeHandler):

    def __init__(self, h5_file, cube_utils):
        super(SDSSCubeReader, self).__init__(h5_file, cube_utils)
        if self.INCLUDE_FITS_PATH:
            self.array_type = np.dtype('i8, f8, f8, f8, f8, f8, f8, f8, f8, S32, S32')
        else:
            self.array_type = np.dtype('i8, f8, f8, f8, f8, f8, f8')
        self.spectral_cube = np.empty((self.INIT_ARRAY_SIZE, 1), dtype=self.array_type)
        self.OUTPUT_HEAL_ORDER = 19
        self.output_res = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_counter = 0

    def get_spectral_cube_for_res(self, resolution):
        self.output_res = resolution
        self.get_spectral_cube(self.f)
        truncated_cube = self.spectral_cube[:self.output_counter]
        self.spectral_cube = truncated_cube
        return self.spectral_cube

    def get_spectral_cube(self, h5_parent):
        if "mime-type" in h5_parent.attrs and h5_parent.attrs["mime-type"] == "spectrum":
            if h5_parent.parent.attrs["res_zoom"] == self.output_res:
                self.read_spectral_dataset(h5_parent)

        if isinstance(h5_parent, h5py.Group):
            for h5_child in h5_parent.keys():
                self.get_spectral_cube(h5_parent[h5_child])
        return

    def read_spectral_dataset(self, h5_object):
        spectrum_5d = self.get_pixels_from_spectrum(h5_object.name, h5_object)
        self.resize_output_if_necessary(spectrum_5d)
        self.spectral_cube[self.output_counter:self.output_counter + spectrum_5d.shape[0]] = spectrum_5d
        self.output_counter += spectrum_5d.shape[0]
        cutout_refs = h5_object.attrs["image_cutouts"]

        if len(cutout_refs) > 0:
            for region_ref in cutout_refs:
                print("reading region %s dataset: %s" % (region_ref, h5_object.name))
                try:
                    image_5d = self.get_pixels_from_image_cutout(h5_object, self.output_res, region_ref)
                    self.resize_output_if_necessary(image_5d)
                    self.spectral_cube[self.output_counter:self.output_counter + image_5d.shape[0]] = image_5d
                    self.output_counter += image_5d.shape[0]
                except ValueError as e:
                    self.logger.error("Could not process region for %s, message: %s" % (h5_object.name, str(e)))

    def resize_output_if_necessary(self, spectrum_5d):
        if self.output_counter + spectrum_5d.shape[0] > self.spectral_cube.shape[0]:
            self.spectral_cube.resize((self.spectral_cube.shape[0] * 2, 1), refcheck=False)

    def get_pixels_from_spectrum(self, spectrum_h5_path, spectrum_ds):
        res = int(spectrum_h5_path.split('/')[-2])
        ra, dec = spectrum_ds.attrs["PLUG_RA"], spectrum_ds.attrs["PLUG_DEC"]
        heal_id = hp.ang2pix(hp.order2nside(self.OUTPUT_HEAL_ORDER),
                             ra, dec,
                             nest=True,
                             lonlat=True)
        time = spectrum_ds.attrs["MJD"]
        heal_id_column = np.repeat([[heal_id]], res, axis=0)
        ra_column = np.repeat([[ra]], res, axis=0)
        dec_column = np.repeat([[dec]], res, axis=0)
        time_column = np.repeat([[time]], res, axis=0)
        spectrum_part = spectrum_ds[0]

        if self.INCLUDE_FITS_PATH is False:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, 1),
                                spectrum_part[:, 1].reshape(res, 1),
                                spectrum_part[:, 2].reshape(res, 1)]
        else:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                    'spectrum_fits_name '
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self.get_spectrum_columns(res,
                                                                                                      spectrum_ds)
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, 1),
                                spectrum_part[:, 1].reshape(res, 1),
                                spectrum_part[:, 2].reshape(res, 1),
                                spectrum_ra_column,
                                spectrum_dec_column,
                                spectrum_name_column,
                                spectrum_name_column]
        return np.rec.fromarrays(spectrum_columns, names=spectrum_column_names)

    def get_pixels_from_image_cutout(self, spectrum_ds, res_idx, region_ref):
        image_ds = self.f[region_ref]
        image_region = image_ds[region_ref]
        try:
            time = Time(image_ds.attrs["DATE-OBS"], format='isot', scale='tai').mjd
        except ValueError:
            time = Time(datetime.strptime(image_ds.attrs["DATE-OBS"], "%d/%m/%y")).mjd
        wl = image_ds.name.split('/')[-3]

        w = get_optimized_wcs(image_ds.attrs)
        cutout_bounds = self.get_cutout_bounds(image_ds, res_idx, spectrum_ds.attrs)
        y = np.arange(cutout_bounds[0][1][1], cutout_bounds[1][1][1])
        x = np.arange(cutout_bounds[0][0][0], cutout_bounds[1][1][0])
        X, Y = np.meshgrid(x, y)
        ra, dec = w.wcs_pix2world(X, Y, 0)
        pixel_IDs = hp.ang2pix(hp.order2nside(self.OUTPUT_HEAL_ORDER), ra, dec, nest=True, lonlat=True)
        no_pixels = pixel_IDs.size
        pixel_IDs_column = pixel_IDs.reshape((no_pixels, 1))
        ra_column = ra.reshape(no_pixels, 1)
        dec_column = dec.reshape(no_pixels, 1)
        data_columns = np.reshape(image_region, (no_pixels, 2))
        wl_column = np.repeat([[wl]], no_pixels, axis=0)
        time_column = np.repeat([[time]], no_pixels, axis=0)

        if self.INCLUDE_FITS_PATH is False:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            image_columns = [pixel_IDs_column, ra_column, dec_column, time_column, wl_column,
                             data_columns[:, 0].reshape(no_pixels, 1),
                             data_columns[:, 1].reshape(no_pixels, 1)]
        else:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                 'spectrum_fits_name '
            image_fits_name = image_ds.name.split('/')[-1]
            image_fits_name_casted = np.array([[image_fits_name]]).astype(np.dtype('S32'))
            image_fits_name_column = np.repeat(image_fits_name_casted, no_pixels, axis=0)
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self.get_spectrum_columns(no_pixels,
                                                                                                      spectrum_ds)

            image_columns = [pixel_IDs_column,
                             ra_column,
                             dec_column,
                             time_column,
                             wl_column,
                             data_columns[:, 0].reshape(no_pixels, 1),
                             data_columns[:, 1].reshape(no_pixels, 1),
                             spectrum_ra_column,
                             spectrum_dec_column,
                             image_fits_name_column,
                             spectrum_name_column]

        return np.rec.fromarrays(image_columns,
                                 names=image_column_names)

    def get_spectrum_columns(self, no_pixels, spectrum_ds):
        spectrum_fits_name = spectrum_ds.name.split('/')[-1]
        spectrum_fits_name_casted = np.array([[spectrum_fits_name]]).astype(np.dtype('S32'))
        spectrum_ra_column = np.repeat([[spectrum_ds.attrs["PLUG_RA"]]], no_pixels, axis=0)
        spectrum_dec_column = np.repeat([[spectrum_ds.attrs["PLUG_DEC"]]], no_pixels, axis=0)
        spectrum_name_column = np.repeat(spectrum_fits_name_casted, no_pixels, axis=0)
        return spectrum_ra_column, spectrum_dec_column, spectrum_name_column

    def write_VOTable(self, output_path):
        table = self.get_q_table()
        votable = from_table(table)
        writeto(votable, output_path, tabledata_format="binary")

    def write_FITS(self, output_path):
        table = self.get_q_table()
        table.write(output_path, overwrite=True, format='fits')

    def get_q_table(self):
        if not self.INCLUDE_FITS_PATH:
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma"),
                           meta={'name': 'SDSS Cube'})
        else:
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma",
                                                      "Spectrum RA", "Spectrum DEC", "FITS Name", "Spectrum FITS Name"),
                           meta={'name': 'SDSS Cube'})
        return table
