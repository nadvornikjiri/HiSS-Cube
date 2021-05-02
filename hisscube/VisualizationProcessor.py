import h5py
import healpy as hp
import numpy as np
from astropy.io.votable import from_table, writeto
from astropy.table import QTable

from hisscube.Processor import Processor


class VisualizationProcessor(Processor):

    def __init__(self, h5_file, cube_utils):
        super().__init__(h5_file, cube_utils)
        if self.config.getboolean("Handler",
                                  "INCLUDE_ADDITIONAL_METADATA"):  # TODO add the grouprefs to the images and spectra from dense cube
            self.array_type = [('heal_id', '<i8'), ('ra', '<f4'), ('dec', '<f4'), ('time', '<f4'), ('wl', '<f4'),
                               ('mean', '<f4'), ('sigma', '<f4'), ('spec_ra', '<f4'), ('spec_dec', '<f4'),
                               ('fits_name', 'S32'), ('spec_fits_name', 'S32')]
        else:
            self.array_type = [('heal_id', '<i8'), ('ra', '<f8'), ('dec', '<f8'), ('time', '<f8'), ('wl', '<f8'),
                               ('mean', '<f8'), ('sigma', '<f8')]
        self.output_counter = 0
        self.spectral_cube = None
        self.output_res = None

    def read_spectral_cube_table(self, res):
        """
        This method just reads the dense cube dataset and returns it as numpy array.
        Parameters
        ----------
        resolution  int

        Returns     numpy array
        -------

        """
        spectral_cube_path = "%s/%d/%s/dense_cube_zoom_%d" % (
            self.config.get("Handler", "DENSE_CUBE_NAME"), res, "visualization", res)
        self.spectral_cube = self.f[spectral_cube_path][()]
        return self.spectral_cube

    def construct_spectral_cube_table(self, zoom):
        """
        This method constructs the dense cube from the semi-sparse cube tree in the HDF5 file for a given resolution.
        It iterates over the individual spectra, reads the "image_cutouts" attribute and de-references all of the
        region references there. Returns coordinates for every voxel of the constructed 4D spectral cube (ra, dec,
        time, wavelength) + optional metadata if enabled by the self.INCLUDE_ADDITIONAL_METADATA.

        Parameters
        ----------
        resolution  int

        Returns
        -------

        """
        self.output_counter = 0
        self.spectral_cube = np.empty((self.config.getint("Handler", "INIT_ARRAY_SIZE"),), dtype=self.array_type)
        self.output_res = zoom
        self.construct_multires_spectral_cube_table(self.f)
        truncated_cube = self.spectral_cube[:self.output_counter]
        self.spectral_cube = truncated_cube
        return self.spectral_cube

    def construct_multires_spectral_cube_table(self, h5_parent):
        if "mime-type" in h5_parent.attrs and h5_parent.attrs["mime-type"] == "spectrum":
            if h5_parent.parent.attrs["res_zoom"] == self.output_res:
                self.construct_spectrum_table(h5_parent)

        if isinstance(h5_parent, h5py.Group):
            for h5_child in h5_parent.keys():
                self.construct_multires_spectral_cube_table(h5_parent[h5_child])
        return

    def construct_spectrum_table(self, spectrum_ds):
        """
        Reads the spectral dataset and writes it ot self.spectral_cube.
        Parameters
        ----------
        spectrum_ds HDF5 dataset

        Returns
        -------

        """
        try:
            if spectrum_ds.attrs["orig_res_link"]:
                self.metadata = self.f[spectrum_ds.attrs["orig_res_link"]].attrs
            else:
                self.metadata = spectrum_ds.attrs
        except KeyError:
            self.metadata = spectrum_ds.attrs
        spectrum_5d = self.get_table_pixels_from_spectrum(spectrum_ds.name, spectrum_ds)
        self.resize_output_if_necessary(spectrum_5d)
        self.spectral_cube[self.output_counter:self.output_counter + spectrum_5d.shape[0]] = spectrum_5d
        self.output_counter += spectrum_5d.shape[0]
        cutout_refs = spectrum_ds.parent.parent.parent["image_cutouts_%s" % spectrum_ds.parent.attrs["res_zoom"]]

        if len(cutout_refs) > 0:
            for region_ref in cutout_refs:
                if region_ref:
                    try:
                        image_5d = self.get_table_pixels_from_image_cutout(spectrum_ds, self.output_res, region_ref)
                        self.resize_output_if_necessary(image_5d)
                        self.spectral_cube[self.output_counter:self.output_counter + image_5d.shape[0]] = image_5d
                        self.output_counter += image_5d.shape[0]
                    except ValueError as e:
                        self.logger.error("Could not process region for %s, message: %s" % (spectrum_ds.name, str(e)))
                else:
                    break  # necessary because of how null object references are tested in h5py dataset

    def resize_output_if_necessary(self, spectrum_5d):
        if self.output_counter + spectrum_5d.shape[0] > self.spectral_cube.shape[0]:
            self.spectral_cube.resize((self.spectral_cube.shape[0] * 2,), refcheck=False)

    def get_table_pixels_from_spectrum(self, spectrum_h5_path, spectrum_ds):
        """
        Gets array of pixels from the spectrum, also containing their coordinates.
        Parameters
        ----------
        spectrum_h5_path    String
        spectrum_ds         HDF5 dataset

        Returns             numpy record
        -------

        """
        res = int(spectrum_h5_path.split('/')[-2])
        spectrum_fits_name = spectrum_ds.name.split('/')[-1]
        ra, dec = self.metadata["PLUG_RA"], self.metadata["PLUG_DEC"]
        spectrum_part = spectrum_ds
        return self.get_table_pixels_from_spectrum_generic(dec, ra, res, spectrum_fits_name, spectrum_part)

    def get_table_pixels_from_spectrum_generic(self, dec, ra, res, spectrum_fits_name, spectrum_part):
        heal_id = hp.ang2pix(hp.order2nside(self.OUTPUT_HEAL_ORDER),
                             ra, dec,
                             nest=True,
                             lonlat=True)
        time = self.metadata["MJD"]
        heal_id_column = np.repeat([heal_id], res, axis=0)
        ra_column = np.repeat([ra], res, axis=0)
        dec_column = np.repeat([dec], res, axis=0)
        time_column = np.repeat([time], res, axis=0)
        if self.config.getboolean("Handler", "INCLUDE_ADDITIONAL_METADATA") is False:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, ),
                                spectrum_part[:, 1].reshape(res, ),
                                spectrum_part[:, 2].reshape(res, )]
        else:
            spectrum_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                    'spectrum_fits_name '
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self.get_spectrum_table_columns(res,
                                                                                                            spectrum_fits_name)
            spectrum_columns = [heal_id_column,
                                ra_column,
                                dec_column,
                                time_column,
                                spectrum_part[:, 0].reshape(res, ),
                                spectrum_part[:, 1].reshape(res, ),
                                spectrum_part[:, 2].reshape(res, ),
                                spectrum_ra_column,
                                spectrum_dec_column,
                                spectrum_name_column,
                                spectrum_name_column]
        return np.rec.fromarrays(spectrum_columns, names=spectrum_column_names)

    def get_table_pixels_from_image_cutout(self, spectrum_ds, res_idx, region_ref):
        """
        Gets all of the image pixels for the given cutout, along with its coordinates.
        Parameters
        ----------
        spectrum_ds HDF5 dataset
        res_idx     int
        region_ref  HDF5 region reference

        Returns     numpy record
        -------

        """
        spectrum_path = spectrum_ds.name
        image_ds = self.f[region_ref]
        image_path = image_ds.name
        image_region = image_ds[region_ref]

        cutout_bounds, time, w, wl = self.get_cutout_bounds_from_spectrum(image_ds, res_idx, spectrum_ds)
        return self.get_table_image_pixels_from_cutout_bounds(cutout_bounds, image_path, image_region, spectrum_path,
                                                              time, w,
                                                              wl)

    def get_table_image_pixels_from_cutout_bounds(self, cutout_bounds, image_path, image_region, spectrum_path, time, w,
                                                  wl):
        pixel_ID, ra, dec = self.get_cutout_pixel_coords(cutout_bounds, w)
        no_pixels = ra.size
        pixel_ID = hp.ang2pix(hp.order2nside(self.OUTPUT_HEAL_ORDER),
                              self.metadata["PLUG_RA"], self.metadata["PLUG_DEC"],
                              nest=True, lonlat=True)
        pixel_IDs_column = np.repeat([pixel_ID], no_pixels, axis=0)
        ra_column = ra.reshape(no_pixels, )
        dec_column = dec.reshape(no_pixels, )
        data_columns = np.reshape(image_region, (no_pixels, 2))
        wl_column = np.repeat([wl], no_pixels, axis=0)
        time_column = np.repeat([time], no_pixels, axis=0)
        if self.config.getboolean("Handler", "INCLUDE_ADDITIONAL_METADATA") is False:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma'
            image_columns = [pixel_IDs_column, ra_column, dec_column, time_column, wl_column,
                             data_columns[:, 0].reshape(no_pixels, 1),
                             data_columns[:, 1].reshape(no_pixels, 1)]
        else:
            image_column_names = 'heal, ra, dec, time, wl, mean, sigma, spectrum_ra, spectrum_dec, fits_name, ' \
                                 'spectrum_fits_name '
            image_fits_name = str(image_path).split('/')[-1]
            image_fits_name_casted = np.array([image_fits_name]).astype(np.dtype('S32'))
            image_fits_name_column = np.repeat(image_fits_name_casted, no_pixels, axis=0)
            spectrum_fits_name = str(spectrum_path).split('/')[-1]
            spectrum_ra_column, spectrum_dec_column, spectrum_name_column = self.get_spectrum_table_columns(no_pixels,
                                                                                                            spectrum_fits_name)

            image_columns = [pixel_IDs_column,
                             ra_column,
                             dec_column,
                             time_column,
                             wl_column,
                             data_columns[:, 0].reshape(no_pixels, ),
                             data_columns[:, 1].reshape(no_pixels, ),
                             spectrum_ra_column,
                             spectrum_dec_column,
                             image_fits_name_column,
                             spectrum_name_column]
        return np.rec.fromarrays(image_columns,
                                 names=image_column_names)

    def get_spectrum_table_columns(self, no_pixels, spectrum_fits_name):
        spectrum_fits_name_casted = np.array([spectrum_fits_name]).astype(np.dtype('S32'))
        spectrum_ra_column = np.repeat([self.metadata["PLUG_RA"]], no_pixels, axis=0)
        spectrum_dec_column = np.repeat([self.metadata["PLUG_DEC"]], no_pixels, axis=0)
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
        if not self.config.getboolean("Handler", "INCLUDE_ADDITIONAL_METADATA"):
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma"),
                           meta={'name': 'SDSS Cube'})
        else:
            table = QTable(self.spectral_cube, names=("HealPix ID", "RA", "DEC", "Time", "Wavelength", "Mean", "Sigma",
                                                      "Spectrum RA", "Spectrum DEC", "FITS Name", "Spectrum FITS Name"),
                           meta={'name': 'SDSS Cube'})
        return table
