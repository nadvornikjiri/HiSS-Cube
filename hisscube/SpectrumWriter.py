import os
import pathlib

import fitsio
import h5py
import numpy as np

from hisscube.H5Handler import H5Handler
from hisscube.astrometry import NoCoverageFoundError


class SpectrumWriter(H5Handler):

    def __init__(self, h5_file=None, h5_path=None):
        super().__init__(h5_file, h5_path)
        self.spec_cnt = 0

    def ingest_spectrum(self, spec_path):
        """
        Method that writes a spectrum to the opened HDF5 file (self.f). Needs to be called after all images are already
        ingested, as it also links the spectra to the images via the Region References.
        Parameters
        ----------
        spec_path   String

        Returns     HDF5 dataset (already written to the file)
        -------

        """
        self.write_spectrum_metadata(spec_path)
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(
            spec_path, self.config.getint("Handler", "SPEC_ZOOM_CNT"),
            apply_rebin=self.config.getboolean("Preprocessing", "APPLY_REBIN"),
            rebin_min=self.config.getfloat("Preprocessing", "REBIN_MIN"),
            rebin_max=self.config.getfloat("Preprocessing", "REBIN_MAX"),
            rebin_samples=self.config.getint("Preprocessing", "REBIN_SAMPLES"),
            apply_transmission=self.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE"))
        spec_datasets = self.write_spec_datasets()
        return spec_datasets

    def create_spectrum_index_tree(self):
        """
        Creates the index tree for a spectrum.
        Returns HDF5 group - the one where the spectrum dataset should be placed.
        -------

        """
        spec_grp = self.require_raw_cube_grp()
        spatial_grp = self.require_spectrum_spatial_grp_structure(spec_grp)
        time_grp = self.require_spectrum_time_grp(spatial_grp)
        res_grps = self.require_res_grps(time_grp)
        return res_grps

    def require_spectrum_spatial_grp_structure(self, child_grp):
        """
        Creates the spatial index part for a spectrum. Takes the root group as parameter.
        Parameters
        ----------
        child_grp   HDF5 group

        Returns     HDF5 group
        -------

        """
        spectrum_coord = (self.metadata['PLUG_RA'], self.metadata['PLUG_DEC'])
        for order in range(self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER")):
            child_grp = self.require_spatial_grp(order, child_grp, spectrum_coord)

        for img_zoom in range(self.config.getint("Handler", "IMG_ZOOM_CNT")):
            ds = child_grp.require_dataset("image_cutouts_%d" % img_zoom,
                                           (self.config.getint("Writer", "MAX_CUTOUT_REFS"),),
                                           dtype=h5py.regionref_dtype)
            # ds[0] = None
        return child_grp

    def require_spectrum_time_grp(self, parent_grp):
        time = self.get_time_from_spectrum(self.metadata)
        grp = self.require_group(parent_grp, str(time), track_order=True)
        grp.attrs["type"] = "time"
        return grp

    def create_spec_datasets(self, parent_grp_list):
        spec_datasets = []
        for group in parent_grp_list:
            res = int(group.name.split('/')[-1])
            spec_data_shape = (res,) + (3,)
            dcpl, space, spec_data_dtype = self.get_property_list(spec_data_shape)
            ds_name = self.file_name.encode()
            if not ds_name in group:
                dsid = h5py.h5d.create(group.id, ds_name, spec_data_dtype, space, dcpl=dcpl)
                ds = h5py.Dataset(dsid)
            else:
                ds = group[ds_name]
            ds.attrs["mime-type"] = "spectrum"
            spec_datasets.append(ds)
        return spec_datasets

    def add_image_refs(self, h5_grp, depth=-1):
        if "type" in h5_grp.attrs and \
                h5_grp.attrs["type"] == "spatial" and \
                depth == self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"):
            spec_datasets = []
            for child_grp in h5_grp.values():
                if isinstance(child_grp, h5py.Group) and child_grp.attrs["type"] == "time":
                    time_grp_1st_spectrum = child_grp  # we can take the first, all of the spectra have same coordinates here
                    break

            for res_grp in time_grp_1st_spectrum.values():
                for ds_name, ds in res_grp.items():
                    if ds_name.endswith("fits"):
                        spec_datasets.append(ds)
            self.add_image_refs_to_spectra(spec_datasets)
        else:
            if isinstance(h5_grp, h5py.Group):
                for child_grp in h5_grp.values():
                    if isinstance(child_grp, h5py.Group):
                        self.add_image_refs(child_grp, depth + 1)

    def add_image_refs_to_spectra(self, spec_datasets):
        """
        Adds HDF5 Region references of image cut-outs to spectra attribute "image_cutouts". Throws NoCoverageFoundError
        if the cut-out does not span the whole cutout size for any reason.
        Parameters
        ----------
        spec_datasets   [HDF5 Datasets]

        Returns         [HDF5 Datasets]
        -------

        """

        image_refs = {}
        image_min_zoom_idx = 0
        self.metadata = spec_datasets[0].attrs
        for image_res_idx, image_ds in self.find_images_overlapping_spectrum():
            if not image_res_idx in image_refs:
                image_refs[image_res_idx] = []
            try:
                image_refs[image_res_idx].append(self.get_region_ref(image_res_idx, image_ds))
                if image_res_idx > image_min_zoom_idx:
                    image_min_zoom_idx = image_res_idx
            except NoCoverageFoundError as e:
                # self.logger.debug("No coverage found for spectrum %s and image %s, reason %s" % (self.file_name, image_ds, str(e)))
                pass

        for res in image_refs:
            image_refs[res] = np.array(image_refs[res],
                                       dtype=h5py.regionref_dtype)
        for spec_zoom_idx, spec_ds in enumerate(spec_datasets):
            image_cutout_ds = spec_ds.parent.parent.parent[
                "image_cutouts_%d" % spec_zoom_idx]  # we write image cutout zoom equivalent to the spectral zoom
            if len(image_refs) > 0:
                if spec_zoom_idx > image_min_zoom_idx:
                    no_references = len(image_refs[image_min_zoom_idx])
                    image_cutout_ds[0:no_references] = image_refs[image_min_zoom_idx]
                else:
                    no_references = len(image_refs[spec_zoom_idx])
                    image_cutout_ds[0:no_references] = image_refs[spec_zoom_idx]
        return spec_datasets

    def find_images_overlapping_spectrum(self):
        """Finds images in the HDF5 index structure that overlap the spectrum coordinate. Does so by constructing the
        whole heal_path string to the image and it to get the correct Group containing those images. Yields resolution
        index and the image dataset.

        Yields         (int, HDF5 dataset)
        -------
        """
        heal_path = self.get_heal_path_from_coords()
        heal_path_group = self.f[heal_path]
        for time_grp in heal_path_group.values():
            if isinstance(time_grp, h5py.Group):
                for band_grp in time_grp.values():
                    if isinstance(band_grp, h5py.Group) and band_grp.attrs["type"] == "spectral":
                        for res_idx, res in enumerate(band_grp):
                            res_grp = band_grp[res]
                            for image_ds in res_grp.values():
                                try:
                                    if image_ds.attrs["mime-type"] == "image":
                                        yield res_idx, image_ds
                                except KeyError:
                                    pass

    def write_spectra_metadata(self, spectra_folder, spectra_pattern):
        for fits_path in pathlib.Path(spectra_folder).rglob(
                spectra_pattern):  # TODO Remove test numbers
            self.write_spectrum_metadata(fits_path)
            self.spec_cnt += 1
            if self.spec_cnt >= 50000:
                break
        self.f.attrs["spectrum_count"] = self.spec_cnt

    def write_spectrum_metadata(self, fits_path):
        self.ingest_type = "spectrum"
        self.spectra_path_list.append(str(fits_path))
        self.metadata = fitsio.read_header(fits_path)
        if self.config.getboolean("Preprocessing", "APPLY_REBIN") is False:
            self.spectrum_length = fitsio.read_header(fits_path, 1)["NAXIS2"]
        else:
            self.spectrum_length = self.config.getint("Preprocessing", "REBIN_SAMPLES")
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_spectrum_index_tree()
        spec_datasets = self.create_spec_datasets(res_grps)
        self.add_metadata(spec_datasets)

    def write_spec_datasets(self):
        res_grp_list = self.get_spectral_resolution_groups()
        spec_datasets = []
        for group in res_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in self.data if str(spec["res"]) == res)
            spec_data = np.column_stack((wanted_res["wl"], wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            spec_data[spec_data == np.inf] = np.nan
            if self.config.get("Writer", "COMPRESSION"):
                spec_data = self.float_compress(spec_data)
            ds = group[self.file_name]
            ds.write_direct(spec_data)
            spec_datasets.append(ds)
        return spec_datasets

    def get_spectral_resolution_groups(self):
        spatial_path = self.get_heal_path_from_coords(order=self.config.getint("Handler", "SPEC_SPAT_INDEX_ORDER"))
        try:
            time = self.metadata["TAI"]
        except KeyError:
            time = self.metadata["MJD"]
        path = "/".join([spatial_path, str(time)])
        time_grp = self.f[path]
        for res_grp in time_grp:
            yield time_grp[res_grp]
