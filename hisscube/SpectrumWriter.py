import os
import pathlib

import fitsio
import h5py
import numpy as np

from hisscube.H5Handler import H5Handler
from hisscube.astrometry import NoCoverageFoundError
from timeit import default_timer as timer

from hisscube.fitstools import read_primary_header_quick, read_header_from_path


class SpectrumWriter(H5Handler):

    def __init__(self, h5_file=None, h5_path=None, timings_log="image_timings.csv"):
        super().__init__(h5_file, h5_path, timings_log)

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
            spec_path, self.SPEC_ZOOM_CNT,
            apply_rebin=self.APPLY_REBIN,
            rebin_min=self.REBIN_MIN,
            rebin_max=self.REBIN_MAX,
            rebin_samples=self.REBIN_SAMPLES,
            apply_transmission=self.APPLY_TRANSMISSION_CURVE)
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
        for order in range(self.SPEC_SPAT_INDEX_ORDER):
            child_grp = self.require_spatial_grp(order, child_grp, spectrum_coord)

        for img_zoom in range(self.IMG_ZOOM_CNT):
            self.require_dataset(child_grp, "image_cutouts_%d" % img_zoom,
                                 (self.MAX_CUTOUT_REFS,),
                                 dtype=h5py.regionref_dtype)

        return child_grp

    def require_spectrum_time_grp(self, parent_grp):
        time = self.get_time_from_spectrum(self.metadata)
        grp = self.require_group(parent_grp, str(time), track_order=True)
        self.set_attr(grp, "type", "time")
        return grp

    def create_spec_datasets(self, parent_grp_list):
        spec_datasets = []
        for group in parent_grp_list:
            if self.C_BOOSTER:
                if "spectrum_dataset" in group:
                    raise ValueError(
                        "There is already an image dataset %s within this resolution group. Trying to insert image %s." % (
                            list(group["spectrum_dataset"]), self.file_name))
            elif len(group) > 0:
                raise ValueError(
                    "There is already a spectrum dataset %s within this resolution group. Trying to insert spectrum %s." % (
                        list(group), self.file_name))
            res = int(self.get_name(group).split('/')[-1])
            spec_data_shape = (res,) + (3,)
            ds = self.create_spectrum_h5_dataset(group, spec_data_shape)
            self.set_attr(ds, "mime-type", "spectrum")
            spec_datasets.append(ds)
        return spec_datasets

    def create_spectrum_h5_dataset(self, group, spec_data_shape):
        dcpl, space, spec_data_dtype = self.get_property_list(spec_data_shape)
        ds_name = self.file_name.encode()
        if not ds_name in group:
            dsid = h5py.h5d.create(group.id, ds_name, spec_data_dtype, space, dcpl=dcpl)
            ds = h5py.Dataset(dsid)
        else:
            ds = group[ds_name]
        return ds

    def add_image_refs(self, h5_grp, depth=-1):
        if "type" in h5_grp.attrs and \
                h5_grp.attrs["type"] == "spatial" and \
                depth == self.SPEC_SPAT_INDEX_ORDER:
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
        self.metadata = self.read_serialized_fits_header(spec_datasets[0])
        for image_res_idx, image_ds in self.find_images_overlapping_spectrum():
            if not image_res_idx in image_refs:
                image_refs[image_res_idx] = []
            try:
                image_refs[image_res_idx].append(self.get_region_ref(image_res_idx, image_ds))
                if image_res_idx > image_min_zoom_idx:
                    image_min_zoom_idx = image_res_idx
            except NoCoverageFoundError as e:
                self.logger.debug("No coverage found for spectrum %s and image %s, reason %s" % (self.file_name, image_ds, str(e)))
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

    def write_spectra_metadata(self, spectra_folder, spectra_pattern, no_attrs=False, no_datasets=False):
        start = timer()
        check = 100
        for fits_path in pathlib.Path(spectra_folder).rglob(
                spectra_pattern):
            if self.spec_cnt % check == 0 and self.spec_cnt / check > 0:
                end = timer()
                self.logger.info("100 spectra done in %.4fs" % (end - start))
                self.log_metadata_csv_timing(end - start)
                start = end
                self.logger.info("Spectra cnt: %05d" % self.spec_cnt)
            try:
                self.write_spectrum_metadata(fits_path, no_attrs, no_datasets)
                self.spec_cnt += 1
            except ValueError as e:
                self.logger.warning(
                    "Unable to ingest spectrum %s, message: %s" % (fits_path, str(e)))
            if self.spec_cnt >= self.LIMIT_SPECTRA_COUNT:
                break
        self.set_attr(self.f, "spectrum_count", self.spec_cnt)

    def write_spectrum_metadata(self, fits_path, no_attrs=False, no_datasets=False):
        self.ingest_type = "spectrum"
        self.spectra_path_list.append(str(fits_path))
        self.metadata = fitsio.read_header(fits_path)
        if self.APPLY_REBIN is False:
            self.spectrum_length = fitsio.read_header(fits_path, 1)["NAXIS2"]
        else:
            self.spectrum_length = self.REBIN_SAMPLES
        self.file_name = os.path.basename(fits_path)
        res_grps = self.create_spectrum_index_tree()
        if not no_datasets:
            spec_datasets = self.create_spec_datasets(res_grps)
        if not no_attrs:
            self.add_metadata(spec_datasets)

    def write_spec_datasets(self):
        res_grp_list = self.get_spectral_resolution_groups()
        spec_datasets = []
        for group in res_grp_list:
            res = group.name.split('/')[-1]
            wanted_res = next(spec for spec in self.data if str(spec["res"]) == res)
            spec_data = np.column_stack((wanted_res["wl"], wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            spec_data[spec_data == np.inf] = np.nan
            if self.FLOAT_COMPRESS:
                spec_data = self.float_compress(spec_data)
            ds = group[self.file_name]
            ds.write_direct(spec_data)
            spec_datasets.append(ds)
        return spec_datasets

    def get_spectral_resolution_groups(self):
        spatial_path = self.get_heal_path_from_coords(order=self.SPEC_SPAT_INDEX_ORDER)
        try:
            time = self.metadata["TAI"]
        except KeyError:
            time = self.metadata["MJD"]
        path = "/".join([spatial_path, str(time)])
        time_grp = self.f[path]
        for res_grp in time_grp:
            yield time_grp[res_grp]
