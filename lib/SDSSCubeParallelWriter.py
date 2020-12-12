import os
import pathlib

import h5py
from astropy.io import fits

from lib.SDSSCubeWriter import SDSSCubeWriter
from mpi4py import MPI
from lib import photometry as cu
import numpy as np
from ast import literal_eval as make_tuple


class SDSSCubeParallelWriter(SDSSCubeWriter):

    def __init__(self, h5_path):
        self.comm = MPI.COMM_WORLD
        h5_file = h5py.File(h5_path, 'w', driver='mpio', comm=self.comm)
        lib_path = pathlib.Path(__file__).parent.absolute()
        cube_utils = self.cube_utils = cu.CubeUtils("%s/../config/SDSS_Bands" % lib_path,
                                                    "%s/../config/ccd_gain.tsv" % lib_path,
                                                    "%s/../config/ccd_dark_variance.tsv" % lib_path)
        super().__init__(h5_file, cube_utils)
        self.IMAGE_PATTERN = self.config["Writer"]["IMAGE_PATTERN"]
        self.image_path_list = []
        self.spectra_path_list = []

    def ingest_data(self, image_path, spectra_path):
        rank = MPI.COMM_WORLD.rank
        if rank == 0:
            self.write_image_metadata(image_path)
            self.distribute_work()
            self.comm.Barrier()
            self.write_spectral_metadata(spectra_path)
            self.distribute_work()
            self.comm.Barrier()
            self.f.close()
        else:
            self.write_image_data()
            self.comm.Barrier()
            self.write_spectra_data()
            self.comm.Barrier()

    def ingest_image(self, image_path):
        return super().ingest_image(image_path)

    def ingest_spectrum(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_metadata(self, image_path):
        for fits_path in pathlib.Path(image_path).rglob(self.IMAGE_PATTERN):
            self.image_path_list.append(fits_path)
            with fits.open(fits_path, memmap=self.fitsMemMap) as f:
                self.metadata = f[0].header
                self.file_name = os.path.basename(image_path)
                res_grps = self.create_image_index_tree()
                img_datasets = self.create_img_datasets(res_grps)
                self.add_metadata(img_datasets)
                self.f.flush()

    def require_res_grps(self, parent_grp):
        res_grps = []
        min_res = self.IMG_MIN_RES
        x_lower_res = int(self.metadata["NAXIS1"])
        y_lower_res = int(self.metadata["NAXIS2"])
        res_zoom = 0
        while not (x_lower_res < min_res or y_lower_res < min_res):
            grp = self.require_group(str((y_lower_res, x_lower_res)))
            grp.attrs["type"] = "resolution"
            grp.attrs["res_zoom"] = res_zoom
            res_grps.append(grp)
            res_zoom += 1
            x_lower_res /= 2
            y_lower_res /= 2
        return res_grps

    def create_img_datasets(self, parent_grp_list):
        img_datasets = []
        for group in parent_grp_list:
            res_tuple = group.name.split('/')[-1]
            wanted_res = next(img for img in self.data if str(img["res"]) == res_tuple)  # parsing 2D resolution
            img_data = np.dstack((wanted_res["flux_mean"], wanted_res["flux_sigma"]))
            img_data_shape = make_tuple(res_tuple) + (2,)
            img_data_dtype = np.dtype('f4')

            ds = group.require_dataset(self.file_name, img_data_shape, img_data_dtype,
                                       chunks=self.CHUNK_SIZE,
                                       compression=self.COMPRESSION,
                                       compression_opts=self.COMPRESSION_OPTS,
                                       shuffle=self.SHUFFLE)
            ds.attrs["mime-type"] = "image"
            img_datasets.append(ds)
        return img_datasets

    def write_spectral_metadata(self, spectra_path):
        pass

    def write_image_data(self):
        pass

    def write_spectra_data(self):
        pass
