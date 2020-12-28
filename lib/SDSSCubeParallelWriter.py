import os
import pathlib

import argparse
import fitsio
import h5py
import numpy as np
from mpi4py import MPI
import lib.SDSSCubeWriter as Writer

from lib import photometry as cu

WORK_TAG = 0
DIE_TAG = 1


class SDSSCubeParallelWriter(Writer.SDSSCubeWriter):

    def __init__(self, h5_path=None, h5_file=None):
        super().__init__()
        # mpio
        self.mpio = self.config.getboolean("Handler", "MPIO")
        self.BATCH_SIZE = int(self.config["Writer"]["BATCH_SIZE"])
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()
        # preprocessing
        self.IMAGE_PATTERN = self.config["Writer"]["IMAGE_PATTERN"]
        self.SPECTRA_PATTERN = self.config["Writer"]["SPECTRA_PATTERN"]
        self.MAX_CUTOUT_REFS = int(self.config["Writer"]["MAX_CUTOUT_REFS"])

        # utils
        lib_path = pathlib.Path(__file__).parent.absolute()
        cube_utils = self.cube_utils = cu.Photometry("%s/../config/SDSS_Bands" % lib_path,
                                                     "%s/../config/ccd_gain.tsv" % lib_path,
                                                     "%s/../config/ccd_dark_variance.tsv" % lib_path)
        super().__init__(h5_file, cube_utils)

        self.h5_path = h5_path
        if not h5_file:
            self.open_h5_file()

    def open_h5_file(self):
        if self.mpio:
            self.f = h5py.File(self.h5_path, 'w', driver='mpio', comm=self.comm)
        else:
            self.f = h5py.File(self.h5_path, 'w')

    def ingest_data(self, image_path, spectra_path):
        if self.mpi_rank == 0:
            self.ingest_metadata(image_path, spectra_path)
        # if self.mpi_rank == 0:
        #     self.distribute_work(self.image_path_list)
        # else:
        #     self.write_image_data()
        # self.comm.Barrier()
        # if self.mpi_rank == 0:
        #     self.distribute_work(self.spectra_path_list)
        # else:
        #     self.write_spectra_data_and_links()

    def ingest_image(self, image_path, res_grps):
        self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path, self.IMG_MIN_RES)
        self.file_name = os.path.basename(image_path)
        img_datasets = self.create_img_datasets(res_grps)
        self.add_metadata(img_datasets)
        return img_datasets

    def ingest_spectrum(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)
        while status.Get_tag() != DIE_TAG:
            for path in image_path_list:
                self.ingest_image(path)
            self.comm.send(obj=None, dest=0)
            image_path_list = self.receive_work(status)

    def receive_work(self, status):
        return self.comm.recv(obj=None, source=0, tag=MPI.ANY_TAG, status=status)

    def write_spectra_data_and_links(self):
        status = MPI.Status()
        spectra_path_list = self.receive_work(status)
        while status.Get_tag() != DIE_TAG:
            for path in spectra_path_list:
                self.ingest_spectrum(path)
            self.comm.send(obj=None, dest=0)
            spectra_path_list = self.receive_work(status)

    def distribute_work(self, path_list):
        status = MPI.Status()
        batches = chunks(path_list, self.BATCH_SIZE)
        for i in range(1, self.mpi_size):
            self.send_work(batches, dest=i)
        while len(batches) > 0:
            self.comm.recv(obj=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            self.send_work(batches, status.Get_source())

    def send_work(self, batches, dest):
        try:
            batch = batches.pop()
            tag = WORK_TAG
        except IndexError:
            tag = DIE_TAG
        self.comm.send(obj=batch, dest=dest, tag=tag)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import images and spectra in parallel')
    parser.add_argument('input_path', metavar="input", type=str,
                        help="data folder that includes folders images and spectra")
    parser.add_argument('output_path', metavar="output", type=str,
                        help="path to HDF5 file, does not need to exist")
    args = parser.parse_args()

    fits_image_path = "%s/images" % args.input_path
    fits_spectra_path = "%s/spectra" % args.input_path

    SDSSCubeParallelWriter(args.output_path).ingest_data(fits_image_path, fits_spectra_path)
