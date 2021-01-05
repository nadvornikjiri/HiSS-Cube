import os
import pathlib

import argparse
import fitsio
import h5py
import numpy as np
from mpi4py import MPI
import lib.SDSSCubeWriter as Writer

from lib import photometry as cu
import pydevd_pycharm

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

port_mapping = [34725, 41905]
pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

print(os.getpid())

WORK_TAG = 0
FINISHED_TAG = 1


class SDSSCubeParallelWriter(Writer.SDSSCubeWriter):

    def __init__(self, h5_path=None, h5_file=None):
        super().__init__()
        # mpio
        self.mpio = self.config.getboolean("Handler", "MPIO")
        self.BATCH_SIZE = int(self.config["Writer"]["BATCH_SIZE"])
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()

        # utils
        lib_path = pathlib.Path(__file__).parent.absolute()
        cube_utils = self.cube_utils = cu.Photometry("%s/../config/SDSS_Bands" % lib_path,
                                                     "%s/../config/ccd_gain.tsv" % lib_path,
                                                     "%s/../config/ccd_dark_variance.tsv" % lib_path)
        super().__init__(h5_file, cube_utils)

        self.h5_path = h5_path

    def open_h5_file_serial(self):
        self.f = h5py.File(self.h5_path, 'r+')

    def open_h5_file_parallel(self):
        if self.mpio:
            self.f = h5py.File(self.h5_path, 'r+', driver='mpio', comm=self.comm)
        else:
            self.f = h5py.File(self.h5_path, 'r+')

    def truncate_h5_file(self):
        self.f = h5py.File(self.h5_path, 'w')
        self.f.close()

    def ingest_data(self, image_path, spectra_path, truncate_file=None):
        if self.mpi_rank == 0:
            if truncate_file:
                self.truncate_h5_file()
            if not self.f:
                self.open_h5_file_serial()
            self.ingest_metadata(image_path, spectra_path)
            self.close_h5_file()
        self.comm.Barrier()
        self.open_h5_file_parallel()
        if self.mpi_rank == 0:
            self.distribute_work(self.image_path_list)
        else:
            self.write_image_data()
        self.comm.Barrier()
        if self.mpi_rank == 0:
            self.distribute_work(self.spectra_path_list)
        else:
            self.write_spectra_data_and_links()
        self.close_h5_file()

    def write_spectrum_data(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)
        while status.Get_tag() != FINISHED_TAG:
            for image_path in image_path_list:
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path, self.IMG_MIN_RES)
                self.file_name = image_path.name
                self.write_img_datasets()
            self.comm.send(obj=None, dest=0)
            image_path_list = self.receive_work(status)

    def receive_work(self, status):
        data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        self.logger.info("Received work from master: %s" % data)
        return data

    def write_spectra_data_and_links(self):
        status = MPI.Status()
        spectra_path_list = self.receive_work(status)
        while status.Get_tag() != FINISHED_TAG:
            for spec_path in spectra_path_list:
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(spec_path,
                                                                                            self.SPEC_MIN_RES)
                spec_datasets = self.write_spec_datasets()
                self.add_image_refs_to_spectra(spec_datasets)
            self.comm.send(obj=None, dest=0)
            spectra_path_list = self.receive_work(status)

    def distribute_work(self, path_list):
        status = MPI.Status()
        batches = list(chunks(path_list, self.BATCH_SIZE))
        for i in range(1, self.mpi_size):
            self.send_work(batches, dest=i)
        while len(batches) > 0:
            self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            self.send_work(batches, status.Get_source())

    def send_work(self, batches, dest):
        self.logger.info("Sending work batch to dest:%d " % dest)
        batch = []
        try:
            batch = batches.pop()
            tag = WORK_TAG
        except IndexError:
            tag = FINISHED_TAG
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
    parser.add_argument('-t', '--truncate', action='store_const', const=True,
                        help="Should truncate the file if exists?")
    args = parser.parse_args()

    fits_image_path = "%s/images" % args.input_path
    fits_spectra_path = "%s/spectra" % args.input_path

    SDSSCubeParallelWriter(args.output_path).ingest_data(fits_image_path, fits_spectra_path, args.truncate)
