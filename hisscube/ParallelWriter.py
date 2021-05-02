from setuptools import Extension, setup

import argparse
import os
import pathlib

import h5py
import pydevd_pycharm
from mpi4py import MPI
import logging
from os.path import abspath

from hisscube import Photometry as cu
from hisscube.Writer import Writer
import pdb

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# port_mapping = [46139, 40147, 42877, 45603]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

print(os.getpid())

WORK_TAG = 0
FINISHED_TAG = 1


class MPIFileHandler(logging.FileHandler):
    def __init__(self,
                 filename,
                 mode=MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND,
                 encoding='utf-8',
                 delay=False,
                 comm=MPI.COMM_WORLD):
        self.baseFilename = abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.comm = comm
        if delay:
            # We don't open the stream, but we still need to call the
            # Handler constructor to set level, formatter, lock etc.
            logging.Handler.__init__(self)
            self.stream = None
        else:
            logging.StreamHandler.__init__(self, self._open())

    def _open(self):
        stream = MPI.File.Open(self.comm, self.baseFilename, self.mode)
        stream.Set_atomicity(True)
        return stream

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.

        Modification:
            stream is MPI.File, so it must use `Write_shared` method rather
            than `write` method. And `Write_shared` method only accept
            bytestring, so `encode` is used. `Write_shared` should be invoked
            only once in each all of this emit function to keep atomicity.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            stream.Write_shared((msg + self.terminator).encode(self.encoding))
            # self.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        if self.stream:
            self.stream.Sync()
            self.stream.Close()
            self.stream = None


class ParallelWriter(Writer):

    def __init__(self, h5_path=None, h5_file=None):
        super().__init__()
        # mpio
        self.mpio = self.config.getboolean("Handler", "MPIO")
        self.BATCH_SIZE = int(self.config["Writer"]["BATCH_SIZE"])
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()
        self.work_cnt = 0

        logging.basicConfig()
        logging.root.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("rank[%i]" % self.comm.rank)
        mh = MPIFileHandler("logfile.log")
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
        mh.setFormatter(formatter)
        self.logger.addHandler(mh)

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
            if self.mpi_rank == 0:
                self.f = h5py.File(self.h5_path, 'r', driver='mpio', comm=self.comm)
            else:
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
            self.write_spectra_data()
        self.close_h5_file()
        if self.mpi_rank == 0:
            self.open_h5_file_serial()
            self.add_image_refs(self.f)
            self.close_h5_file()

    def write_spectrum_data(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)

        while status.Get_tag() != FINISHED_TAG:
            for image_path in image_path_list:
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path,
                                                                                         self.config.getint(
                                                                                             "Handler", "IMG_ZOOM_CNT"))
                self.file_name = image_path.name
                self.write_img_datasets()
            self.comm.send(obj=None, dest=0)
            image_path_list = self.receive_work(status)

    def receive_work(self, status):
        data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        self.logger.info("Received work no. %02d from master: %d" % (self.work_cnt, hash(str(data))))
        self.work_cnt += 1
        return data

    def write_spectra_data(self):
        status = MPI.Status()
        spectra_path_list = self.receive_work(status)
        while status.Get_tag() != FINISHED_TAG:
            for spec_path in spectra_path_list:
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(
                    spec_path, self.config.getint("Handler", "SPEC_ZOOM_CNT"),
                    apply_rebin=self.config.getboolean("Preprocessing", "APPLY_REBIN"),
                    rebin_min=self.config.getfloat("Preprocessing", "REBIN_MIN"),
                    rebin_max=self.config.getfloat("Preprocessing", "REBIN_MAX"),
                    rebin_samples=self.config.getint("Preprocessing", "REBIN_SAMPLES"),
                    apply_transmission=self.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE"))
                self.file_name = spec_path.name
                self.write_spec_datasets()
            self.comm.send(obj=None, dest=0)
            spectra_path_list = self.receive_work(status)

    def distribute_work(self, path_list):
        status = MPI.Status()
        batches = list(chunks(path_list, self.BATCH_SIZE))
        for i in range(1, self.mpi_size):
            self.send_work(batches, dest=i)
        while len(batches) > 0:
            self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            self.logger.info("Received response from. dest %02d: %d " % (status.Get_source(), self.work_cnt))
            self.send_work(batches, status.Get_source())
        for i in range(1, self.mpi_size):
            self.send_work_finished(dest=i)

    def send_work(self, batches, dest):
        batch = batches.pop()
        tag = WORK_TAG
        self.logger.info("Sending work batch no. %02d to dest %02d: %d " % (self.work_cnt, dest, hash(str(batch))))
        self.comm.send(obj=batch, dest=dest, tag=tag)
        self.work_cnt += 1

    def send_work_finished(self, dest):
        tag = FINISHED_TAG
        self.logger.info("Terminating worker: %0d" % dest)
        self.comm.send(obj=None, dest=dest, tag=tag)


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

    ParallelWriter(args.output_path).ingest_data(fits_image_path, fits_spectra_path, args.truncate)
