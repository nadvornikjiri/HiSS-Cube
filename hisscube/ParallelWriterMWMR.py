import os
from pathlib import Path

import h5py
from mpi4py import MPI
from tqdm import tqdm

from hisscube.ParallelWriter import ParallelWriter, chunks
from timeit import default_timer as timer
import time

print(os.getpid())


def barrier(comm, tag=0, sleep=0.01):
    size = comm.Get_size()
    if size == 1:
        return
    rank = comm.Get_rank()
    mask = 1
    while mask < size:
        dst = (rank + mask) % size
        src = (rank - mask + size) % size
        req = comm.isend(None, dst, tag)
        while not comm.Iprobe(src, tag):
            time.sleep(sleep)
        comm.recv(None, src, tag)
        req.Wait()
        mask <<= 1


class ParallelWriterMWMR(ParallelWriter):
    def ingest_data(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, truncate_file=None):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        if self.mpi_rank == 0:
            self.logger.info("Writing metadata.")
            self.open_h5_file_serial(truncate=truncate_file)
            self.ingest_metadata(image_path, spectra_path, image_pattern, spectra_pattern)
            self.close_h5_file()
        barrier(self.comm)
        self.parse_path_lists(image_path, spectra_path, image_pattern, spectra_pattern)
        self.open_h5_file_parallel()
        start = timer()
        if self.mpi_rank == 0:
            self.logger.info("Processing images.")
            self.distribute_work(self.image_path_list)
        else:
            self.write_image_data()
        barrier(self.comm)
        if self.mpi_rank == 0:
            self.logger.info("Processing spectra.")
            self.distribute_work(self.spectra_path_list)
        else:
            self.write_spectra_data()
        barrier(self.comm)
        self.close_h5_file()
        end = timer()
        if self.mpi_rank == 0:
            self.logger.info("Adding image region references.")
            self.open_h5_file_serial()
            self.add_image_refs(self.f)
            self.close_h5_file()
        self.logger.info("Parallel part time: %s", end - start)

    def parse_path_lists(self, image_path, spectra_path, image_pattern, spectra_pattern):
        image_path_list = list(Path(image_path).rglob(image_pattern))
        spectra_path_list = list(Path(spectra_path).rglob(spectra_pattern))
        self.image_path_list = [str(e) for e in image_path_list]
        self.spectra_path_list = [str(e) for e in spectra_path_list]

    def open_and_truncate(self):
        self.f = h5py.File(self.h5_path, 'w')

    def write_spectrum_data(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)

        while status.Get_tag() != self.KILL_TAG:
            for image_path in image_path_list:
                # self.logger.info("Rank %02d: Processing image %s." % (self.mpi_rank, image_path))
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_image(image_path,
                                                                                         self.config.getint(
                                                                                             "Handler", "IMG_ZOOM_CNT"))
                self.file_name = image_path.split('/')[-1]
                self.write_img_datasets()
            self.comm.send(obj=None, tag=self.FINISHED_TAG, dest=0)
            image_path_list = self.receive_work(status)

    def write_spectra_data(self):
        status = MPI.Status()
        spectra_path_list = self.receive_work(status)
        while status.Get_tag() != self.KILL_TAG:
            for spec_path in spectra_path_list:
                # self.logger.info("Rank %02d: Processing spectrum %s." % (self.mpi_rank, spec_path))
                self.metadata, self.data = self.cube_utils.get_multiple_resolution_spectrum(
                    spec_path, self.config.getint("Handler", "SPEC_ZOOM_CNT"),
                    apply_rebin=self.config.getboolean("Preprocessing", "APPLY_REBIN"),
                    rebin_min=self.config.getfloat("Preprocessing", "REBIN_MIN"),
                    rebin_max=self.config.getfloat("Preprocessing", "REBIN_MAX"),
                    rebin_samples=self.config.getint("Preprocessing", "REBIN_SAMPLES"),
                    apply_transmission=self.config.getboolean("Preprocessing", "APPLY_TRANSMISSION_CURVE"))
                self.file_name = spec_path.split('/')[-1]
                self.write_spec_datasets()
            self.comm.send(obj=None, tag=self.FINISHED_TAG, dest=0)
            spectra_path_list = self.receive_work(status)

    def send_work_finished(self, dest):
        tag = self.KILL_TAG
        self.logger.info("Terminating worker: %0d" % dest)
        self.comm.send(obj=None, dest=dest, tag=tag)

    def distribute_work(self, path_list):
        status = MPI.Status()
        batches = list(chunks(path_list, self.BATCH_SIZE))
        for i in tqdm(range(1, len(batches))):
            if i < (self.mpi_size):
                self.send_work(batches, dest=i)
            else:
                self.wait_for_message(source=MPI.ANY_SOURCE, tag=self.FINISHED_TAG, status=status)
                self.comm.recv(source=MPI.ANY_SOURCE, tag=self.FINISHED_TAG, status=status)
                self.logger.info("Received response from. dest %02d: %d " % (status.Get_source(), self.sent_work_cnt))
                self.send_work(batches, status.Get_source())
        for i in range(1, self.mpi_size):
            self.send_work_finished(dest=i)
