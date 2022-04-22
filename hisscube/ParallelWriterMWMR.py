import os
from pathlib import Path

import h5py
from mpi4py import MPI
from tqdm import tqdm

from hisscube.ParallelWriter import ParallelWriter, chunks
from timeit import default_timer as timer
import time
import cProfile, pstats

import os


def measured_time():
    times = os.times()
    # return times.elapsed - (times.system + times.user)
    return times()


def profile(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile(measured_time)
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result

        return wrap_f

    return prof_decorator


class ParallelWriterMWMR(ParallelWriter):
    def ingest(self, image_path, spectra_path, image_pattern=None, spectra_pattern=None, truncate_file=None):
        self.process_metadata(image_path, image_pattern, spectra_path, spectra_pattern, truncate_file)
        self.process_data()
        if self.config.getboolean("Writer", "CREATE_REFERENCES"):
            self.add_region_references()
        if self.config.getboolean("Writer", "CREATE_DENSE_CUBE"):
            self.create_dense_cube()

    @profile(filename="profile_process_metadata")
    def process_metadata(self, image_path, image_pattern, spectra_path, spectra_pattern, truncate_file, no_attrs=False,
                         no_datasets=False):
        image_pattern, spectra_pattern = self.get_path_patterns(image_pattern, spectra_pattern)
        if self.mpi_rank == 0:
            self.logger.info("Writing metadata.")
            self.open_h5_file_serial(truncate=truncate_file)
            self.ingest_metadata(image_path, spectra_path, image_pattern, spectra_pattern)
            self.close_h5_file()
            self.metadata_timings_log_csv_file.close()
        self.barrier(self.comm)

    @profile(filename="profile_process_data")
    def process_data(self):
        self.open_h5_file_parallel()
        start = timer()
        if self.mpi_rank == 0:
            self.logger.info("Processing images.")
            self.distribute_work(self.image_path_list, "image")
        else:
            self.write_image_data()
        self.barrier(self.comm)
        if self.mpi_rank == 0:
            self.logger.info("Processing spectra.")
            self.distribute_work(self.spectra_path_list, "spectrum")
            self.data_timings_log_csv_file.close()
        else:
            self.write_spectra_data()
        self.barrier(self.comm)
        self.close_h5_file()
        end = timer()
        self.logger.info("Parallel part time: %s", end - start)

    def add_region_references(self):
        if self.mpi_rank == 0:
            self.logger.debug("Adding image region references.")
            self.open_h5_file_serial()
            self.add_image_refs(self.f)
            self.close_h5_file()

    def open_and_truncate(self):
        self.f = h5py.File(self.h5_path, 'w', libver="latest")

    def write_spectrum_data(self, spec_path):
        return super().ingest_spectrum(spec_path)

    def write_image_data(self):
        status = MPI.Status()
        image_path_list = self.receive_work(status)

        while status.Get_tag() != self.KILL_TAG:
            for image_path in image_path_list:
                self.logger.debug("Rank %02d: Processing image %s." % (self.mpi_rank, image_path))
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
                self.logger.debug("Rank %02d: Processing spectrum %s." % (self.mpi_rank, spec_path))
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
        self.logger.debug("Terminating worker: %0d" % dest)
        self.comm.send(obj=None, dest=dest, tag=tag)

    def distribute_work(self, path_list, batch_type):
        status = MPI.Status()
        batches = list(chunks(path_list, self.BATCH_SIZE))
        for i in tqdm(range(1, len(batches) + 1)):
            if i < (self.mpi_size):
                self.send_work(batches, dest=i)
                self.active_workers += 1
            else:
                self.wait_for_message(source=MPI.ANY_SOURCE, tag=self.FINISHED_TAG, status=status)
                self.process_response(batch_type, status)
                self.send_work(batches, status.Get_source())
        for i in range(1, self.mpi_size):
            self.send_work_finished(dest=i)
        for i in range(0, self.active_workers):
            self.process_response(batch_type, status)
        self.active_workers = 0

    def process_response(self, batch_type, status):
        self.comm.recv(source=MPI.ANY_SOURCE, tag=self.FINISHED_TAG, status=status)
        end = timer()
        if batch_type == "image":
            self.image_batch_cnt += 1
        if batch_type == "spectrum":
            self.spectrum_batch_cnt += 1
        self.log_data_csv_timing(
            end - self.start_timers[status.Get_source()], self.image_batch_cnt, self.spectrum_batch_cnt)
        self.logger.debug("Received response from. dest %02d: %d " % (status.Get_source(), self.sent_work_cnt))

    def barrier(self, comm, tag=0):
        sleep = self.config.getfloat("Writer", "POLL_INTERVAL")
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
