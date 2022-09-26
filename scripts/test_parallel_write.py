import pydevd_pycharm
from mpi4py import MPI
import os

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
# port_mapping = [33469, 41495]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)

import h5py
import numpy as np
from mpi4py import MPI

print(os.getpid())

H5PATH = "../data/processed/test_parallel.h5"


class ParallelWriter:

    def __init__(self, h5_path):
        self.comm = MPI.COMM_WORLD
        self.mpi_size = self.comm.Get_size()
        self.mpi_rank = self.comm.Get_rank()
        self.h5_path = h5_path
        self.f = None

    def ingest_data(self, truncate_file=None):
        if self.mpi_rank == 0:  # if I'm the master, write all spectrum_metadata and create datasets
            self.open_h5_file_serial()
            self.ingest_metadata()
            self.close_h5_file()  # close the file opened in serial mode
        self.open_h5_file_parallel()  # all, including master, let's open file in mpio mode
        if self.mpi_rank == 0:
            self.distribute_work(self.mpi_size)  # master distributes the work and reads information from the file
            self.write_image_data()
        else:
            self.write_image_data()  # slaves only write the data, in parallel.
        self.close_h5_file()  # closing the mpio opened file hangs.

    def ingest_metadata(self):
        for i in range(self.mpi_size + 1):
            self.f.require_dataset("dataset_%d" % i, (1000, 1000), np.float)

    def open_h5_file_serial(self):
        self.f = h5py.File(self.h5_path, 'r+')

    def open_h5_file_parallel(self):
        self.f = h5py.File(self.h5_path, 'r+', driver='mpio', comm=self.comm)

    def truncate_h5_file(self):
        self.f = h5py.File(self.h5_path, 'w')
        self.f.close()

    def distribute_work(self, mpi_size):
        for dest in range(1, mpi_size):
            print("Sending work to dest %02d " % dest)
            self.comm.send(obj="test", dest=dest)

    def write_image_data(self):
        status = MPI.Status()
        message = self.receive_work(status)
        img_data = np.zeros((1000, 1000))

        self.f["dataset_%d" % self.mpi_rank].write_direct(img_data)
        print("wrote data to ds: %d" % self.mpi_rank)

    def receive_work(self, status):
        message = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        print("Received message from master: %s" % message)
        return message


writer = ParallelWriter(H5PATH)
writer.ingest_data(truncate_file=True)
