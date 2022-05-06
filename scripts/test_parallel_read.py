from time import sleep

import h5py

from mpi4py import MPI
import numpy as np
from timeit import default_timer as timer
import os

H5_PATH = "../results/read_test.h5"
TEST_DS_PATH = "/big_contiguos_data"
MAX_CHUNK_SIZE = 2 * 1024 * 1024 * 1024


h5_file = h5py.File(H5_PATH, "r", driver="mpio", comm=MPI.COMM_WORLD)

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# import pydevd_pycharm
# port_mapping = [39337, 33083]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
# print(os.getpid())

dense_cube_ds = h5_file[TEST_DS_PATH]
dense_cube_length = len(dense_cube_ds)
whole_chunk_len = int(dense_cube_length / size)
max_chunk_len = int(MAX_CHUNK_SIZE / dense_cube_ds.dtype.itemsize)
my_chunk_bounds = range(0, whole_chunk_len, max_chunk_len)
for start_idx in my_chunk_bounds:
    start_i = start_idx + (rank * whole_chunk_len)
    if start_idx + max_chunk_len > whole_chunk_len:
        end_i = whole_chunk_len * (rank + 1)
        chunk_len = whole_chunk_len - start_idx
    else:
        end_i = start_i + max_chunk_len
        chunk_len = max_chunk_len
    my_arr = np.zeros((chunk_len,), dense_cube_ds.dtype)
    start = timer()
    dense_cube_ds.read_direct(my_arr, np.s_[start_i: end_i])
    end = timer()
    print("Rank %d: Dataset of size %d read in: %fs, MB/s = %f" % (
        rank, my_arr.nbytes, end - start, (my_arr.nbytes / 1024 / 1024) / (end - start)))
MPI.COMM_WORLD.barrier()
if rank == 0:
    end = timer()
    sleep(1)
    print("Read in total %d bytes, MB/s = %f" % (
    dense_cube_ds.nbytes, (dense_cube_ds.nbytes / 1024 / 1024) / (end - start)))

h5_file.close()
