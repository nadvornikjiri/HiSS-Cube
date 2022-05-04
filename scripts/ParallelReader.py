import h5py

from mpi4py import MPI
import numpy as np
from timeit import default_timer as timer
import os

H5_PATH = "../data/processed/galaxy_small.h5"

h5_file = h5py.File(H5_PATH, "r", driver="mpio", comm=MPI.COMM_WORLD)

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()

# import pydevd_pycharm
# port_mapping = [39337, 33083]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
# print(os.getpid())

dense_cube_ds = h5_file["/dense_cube/0/visualization/dense_cube_zoom_0"]
dense_cube_length = len(dense_cube_ds)
my_chunk_size = int(dense_cube_length / size)
start_i = int(rank * my_chunk_size)
end_i = (rank + 1) * my_chunk_size
my_arr = np.zeros((my_chunk_size,), dense_cube_ds.dtype)
start = timer()
dense_cube_ds.read_direct(my_arr, np.s_[start_i: end_i])
h5_file.close()
end = timer()
print("Rank %d: Dataset of size %d read in: %fs, MB/s = %f" % (
    rank, my_arr.nbytes, end - start, (my_arr.nbytes / 1024 / 1024) / (end - start)))
