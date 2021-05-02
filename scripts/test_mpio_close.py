import h5py
from mpi4py import MPI

H5PATH = "../data/processed/galaxy_small.h5"

f = h5py.File(H5PATH, 'r+', driver='mpio', comm=MPI.COMM_WORLD)
f.close()
