import h5py
from mpi4py import MPI
import numpy as np

H5PATH = "../results/SDSS_cube_parallel.h5"
f = h5py.File(H5PATH, 'r+', driver='mpio', comm=MPI.COMM_WORLD)

rank = MPI.COMM_WORLD.Get_rank()

if rank == 0:
    print("writing no: %d" % rank)
    ds = f["/semi_sparse_cube/5/22/90/362/1450/5802/23208/92832/4604806771.19/3551/(2048, 1489)/frame-u-004899-2-0260.fits"]
    ds.write_direct(np.ones((1489, 2048, 2)))
if rank == 1:
    print("writing no: %d" % rank)
    ds = f["/semi_sparse_cube/5/22/90/362/1450/5802/23208/92832/4604806842.83/8932/(2048, 1489)/frame-z-004899-2-0260.fits"]
    ds.write_direct(np.ones((1489, 2048, 2)))
if rank == 2:
    print("writing no: %d" % rank)
    ds = f["/semi_sparse_cube/5/22/90/362/1450/5802/23208/92835/371341/1485364/5941459/23765838/95063352/380253411/1521013644/6084054576/4481683956.2/4620/spec-0412-51871-0308.fits"]
    ds.write_direct(np.ones((4620, 3)))
f.close()
