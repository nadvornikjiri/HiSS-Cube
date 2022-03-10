import h5py
from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
# import pydevd_pycharm
# port_mapping = [46257, 34673]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)




f = h5py.File("../results/SDSS_cube_c_par.h5", "r+", driver='mpio', comm= MPI.COMM_WORLD, libver="latest")
f["/semi_sparse_cube/5/22/90/362/1450/5802/23208/92835/4482019750.3/8932"]
f.close()
