# ring_mpe.py

"""
Demonstrates mpi4py profiling with MPE.

Run this with 8 processes like:
$ mpiexec -n 8 python ring_mpe.py
"""

import os
os.environ['MPE_LOGFILE_PREFIX'] = 'ring'
import mpi4py
mpi4py.profile('mpe')
print("test")
# or
# mpi4py.profile('mpe', logfile='ring')

from mpi4py import MPI
from array import array

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()