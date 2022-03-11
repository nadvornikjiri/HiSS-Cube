from time import sleep

import h5py
from mpi4py import MPI
import sys
sys.path.append('../')

from hisscube.CWriter import CWriter

FITS_IMAGE_PATH = "../data/raw/galaxy_small/images"
FITS_SPECTRA_PATH = "../data/raw/galaxy_small/spectra"
H5_PATH = "../results/SDSS_cube_c.h5"

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
comms = MPI.COMM_WORLD
# import pydevd_pycharm
# port_mapping = [34673, 40197]
# pydevd_pycharm.settrace('localhost', port=port_mapping[rank], stdoutToServer=True, stderrToServer=True)
# print("Rank: %d" %rank)

writer = CWriter(h5_path=H5_PATH)
image_pattern, spectra_pattern = writer.get_path_patterns()
writer.process_metadata(FITS_IMAGE_PATH, image_pattern, FITS_SPECTRA_PATH, spectra_pattern, truncate_file=True)

comms.barrier()
f = h5py.File(H5_PATH, "r+", driver='mpio', comm=comms, libver="latest")
print(f["/semi_sparse_cube/5/22/90/362/1450/5802/23208/92835/4482019750.3/8932"])
f.close()
