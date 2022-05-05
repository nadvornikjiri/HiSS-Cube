import h5py
import numpy as np

H5_FILE = "../results/read_test.h5"
SIZE = 100 * 1024 * 1024 * 1024 # 400 GB

h5_file = h5py.File(H5_FILE, "w")
ds = h5_file.create_dataset("big_contiguos_data", (SIZE,), dtype='f4')
ds.write_direct(np.ones(SIZE,dtype=float))
h5_file.close()
