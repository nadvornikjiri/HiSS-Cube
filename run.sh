#bin/bash
mpirun -x DARSHAN_CONFIG_PATH=/gpfs/raid/SDSSCube/darshan.conf -x LD_PRELOAD=/gpfs/raid/shared_libs/darshan/darshan-runtime/lib/.libs/libdarshan.so:/gpfs/raid/SDSSCube/ext_lib/hdf5-1.12.0/hdf5/lib/libhdf5.so -np 129 --hostfile hosts --map-by node /gpfs/raid/SDSSCube/venv_par/bin/python /gpfs/raid/SDSSCube/hisscube.py --truncate ../sdss_data/ results/SDSS_cube_c_par_small.h5

