#bin/bash
mpirun -x DARSHAN_CONFIG_PATH=/gpfs/raid/SDSSCube/darshan.conf -x LD_PRELOAD=/gpfs/raid/shared_libs/darshan/darshan-runtime/lib/.libs/libdarshan.so:/gpfs/raid/SDSSCube/ext_lib/hdf5-1.12.0/hdf5/lib/libhdf5.so -np "$1" --hostfile /gpfs/raid/SDSSCube/hosts --map-by node /gpfs/raid/SDSSCube/venv_par/bin/python /gpfs/raid/SDSSCube/scripts/test_parallel_read.py
