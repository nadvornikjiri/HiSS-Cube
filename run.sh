#bin/bash
dir=$(pwd -P)
mpirun -x DARSHAN_CONFIG_PATH=${dir}/darshan.conf -x LD_PRELOAD=${dir}/../shared_libs/darshan/darshan-runtime/lib/.libs/libdarshan.so:${dir}/ext_lib/hdf5-1.12.2/hdf5/lib/libhdf5.so -x LD_LIBRARY_PATH=${dir}/ext_lib/hdf5-1.12.2/hdf5/lib:${LD_LIBRARY_PATH} -np "$1" --hostfile hosts1 --map-by node ${dir}/venv_par/bin/python ${dir}/hisscube.py "${@:2}"
