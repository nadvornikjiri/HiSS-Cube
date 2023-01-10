#!/bin/bash

run_dir="/scratch/project/dd-22-88/run_dir"
dir="/home/caucau/SDSSCube"
h5_path=results/SDSS_cube.h5
actions="data"
nodes=6
np=$nodes*128

ml Python/3.8.2-GCCcore-9.3.0
ml OpenMPI/4.1.4-GCC-11.3.0

cd ${run_dir}
#rm ${run_dir}/${h5_path}

# execute the calculation

for action in $actions
do
	mpirun 	-x DARSHAN_CONFIG_PATH=${dir}/darshan.conf \
		-x LD_PRELOAD=${dir}/../shared_libs/darshan/darshan-runtime/lib/.libs/libdarshan.so:${dir}/ext_lib/hdf5-1.13.1/hdf5/lib/libhdf5.so \
		-x LD_LIBRARY_PATH=${dir}/ext_lib/hdf5-1.13.1/hdf5/lib:${LD_LIBRARY_PATH} \
		-np ${np} \
		${dir}/venv_par/bin/python \
		${dir}/hisscube.py \
		${run_dir}/sdss_data \
		${run_dir}/${h5_path} \
		update --${action}
done

exit
