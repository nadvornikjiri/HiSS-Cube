#bin/bash
mpirun -np 128 --hostfile hosts --map-by node /gpfs/raid/SDSSCube/venv_par/bin/python hisscube.py --truncate ../sdss_data/ results/SDSS_cube_c_par.h5
