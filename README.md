# HiSS-Cube
Software package for handling Hierarchical Semi-Sparse data within HDF5. This framework and its usage is described in detail in the article [HiSS-Cube](https://www.researchgate.net/publication/350585859_HiSS-Cube_A_scalable_framework_for_Hierarchical_Semi-Sparse_Cubes_preserving_uncertainties), while the parallel version is described in this [ADASS poster](https://adass2021.ac.za/uploads/X1-015/upload/X1-015_latest.pdf).

## Installation instructions

1. Installing dependencies
```bash
apt-get update
apt-get install -y python3-pip libbz2-dev libsm6 libfontconfig1 libxrender1 libopenmpi-dev ffmpeg libsm6 libxext6
```

### Development version - parallel implementation with MPIO
1. Download h5py and hdf5 into ext_lib folder within the SDSSCube git folder. 
  1. ```mkdir h5py && cd h5py && git clone https://github.com/h5py/h5py.git . ```
  2. Latest tar release of HDF5 from here: https://www.hdfgroup.org/downloads/hdf5/source-code/.
2. Build & Install HDF5 parallel
```
./configure --enable-build-mode=debug --enable-parallel --enable-codestack
make -j8
make install
```
3. Build & Install h5py
```
export CC=mpicc
export HDF5_MPI="ON"
export HDF5_DIR=~/SDSSCube/ext_lib/hdf5-1.12.1/
export LD_LIBRARY_PATH=~/SDSSCube/ext_lib/hdf5-1.12.1/hdf5/lib:$LD_LIBRARY_PATH
python setup_configure.py --mpi
pip uninstall h5py
python setup.py install
```

2. Download code
```bash
mkdir SDSSCube
cd SDSSCube
git clone https://github.com/nadvornikjiri/HiSS-Cube.git .
```
4.Create virtual environment
```bash
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```




## Download data from Zenodo
Download the "data.tar.gz" from [Zenodo HiSS Cube](https://zenodo.org/record/4273993#.X8ESdWhKiUk) and extract the contentse ìnside the git repository. The "data" folder should afterwards contain "galaxy_small", "images", "spectra", etc. All of the tests in the scripts/tests folder will pass afterwards and the Ipython notebook pre-set paths will work as well.


## Running the IPython notebook

1. Run the IPython notebook
```bash
jupyter notebook
````

2. Open the SDSS Cube.ipynb file.

3. Run all of the cells. Note that you should have already opened TOPCAT if running the last cell that tries to send the data via SAMP.



