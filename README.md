# HiSS-Cube
Software package for handling multi-dimensional multi-resolution data within HDF5.

## Installation instructions

1. Installing dependencies
```bash
sudo apt-get update
sudo apt-get install python3-pip
sudo apt-get install libbz2-dev
sudo apt-get install -y libsm6
sudo apt-get install libfontconfig1 libxrender1
```

2. Download code
```bash
git clone git@github.com:nadvornikjiri/HiSS-Cube.git
cd SDSSCube
```
4.Create virtual environment
```bash
pip3 install virtualenv
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Development version - parallel implementation with MPIO
1. Download h5py and hdf5 into ext_lib folder within the SDSSCube git folder. 
  1. https://github.com/h5py/h5py.git
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
python setup_configure.py --mpi
pip uninstall h5py
python setup.py install
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



