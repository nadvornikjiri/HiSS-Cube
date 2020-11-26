# HiSS-Cube
Software package for handling multi-dimensional multi-resolution data within HDF5.

## Installation instructions

1. Installing dependencies
```bash
sudo apt-get update
sudo apt-get install python3-pip
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

## Download data from Zenodo
Download the data from zenodo and extract the contents of "input" archive to the "data" folder. All of the tests in the scripts/tests folder will pass afterwards and the Ipython notebook pre-set paths will work as well.


## Running the IPython notebook

1. Run the IPython notebook
```bash
jupyter notebook
````

2. Open the SDSS Cube.ipynb file.

3. Run all of the cells. Note that you should have already opened TOPCAT if running the last cell that tries to send the data via SAMP.



