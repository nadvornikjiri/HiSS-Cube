from distutils.core import setup, Extension
import sysconfig


include_dirs = ['/home/caucau/SDSSCube/ext_lib/hdf5_tar/hdf5-1.12.0/hdf5/include',
                '/usr/lib/x86_64-linux-gnu/openmpi/include',
                '/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent',
                '/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include']

library_dirs = ['/home/caucau/SDSSCube/ext_lib/hdf5_tar/hdf5-1.12.0/hdf5/lib']
libraries = ['hdf5', 'hdf5_hl']
h5writer_module = Extension('h5writer',
                            include_dirs=include_dirs,
                            library_dirs=library_dirs,
                            libraries=libraries,
                            sources=['c_boosters/h5writer/writer.c', 'c_boosters/h5writer/main.c'])

setup(name='h5writer',
      version='1.0',
      description='HDF5 write booster',
      ext_modules=[h5writer_module])
