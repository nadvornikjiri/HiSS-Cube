import pathlib
from distutils.core import setup, Extension

setup_path = pathlib.Path(__file__).parent.absolute()

hdf5_version = "1.13.1"

include_dirs = ['%s/ext_lib/hdf5-%s/hdf5/include' % (setup_path, hdf5_version),
                '/usr/lib/x86_64-linux-gnu/openmpi/include',
                '/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent',
                '/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include']

library_dirs = ['%s/ext_lib/hdf5-%s/hdf5/lib' % (setup_path, hdf5_version)]
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
