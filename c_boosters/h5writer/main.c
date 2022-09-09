#include "main.h"
#include "hdf5.h"
#include "writer.h"

#include <stdio.h>

int write_hdf5_metadata(PyObject *self, PyObject *args){
    process_h5_dict(self, args);
    return 0;
}

static PyObject *py_write_hdf5_metadata(PyObject *self, PyObject *args) {
    write_hdf5_metadata(self, args);
    Py_RETURN_NONE;
}


PyMethodDef methods[] = {
        {
                "write_hdf5_metadata",
                (PyCFunction) py_write_hdf5_metadata,
                METH_VARARGS,
                "Say Import HDF5!",
        },
        {NULL}
};

PyDoc_STRVAR(write_hdf5_metadata_doc, "Provides API for writing HDF5 file");

PyModuleDef h5writer_module = {
        PyModuleDef_HEAD_INIT,
        "h5writer",
        write_hdf5_metadata_doc,
        -1,
        methods,
        NULL,
        NULL,
        NULL,
        NULL
};


PyMODINIT_FUNC PyInit_h5writer(void) {
    return PyModule_Create(&h5writer_module);
};
