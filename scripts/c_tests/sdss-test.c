
#include "hdf5.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FILE_NAME "sdss.txt"  // the input file (contains 200,000 lines)
#define MAX_COUNT 200000         // the max. number of datasets to create
#define MAX_LINE_LENGTH 512   // assumption about the line length

// convenience structure for dataset creation
struct descr_t {
  char path[MAX_LINE_LENGTH];
  int  dims[3];
};

int main()
{
  // parse the input file
  FILE *fp = fopen(FILE_NAME, "r");

  if (fp == NULL) {
    printf("Error: could not open file %s", FILE_NAME);
    return 1;
  }

  struct descr_t* doit = malloc(MAX_COUNT * sizeof(struct descr_t));
  size_t i = 0;
  char buffer[MAX_LINE_LENGTH];

  while (fgets(buffer, MAX_LINE_LENGTH, fp)) {
    strcpy(doit[i].path, strtok(buffer, "@"));
    sscanf(strtok(NULL, "@"), " {%d, %d, %d}",
           doit[i].dims, doit[i].dims+1, doit[i].dims+2);
    ++i;

    if (i == MAX_COUNT)
      break;
  }

  // at this point, I contains the length of the DOIT array

  fclose(fp);

  // create the HDF5 file and loop invariants
  hid_t hfile = H5Fcreate("sdss.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(lcpl, 1);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
  H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);

  // TODO Support chunked datasets (What was the chunk size?)

  hid_t fspace;

  for (size_t ii = 0; ii < i; ++ii) {
    fspace = H5Screate_simple(3, (hsize_t[]) {doit[ii].dims[0],
                                              doit[ii].dims[1],
                                              doit[ii].dims[2]}, NULL);
    H5Dclose(H5Dcreate(hfile, doit[ii].path, H5T_NATIVE_FLOAT, fspace,
                       lcpl, dcpl, H5P_DEFAULT));
    H5Sclose(fspace);
  }

  H5Pclose(dcpl);
  H5Pclose(lcpl);

  H5Fclose(hfile);

  free(doit);

  return 0;
}
