
#include "hdf5.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define H5PATH "../../results/sdss.h5"
#define FILE_NAME "sdss.txt"  // the input file (contains 200,000 lines)
#define LOG_FILE_NAME "timings.csv"
#define MAX_COUNT 1200000         // the max. number of datasets to create
#define MAX_LINE_LENGTH 512   // assumption about the line length
#define CHUNKED_LAYOUT  1     // use chunked layout
#define LOG_CHUNK 100

// convenience structure for dataset creation
struct descr_t {
  char path[MAX_LINE_LENGTH];
  int  dims[3];
};

int main()
{
  // parse the input file
  FILE *fp = fopen(FILE_NAME, "r");
  FILE *logfp = fopen(LOG_FILE_NAME, "w+");
  clock_t start, end;
  double elapsed_time;

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

  // libver latest
  hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

  //page buffering
  hid_t fcpl = H5Pcreate(H5P_FILE_CREATE);
  H5Pset_file_space_strategy(fcpl, H5F_FSPACE_STRATEGY_PAGE, 0, 1);
  hid_t hfile = H5Fcreate(H5PATH, H5F_ACC_TRUNC, fcpl, fapl);
  hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(lcpl, 1);
  hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_alloc_time(dcpl, H5D_ALLOC_TIME_EARLY);
  H5Pset_fill_time(dcpl, H5D_FILL_TIME_NEVER);
#ifdef CHUNKED_LAYOUT
  H5Pset_chunk(dcpl, 3, (hsize_t[]) {128, 128, 2});
#endif

  hid_t fspace;


  fprintf(logfp,"Dataset count,Time\n");
  start = clock();
  for (size_t ii = 0; ii < i; ++ii) {
    fspace = H5Screate_simple(3, (hsize_t[]) {doit[ii].dims[0],
                                              doit[ii].dims[1],
                                              doit[ii].dims[2]}, NULL);
    H5Dclose(H5Dcreate(hfile, doit[ii].path, H5T_NATIVE_FLOAT, fspace,
                       lcpl, dcpl, H5P_DEFAULT));
    H5Sclose(fspace);

    //timing logging
    if (ii % LOG_CHUNK == 0 && ii > 0){
    	end = clock();
    	elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    	fprintf(logfp, "%zu,%f\n", ii, elapsed_time);
    	start = end;
    }
  }

  H5Pclose(dcpl);
  H5Pclose(lcpl);

  fclose(logfp);
  H5Fclose(hfile);

  free(doit);

  return 0;
}
