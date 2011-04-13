#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "gol_data.h"

struct _gol_data
{
  int   rows;
  int   cols;
  int*  data_curr;
  int*  data_next;
  int   data[];
};

#ifdef GPU
extern "C" {
#endif

gol_data* 
gol_data_from_l(
  const char* path)
{
  assert(path != NULL);
  assert(path[0] != '\0');

  FILE*      l_file;
  gol_data*  new_gol;
  int        rows;
  int        cols;
  bool       know_cols;
  int        c_char;
  int        d_ptr;

  rows = 0;
  cols = 0;
  know_cols = false;

  l_file = fopen(path,"rt");

  while ((c_char = fgetc(l_file)) != EOF) {
    if (c_char == '\n') {
      rows += 1;
      know_cols = true;
    } else {
      if (!know_cols) {
	cols += 1;
      }
    }
  }

  new_gol = (gol_data*)malloc(sizeof(gol_data) + sizeof(int) * 2 * rows * cols);

  new_gol->rows = rows;
  new_gol->cols = cols;
  new_gol->data_curr = new_gol->data;
  new_gol->data_next = new_gol->data + rows * cols;

  fseek(l_file,0,SEEK_SET);

  d_ptr = 0;

  while ((c_char = fgetc(l_file)) != EOF) {
    if (c_char == '.') {
      new_gol->data[d_ptr] = DEAD;
      new_gol->data[rows * cols + d_ptr] = DEAD;
      d_ptr += 1;
    } else if (c_char == 'X') {
      new_gol->data[d_ptr] = ALIVE;
      new_gol->data[rows * cols + d_ptr] = ALIVE;
      d_ptr += 1;
    } else if (c_char == '\n') {
    } else {
      assert(0);
    }
  }

  fclose(l_file);

  return new_gol;
}

void
gol_data_free(
  gol_data* gol)
{
  assert(gol_data_is_valid(gol));

  gol->rows = -1;
  gol->cols = -1;
  gol->data_curr = NULL;
  gol->data_next = NULL;

  free(gol);
}


bool
gol_data_is_valid(
  const gol_data* gol)
{
  int  i;
  int  j;

  if (gol == NULL) {
    return false;
  }

  if (gol->rows < 1) {
    return false;
  }

  if (gol->cols < 1) {
    return false;
  }
  
  if (!((gol->data_curr == gol->data &&
	 gol->data_next == gol->data + gol->rows * gol->cols) ||
	(gol->data_curr == gol->data + gol->rows * gol->cols &&
	 gol->data_next == gol->data))) {
    return false;
  }

  for (i = 0; i < gol->rows; i++) {
    for (j = 0; j < gol->cols; j++) {
      if (gol->data[i * gol->cols + j] != ALIVE &&
	  gol->data[i * gol->cols + j] != DEAD) {
	return false;
      }
    }
  }

  return true;
}


#ifdef CPU
gol_data*
gol_data_evolve(
  gol_data* gol,
  int iters)
{
  assert(gol_data_is_valid(gol));
  assert(iters > 0);

  int*  temp;
  int   count;
  int   i;
  int   j;

  while (iters > 0) {
    for (i = 0; i < gol->rows; i++) {
      for (j = 0; j < gol->cols; j++) {
	count = gol_data_neigh_count(gol,i,j);

	if (gol_data_get(gol,i,j) == ALIVE) {
	  if (count < 2) {
	    gol->data_next[i * gol->cols + j] = DEAD;
	  } else if (count == 2 || count == 3) {
	    gol->data_next[i * gol->cols + j] = ALIVE;
	  } else {
	    gol->data_next[i * gol->cols + j] = DEAD;
	  }
	} else {
	  if (count == 3) {
	    gol->data_next[i * gol->cols + j] = ALIVE;
	  } else {
	    gol->data_next[i * gol->cols + j] = DEAD;
	  }
	}
      }
    }

    temp = gol->data_curr;
    gol->data_curr = gol->data_next;
    gol->data_next = temp;
    iters = iters - 1;
  }

  return gol;
}

int
gol_data_neigh_count(
  const gol_data* gol,
  int  i,
  int  j)
{
  assert(gol_data_is_valid(gol));
  assert(i >= 0 && i < gol->rows);
  assert(j >= 0 && j < gol->cols);

  int  ip1;
  int  im1;
  int  jp1;
  int  jm1;

  ip1 = (i != gol->rows - 1) ? (i + 1) : 0;
  im1 = (i != 0) ? (i - 1) : (gol->rows - 1);
  jp1 = (j != gol->cols - 1) ? (j + 1) : 0;
  jm1 = (j != 0) ? (j - 1) : (gol->cols - 1);

  return gol_data_get(gol,im1,j) + 
         gol_data_get(gol,im1,jp1) +
         gol_data_get(gol,i,jp1) +
         gol_data_get(gol,ip1,jp1) +
         gol_data_get(gol,ip1,j) +
         gol_data_get(gol,ip1,jm1) +
         gol_data_get(gol,i,jm1) +
         gol_data_get(gol,im1,jm1);
}
#endif

#ifdef GPU
__global__ void
_gol_data_evolve(
  int* curr,
  int* next,
  int rows,
  int cols)
{
  int  i = blockIdx.x * blockDim.x + threadIdx.x;
  int  j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    int   ip1 = (i != rows - 1) ? (i + 1) : 0;
    int   im1 = (i != 0) ? (i - 1) : (rows - 1);
    int   jp1 = (j != cols - 1) ? (j + 1) : 0;
    int   jm1 = (j != 0) ? (j - 1) : (cols - 1);
    int   count;

    count = curr[im1 * cols + j] +
            curr[im1 * cols + jp1] +
            curr[i * cols + jp1] +
            curr[ip1 * cols + jp1] +
            curr[ip1 * cols + j] +
            curr[ip1 * cols + jm1] +
            curr[i * cols + jm1] +
            curr[im1 * cols + jm1];

    if (curr[i * cols + j] == ALIVE) {
      if (count < 2) {
	next[i * cols + j] = DEAD;
      } else if (count == 2 || count == 3) {
	next[i * cols + j] = ALIVE;
      } else {
	next[i * cols + j] = DEAD;
      }
    } else {
      if (count == 3) {
	next[i * cols + j] = ALIVE;
      } else {
	next[i * cols + j] = DEAD;
      }
    }
  }
}

gol_data*
gol_data_evolve(
  gol_data* gol,
  int iters)
{
  assert(gol_data_is_valid(gol));
  assert(iters > 0);

  int*  gpu_data;
  int*  gpu_curr;
  int*  gpu_next;
  int*  temp;
  int   block_dim_x = 16;
  int   block_dim_y = 16;
  dim3  block = dim3(block_dim_x,block_dim_y);
  int   grid_dim_x = (int)ceil((float)gol->rows / block_dim_x);
  int   grid_dim_y = (int)ceil((float)gol->cols / block_dim_y);
  dim3  grid = dim3(grid_dim_x,grid_dim_y);
  int   linear_size = gol->rows * gol->cols;
  int   i;

  cudaMalloc((void**)&gpu_data,2 * sizeof(int) * linear_size);
  cudaMemcpy(gpu_data,gol->data_curr,sizeof(int) * linear_size,cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_data + linear_size,gol->data_next,sizeof(int) * linear_size,cudaMemcpyHostToDevice);

  gpu_curr = gpu_data;
  gpu_next = gpu_data + linear_size;

  for (i = 0; i < iters; i++) {
    _gol_data_evolve <<<grid,block>>> (gpu_curr,gpu_next,gol->rows,gol->cols);
    cudaThreadSynchronize();

    temp = gpu_curr;
    gpu_curr = gpu_next;
    gpu_next = temp;
  }

  cudaMemcpy(gol->data,gpu_data,2 * sizeof(int) * linear_size,cudaMemcpyDeviceToHost);

  cudaFree(gpu_data);

  if (iters % 2 == 0) {
    gol->data_curr = gol->data;
    gol->data_next = gol->data + linear_size;
  } else {
    gol->data_curr = gol->data + linear_size;
    gol->data_next = gol->data;
  }

  return gol;
}
#endif

int
gol_data_get_rows(
  const gol_data* gol)
{
  assert(gol_data_is_valid(gol));
  
  return gol->rows;
}

int
gol_data_get_cols(
  const gol_data* gol)
{
  assert(gol_data_is_valid(gol));
  
  return gol->cols;
}

int
gol_data_get(
  const gol_data* gol,
  int i,
  int j)
{
  assert(gol_data_is_valid(gol));
  assert(i >= 0 && i < gol->rows);
  assert(j >= 0 && j < gol->cols);

  return gol->data_curr[i * gol->cols + j];
}

void
gol_data_set(
  gol_data* gol,
  int i,
  int j,
  int cell_state)
{
  assert(gol_data_is_valid(gol));
  assert(i >= 0 && i < gol->rows);
  assert(j >= 0 && j < gol->cols);
  assert(cell_state == ALIVE || cell_state == DEAD);

  gol->data_curr[i * gol->cols + j] = cell_state;
}


void
gol_data_save_l(
  const gol_data* gol,
  const char* path)
{
  assert(gol_data_is_valid(gol));
  assert(path != NULL);
  assert(path[0] != '\0');

  FILE*  l_file;
  int    i;
  int    j;

  l_file = fopen(path,"wt");

  for (i = 0; i < gol->rows; i++) {
    for (j = 0; j < gol->cols; j++) {
      if (gol_data_get(gol,i,j) == ALIVE) {
	fprintf(l_file,"X");
      } else {
	fprintf(l_file,".");
      }
    }
    
    fprintf(l_file,"\n");
  }

  fclose(l_file);
}

#ifdef GPU
}
#endif
