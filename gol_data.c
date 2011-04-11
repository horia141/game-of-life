#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "gol_data.h"

struct _gol_data
{
  int  rows;
  int  cols;
  int  data[];
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

  new_gol = (gol_data*)malloc(sizeof(gol_data) + sizeof(int) * rows * cols);

  new_gol->rows = rows;
  new_gol->cols = cols;

  fseek(l_file,0,SEEK_SET);

  d_ptr = 0;

  while ((c_char = fgetc(l_file)) != EOF) {
    if (c_char == '.') {
      new_gol->data[d_ptr] = DEAD;
      d_ptr += 1;
    } else if (c_char == 'X') {
      new_gol->data[d_ptr] = ALIVE;
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
  gol_data* dst,
  const gol_data* src)
{
  assert(gol_data_is_valid(dst));
  assert(gol_data_is_valid(src));
  assert(dst->rows == src->rows);
  assert(dst->cols == src->cols);

  int  count;
  int  i;
  int  j;

  for (i = 0; i < dst->rows; i++) {
    for (j = 0; j < dst->cols; j++) {
      count = gol_data_neigh_count(src,i,j);

      if (gol_data_get(src,i,j) == ALIVE) {
	if (count < 2) {
	  gol_data_set(dst,i,j,DEAD);
	} else if (count == 2 || count == 3) {
	  gol_data_set(dst,i,j,ALIVE);
	} else {
	  gol_data_set(dst,i,j,DEAD);
	}
      } else {
	if (count == 3) {
	  gol_data_set(dst,i,j,ALIVE);
	} else {
	  gol_data_set(dst,i,j,DEAD);
	}
      }
    }
  }

  return dst;
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
  int* dst_data,
  const int* src_data,
  int rows,
  int cols)
{
  int  i = blockIdx.x * blockDim.x + threadIdx.x;
  int  j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < rows && j < cols) {
    int  ip1 = (i != rows - 1) ? (i + 1) : 0;
    int  im1 = (i != 0) ? (i - 1) : (rows - 1);
    int  jp1 = (j != cols - 1) ? (j + 1) : 0;
    int  jm1 = (j != 0) ? (j - 1) : (cols - 1);
    int  count;

    count = src_data[im1 * cols + j] +
            src_data[im1 * cols + jp1] +
            src_data[i * cols + jp1] +
            src_data[ip1 * cols + jp1] +
            src_data[ip1 * cols + j] +
            src_data[ip1 * cols + jm1] +
            src_data[i * cols + jm1] +
            src_data[im1 * cols + jm1];

    if (src_data[i * cols + j] == ALIVE) {
      if (count < 2) {
	dst_data[i * cols + j] = DEAD;
      } else if (count == 2 || count == 3) {
	dst_data[i * cols + j] = ALIVE;
      } else {
	dst_data[i * cols + j] = DEAD;
      }
    } else {
      if (count == 3) {
	dst_data[i * cols + j] = ALIVE;
      } else {
	dst_data[i * cols + j] = DEAD;
      }
    }
  }
}

gol_data*
gol_data_evolve(
  gol_data* dst,
  const gol_data* src)
{
  assert(gol_data_is_valid(dst));
  assert(gol_data_is_valid(src));
  assert(dst->rows == src->rows);
  assert(dst->cols == src->cols);

  int*  dst_data;
  int*  src_data;
  int   block_dim_x = 16;
  int   block_dim_y = 16;
  int   grid_dim_x = (int)ceil((float)dst->rows / block_dim_x);
  int   grid_dim_y = (int)ceil((float)dst->cols / block_dim_y);

  cudaMalloc((void**)&dst_data,sizeof(int) * dst->rows * dst->cols);
  cudaMalloc((void**)&src_data,sizeof(int) * src->rows * src->cols);

  cudaMemcpy(dst_data,dst->data,sizeof(int) * dst->rows * dst->cols,cudaMemcpyHostToDevice);
  cudaMemcpy(src_data,src->data,sizeof(int) * src->rows * src->cols,cudaMemcpyHostToDevice);

  _gol_data_evolve <<<dim3(grid_dim_x,grid_dim_y),dim3(block_dim_x,block_dim_y)>>> (dst_data,src_data,dst->rows,dst->cols);
  cudaThreadSynchronize();

  cudaMemcpy(dst->data,dst_data,sizeof(int) * dst->rows * dst->cols,cudaMemcpyDeviceToHost);

  cudaFree(dst_data);
  cudaFree(src_data);

  return dst;
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

  return gol->data[i * gol->cols + j];
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

  gol->data[i * gol->cols + j] = cell_state;
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
