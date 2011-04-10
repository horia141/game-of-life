#include <stdio.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "ext/snt/utils.h"
#include "ext/snt/color.h"
#include "ext/snt/rectangle.h"
#include "ext/snt/image.h"
#include "ext/snt/driver.h"

#define ALIVE 1
#define DEAD 0

typedef struct _gol_data  gol_data;

static struct {
  int     max_iteration;
  int     ms_per_frame;
  char*   life_path;
  bool    exit_on_stop;
} config;

static struct {
  int        curr_iteration;
  int        show_which;
  gol_data*  gol_data0;
  gol_data*  gol_data1;
  int        rows;
  int        cols;
  driver*    drv;
  tquad*     tq;
} state;

struct _gol_data
{
  int  rows;
  int  cols;
  int  data[];
};

static gol_data*  gol_data_from_l(const char* path);
static void       gol_data_free(gol_data* gol);

static bool       gol_data_is_valid(const gol_data* gol);

static gol_data*  gol_data_evolve(gol_data* dst, const gol_data* src);
static int        gol_data_neigh_count(const gol_data* gol, int i, int j);
static int        gol_data_get_rows(const gol_data* gol);
static int        gol_data_get_cols(const gol_data* gol);
static int        gol_data_get(const gol_data* gol, int i, int j);
static void       gol_data_set(gol_data* gol, int i, int j, int cell_state);

static gol_data* 
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

  new_gol = malloc(sizeof(gol_data) + sizeof(int) * rows * cols);

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

static void
gol_data_free(
  gol_data* gol)
{
  assert(gol_data_is_valid(gol));

  gol->rows = -1;
  gol->cols = -1;

  free(gol);
}

static bool
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


static gol_data*
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

static int
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

  return (gol_data_get(gol,im1,j) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,im1,jp1) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,i,jp1) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,ip1,jp1) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,ip1,j) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,ip1,jm1) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,i,jm1) == ALIVE ? 1 : 0) +
         (gol_data_get(gol,im1,jm1) == ALIVE ? 1 : 0);
}

static int
gol_data_get_rows(
  const gol_data* gol)
{
  assert(gol_data_is_valid(gol));
  
  return gol->rows;
}

static int
gol_data_get_cols(
  const gol_data* gol)
{
  assert(gol_data_is_valid(gol));
  
  return gol->cols;
}

static int
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

static void
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

static int
gol_frame_cb()
{
  if (state.curr_iteration < config.max_iteration) {
    gol_data*  prev;
    gol_data*  curr;
    int        i;
    int        j;

    fprintf(stderr,"Iteration #%d\n",state.curr_iteration + 1);
    
    if (state.show_which == 0) {
      prev = state.gol_data0;
      curr = state.gol_data1;
      state.show_which = 1;
    } else {
      prev = state.gol_data1;
      curr = state.gol_data0;
      state.show_which = 0;
    }

    gol_data_evolve(curr,prev);

    for (i = 0; i < state.rows; i++) {
      for (j = 0; j < state.cols; j++) {
	if (gol_data_get(curr,i,j) == ALIVE) {
	  tquad_texture_set(state.tq,i,j,&(color){0,0,0,1});
	} else {
	  tquad_texture_set(state.tq,i,j,&(color){1,1,1,1});
	}
      }
    }

    state.curr_iteration += 1;

    return 1;
  } else {
    if (config.exit_on_stop) {
      return 0;
    } else {
      return 1;
    }
  }
}

int
main(
  int argc,
  char** argv)
{
  config.max_iteration = 10000;
  config.ms_per_frame = 10;
  config.life_path = "data/life12.l";
  config.exit_on_stop = true;

  state.curr_iteration = 0;
  state.show_which = 0;
  state.gol_data0 = gol_data_from_l(config.life_path);
  state.gol_data1 = gol_data_from_l(config.life_path);
  state.rows = gol_data_get_rows(state.gol_data0);
  state.cols = gol_data_get_cols(state.gol_data1);
  state.drv = driver_make(gol_frame_cb,config.ms_per_frame);
  state.tq = driver_tquad_make_color(state.drv,&(rectangle){0,0,1,1},state.rows,state.cols,&(color){1,1,1,1});

  driver_start(state.drv);

  driver_free(state.drv);
  gol_data_free(state.gol_data0);
  gol_data_free(state.gol_data1);

  return 0;
}
