#include <stdio.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "ext/snt/utils.h"
#include "ext/snt/color.h"
#include "ext/snt/rectangle.h"
#include "ext/snt/image.h"
#include "ext/snt/driver.h"

static struct {
  int     max_iteration;
  int     ms_per_frame;
  image*  initial_state;
  bool    exit_on_stop;
} config;

static struct {
  int      curr_iteration;
  int      show_which;
  driver*  drv;
  tquad*   gol_data0;
  tquad*   gol_data1;
} state;

static int
gol_count_live_neighbours(
  tquad* tq,
  int i,
  int j)
{
  assert(tquad_is_valid(tq));
  assert(i >= 1 && i < image_get_rows(tquad_get_texture(tq)) - 1);
  assert(j >= 1 && j < image_get_rows(tquad_get_texture(tq)) - 1);

  int    count = 0;
  color  alive = color_make_rgb(0,0,0);

  if (color_equal(tquad_texture_get(tq,i-1,j),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i-1,j+1),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i,j+1),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i+1,j+1),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i+1,j),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i+1,j-1),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i,j-1),&alive)) {
    count += 1;
  }

  if (color_equal(tquad_texture_get(tq,i-1,j-1),&alive)) {
    count += 1;
  }

  return count;
}

static int
gol_frame_cb()
{
  if (state.curr_iteration < config.max_iteration) {
    tquad*  prev;
    tquad*  curr;
    color   dead = color_make_rgb(1,1,1);
    color   alive = color_make_rgb(0,0,0);
    int     rows;
    int     cols;
    int     count;
    int     i;
    int     j;

    if (state.show_which == 0) {
      prev = state.gol_data0;
      curr = state.gol_data1;
    } else {
      prev = state.gol_data1;
      curr = state.gol_data0;
    }

    rows = image_get_rows(tquad_get_texture(prev));
    cols = image_get_cols(tquad_get_texture(prev));

    for (i = 1; i < rows-1; i++) {
      for (j = 1; j < cols-1; j++) {
	count = gol_count_live_neighbours(prev,i,j);

	if (color_equal(tquad_texture_get(prev,i,j),&alive)) {
	  if (count < 2) {
	    tquad_texture_set(curr,i,j,&dead);
	  } else if (count == 2 || count == 3) {
	    tquad_texture_set(curr,i,j,&alive);
	  } else {
	    tquad_texture_set(curr,i,j,&dead);
	  }
	} else {
	  if (count == 3) {
	    tquad_texture_set(curr,i,j,&alive);
	  } else {
	    tquad_texture_set(curr,i,j,&dead);
	  }
	}
      }
    }

    tquad_hide(prev);
    tquad_show(curr);

    if (state.show_which == 0) {
      state.show_which = 1;
    } else {
      state.show_which = 0;
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
  config.max_iteration = 1000;
  config.ms_per_frame = 250;
  config.initial_state = image_from_ppm_t("gol4.ppm");
  config.exit_on_stop = false;

  state.curr_iteration = 0;
  state.show_which = 0;
  state.drv = driver_make(gol_frame_cb,config.ms_per_frame);
  state.gol_data0 = driver_tquad_make_image(state.drv,&(rectangle){0,0,1,1},config.initial_state);
  state.gol_data1 = driver_tquad_make_copy(state.drv,state.gol_data0);

  tquad_hide(state.gol_data1);

  driver_start(state.drv);

  driver_free(state.drv);
  image_free(config.initial_state);

  return 0;
}
