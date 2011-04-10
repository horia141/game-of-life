#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "ext/snt/utils.h"
#include "ext/snt/color.h"
#include "ext/snt/rectangle.h"
#include "ext/snt/image.h"
#include "ext/snt/driver.h"

#include "gol_data.h"

#ifdef CPU
#ifndef NDEBUG
static char
help_msg_header[] =
  "Usage: gol-cpu-debug [OPTION]...\n"
  "Run a Game Of Life simulation on the cpu in debug mode.";
#else
static char
help_msg_header[] =
  "Usage: gol-cpu-debug [OPTION]...\n"
  "Run a Game Of Life simulation on the cpu.";
#endif
#endif

#ifdef GPU
#ifndef NDEBUG
static char
help_msg_header[] =
  "Usage: gol-cpu-debug [OPTION]...\n"
  "Run a Game Of Life simulation on the gpu in debug mode.";
#else
static char
help_msg_header[] =
  "Usage: gol-cpu-debug [OPTION]...\n"
  "Run a Game Of Life simulation on the gpu.";
#endif
#endif

static char
help_msg_options[] =
  "Options:\n"
  "  --max-iters=[number]     Number of iterations to perform.\n"
  "  --ms-per-frame=[number]  Number of milliseconds to spend showing each generation.\n"
  "  --life-path=[path]       File path to initial state file.\n"
  "  --exit-on-stop           The simulator exist when the last iteration is drawn.\n";

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
  static struct option long_options[] = {
    {"max-iters",    required_argument, 0, 1},
    {"ms-per-frame", required_argument, 0, 2},
    {"life-path",    required_argument, 0, 3},
    {"exit-on-stop", no_argument,       0, 4},
    {"help",         no_argument,       0, 5},
    {0,0,0,0}
  };

  int option_idx = 0;
  int getopt_res = 0;
  int scan_res;

  config.max_iteration = 10;
  config.ms_per_frame = 100;
  config.life_path = "data/life01.l";
  config.exit_on_stop = false;

  while ((getopt_res = getopt_long(argc,argv,"",long_options,&option_idx)) != -1) {
    switch (getopt_res) {
    case 1:
      scan_res = sscanf(optarg,"%d",&config.max_iteration);
      if (scan_res != 1) {
	fprintf(stderr,"Invalid argument for option \"max-iters\": %s\n",optarg);
	exit(EXIT_FAILURE);
      }

      break;
    case 2:
      scan_res = sscanf(optarg,"%d",&config.ms_per_frame);
      if (scan_res != 1) {
	fprintf(stderr,"Invalid argument for option \"ms-per-frame\": %s\n",optarg);
	exit(EXIT_FAILURE);
      }	

      break;
    case 3:
      config.life_path = strdup(optarg);
      break;
    case 4:
      config.exit_on_stop = true;
      break;
    case 5:
      printf("%s\n",help_msg_header);
      printf("%s\n",help_msg_options);
      exit(EXIT_SUCCESS);
    default:
      exit(EXIT_FAILURE);
    }
  }

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
  free(config.life_path);

  return 0;
}
