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
  "  --console-only               Skip the visualisation part. Useful for benchmarks.\n"
  "  --max-iters=[number]         Number of iterations to perform.\n"
  "  --iters-stride=ALL|[number]  Show every \"iters-stride\" iterations.\n"
  "  --ms-per-iter=[number]       Number of milliseconds to spend showing each generation.\n"
  "  --life-path=[path]           File path to initial state file.\n"
  "  --exit-on-stop               The simulator exist when the last iteration is drawn.\n"
  "  --help                       Show this help message.";

static struct {
  bool    console_only;
  int     max_iteration;
  int     iters_stride;
  int     ms_per_iter;
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

    if ((state.curr_iteration % config.iters_stride == 0) ||
	(state.curr_iteration == config.max_iteration - 1)) {
      fprintf(stderr,"Iteration #%d\n",state.curr_iteration + 1);

      for (i = 0; i < state.rows; i++) {
	for (j = 0; j < state.cols; j++) {
	  if (gol_data_get(curr,i,j) == ALIVE) {
	    tquad_texture_set(state.tq,i,j,&(color){0,0,0,1});
	  } else {
	    tquad_texture_set(state.tq,i,j,&(color){1,1,1,1});
	  }
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

static int
gol_console_run()
{
  while (state.curr_iteration < config.max_iteration) {
    gol_data*  prev;
    gol_data*  curr;

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

    if ((state.curr_iteration % config.iters_stride == 0) ||
	(state.curr_iteration == config.max_iteration - 1)) {
      fprintf(stderr,"Iteration #%d\n",state.curr_iteration + 1);
    }

    state.curr_iteration += 1;
  }

  return 0;
}

static rectangle*
gol_pos(void)
{
  static rectangle  r;

  if (state.rows > state.cols) {
    float  aspect_cols;

    aspect_cols = (float)state.cols / (float)state.rows;

    r.x = (1 - aspect_cols) / 2;
    r.y = 0;
    r.w = aspect_cols;
    r.h = 1;
  } else {
    float  aspect_rows;
    
    aspect_rows = (float)state.rows / (float)state.cols;

    r.x = 0;
    r.y = (1 - aspect_rows) / 2;
    r.w = 1;
    r.h = aspect_rows;
  }

  return &r;
}


int
main(
  int argc,
  char** argv)
{
  static struct option long_options[] = {
    {"console-only", no_argument,       0, 1},
    {"max-iters",    required_argument, 0, 2},
    {"iters-stride", required_argument, 0, 3},
    {"ms-per-iter",  required_argument, 0, 4},
    {"life-path",    required_argument, 0, 5},
    {"exit-on-stop", no_argument,       0, 6},
    {"help",         no_argument,       0, 7},
    {0,0,0,0}
  };

  int    option_idx = 0;
  int    getopt_res = 0;
  int    scan_res;

  config.console_only = false;
  config.max_iteration = 10;
  config.iters_stride = 1;
  config.ms_per_iter = 100;
  config.life_path = strdup("data/life01.l");
  config.exit_on_stop = false;

  while ((getopt_res = getopt_long(argc,argv,"",long_options,&option_idx)) != -1) {
    switch (getopt_res) {
    case 1:
      config.console_only = true;
      break;
    case 2:
      scan_res = sscanf(optarg,"%d",&config.max_iteration);
      if (scan_res != 1) {
	fprintf(stderr,"Invalid argument for option \"max-iters\": %s\n",optarg);
	exit(EXIT_FAILURE);
      }

      break;
    case 3:
      if (strcasecmp(optarg,"ALL") == 0) {
	config.iters_stride = -1;
      } else {
	scan_res = sscanf(optarg,"%d",&config.iters_stride);

	if (scan_res != 1) {
	  fprintf(stderr,"Invalid argument for option \"iters-stride\": %s\n",optarg);
	  exit(EXIT_FAILURE);
	}

	if (config.iters_stride < 1) {
	  fprintf(stderr,"Invalid argument for option \"iters-stride\": %s\n",optarg);
	  fprintf(stderr,"Must be strictly greater than 0.\n");
	  exit(EXIT_FAILURE);
	}
      }

      break;
    case 4:
      scan_res = sscanf(optarg,"%d",&config.ms_per_iter);
      if (scan_res != 1) {
	fprintf(stderr,"Invalid argument for option \"ms-per-iter\": %s\n",optarg);
	exit(EXIT_FAILURE);
      }	

      break;
    case 5:
      config.life_path = strdup(optarg);
      break;
    case 6:
      config.exit_on_stop = true;
      break;
    case 7:
      printf("%s\n",help_msg_header);
      printf("%s\n",help_msg_options);
      exit(EXIT_SUCCESS);
    default:
      exit(EXIT_FAILURE);
    }
  }

  /* A small integrity test. */

  if (config.console_only && config.exit_on_stop) {
    fprintf(stderr,"Invalid usage of option \"exit-on-stop\" with option \"console-only\"!\n");
    exit(EXIT_FAILURE);
  }

  if (config.iters_stride == -1) {
    config.iters_stride = config.max_iteration;
  }

  state.curr_iteration = 0;
  state.show_which = 0;
  state.gol_data0 = gol_data_from_l(config.life_path);
  state.gol_data1 = gol_data_from_l(config.life_path);
  state.rows = gol_data_get_rows(state.gol_data0);
  state.cols = gol_data_get_cols(state.gol_data1);

  if (!config.console_only) {
    state.drv = driver_make(gol_frame_cb,config.ms_per_iter);
    state.tq = driver_tquad_make_color(state.drv,gol_pos(),state.rows,state.cols,&(color){1,1,1,1});

    driver_start(state.drv);

    driver_free(state.drv);
  } else {
    state.drv = NULL;
    state.tq = NULL;

    gol_console_run();
  }

  gol_data_free(state.gol_data0);
  gol_data_free(state.gol_data1);
  free(config.life_path);

  return 0;
}
