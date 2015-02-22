#include <stdio.h>
#include <assert.h>

#include "driver.h"

static int
test1_frame_cb()
{
  return 0;
}

static struct {
  int      curr_iteration;
  driver*  drv;
  tquad*   tq3;
  tquad*   tq4;
  tquad*   tq5;
  float    reds[];
} _test2_data = {
  .curr_iteration = 0,
  .drv = NULL,
  .tq3 = NULL,
  .tq4 = NULL,
  .tq5 = NULL,
  .reds = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
};

static int
test2_frame_cb()
{
  if (_test2_data.curr_iteration < 10) {
    driver_set_title(_test2_data.drv,"Iteration %d",_test2_data.curr_iteration);
    driver_move_by(_test2_data.drv,10,10);
    driver_resize_by(_test2_data.drv,-10,-10);

    tquad_move_by(_test2_data.tq3,0.1,0.1);
    tquad_resize_by(_test2_data.tq3,-0.01,-0.01);

    tquad_texture_set(_test2_data.tq4,
		      _test2_data.curr_iteration / 5,
		      _test2_data.curr_iteration % 5,
		      &(color){_test2_data.reds[_test2_data.curr_iteration],0,0,1});

    if (_test2_data.curr_iteration % 2) {
      tquad_visiflip(_test2_data.tq5);
    }

    if (_test2_data.curr_iteration == 4) {
      tquad*  tqp;

      tqp = driver_tquad_make_color(_test2_data.drv,&(rectangle){0,0.13,0.14,0.22},4,4,&(color){0,1,0,1});
      assert(tquad_is_valid(tqp));
    }

    _test2_data.curr_iteration += 1;

    return 1;
  } else {
    return 0;
  }
}

int
main(
  int argc,
  char** argv)
{
  {
    driver*  drv;

    drv = driver_make(test1_frame_cb,"Test 1",&(rectangle){0,0,200,200},10);

    assert(driver_is_valid(drv));

    driver_start(drv);
    driver_free(drv);
  }

  {
    driver*  drv;
    image*   texture;
    tquad*   tq1;
    tquad*   tq2;
    tquad*   tq3;
    tquad*   tq4;
    tquad*   tq5;
    int      i;

    drv = driver_make(test2_frame_cb,"Test 2",&(rectangle){100,100,600,600},400);

    assert(driver_is_valid(drv));

    texture = image_from_ppm_t("test_driver.ppm");
    tq1 = driver_tquad_make_color(drv,&(rectangle){0.22,0.44,0.11,0.11},4,4,&(color){1,0,0,1});
    tq2 = driver_tquad_make_image(drv,&(rectangle){0.54,0.82,0.11,0.10},texture);
    tq3 = driver_tquad_make_image(drv,&(rectangle){0,0,0.2,0.2},texture);
    tq4 = driver_tquad_make_color(drv,&(rectangle){0.5,0.5,0.22,0.22},2,5,&(color){1,1,1,1});
    tq5 = driver_tquad_make_copy(drv,tq3);

    tquad_hide(tq5);

    _test2_data.drv = drv;
    _test2_data.tq3 = tq3;
    _test2_data.tq4 = tq4;
    _test2_data.tq5 = tq5;

    assert(tquad_is_valid(tq1));
    assert(tquad_is_valid(tq2));
    assert(tquad_is_valid(tq3));
    assert(tquad_is_valid(tq4));
    assert(tquad_is_valid(tq5));

    driver_start(drv);

    for (i = 0; i < 10; i++) {
      assert((tquad_texture_get(tq4,i / 5,i % 5))->r == _test2_data.reds[i]);
    }

    image_free(texture);
    driver_free(drv);
  }

  return 0;
}
