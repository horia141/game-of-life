#ifndef _DRIVER_H
#define _DRIVER_H

#include "utils.h"
#include "color.h"
#include "rectangle.h"
#include "image.h"

typedef int           (*driver_frame_cb)(void);
typedef struct _driver  driver;
typedef struct _tquad   tquad;

driver*           driver_make(driver_frame_cb frame_cb, int ms_per_frame);
void              driver_free(driver* drv);
	      
bool              driver_is_valid(const driver* drv);
	      
driver*           driver_start(driver* drv);
tquad*            driver_tquad_make_color(driver* drv, const rectangle* geometry, int rows, int cols, const color* color);
tquad*            driver_tquad_make_image(driver* drv, const rectangle* geometry, const image* texture);
tquad*            driver_tquad_make_copy(driver* drv, const tquad* src);
void              driver_tquad_free(driver* drv, tquad* tq);
	      
bool              tquad_is_valid(const tquad* tq);
	      
tquad*            tquad_move_to(tquad* tq, float x, float y);
tquad*            tquad_move_by(tquad* tq, float dx, float dy);
tquad*            tquad_resize_to(tquad* tq, float w, float h);
tquad*            tquad_resize_by(tquad* tq, float dw, float dh);
tquad*            tquad_show(tquad* tq);
tquad*            tquad_hide(tquad* tq);
tquad*            tquad_visiflip(tquad* tq);
const rectangle*  tquad_get_geometry(const tquad* tq);
const image*      tquad_get_texture(const tquad* tq);
const color*      tquad_texture_get(const tquad* tq, int row, int col);
void              tquad_texture_set(tquad* tq, int row, int col, const color* c);

#endif
