#ifndef _POINT_H
#define _POINT_H

#include "utils.h"

typedef struct _point  point;

struct _point
{
  float  x;
  float  y;
  float  z;
};

point   point_make_xy(float x, float y);
point   point_make_xyz(float x, float y, float z);
void    point_free(point* p);

bool    point_is_valid(const point* p);

bool    point_equal(const point* p1, const point* p2);

point*  point_overwrite(point* dst, const point* src);
float   point_distance(const point* a, const point* b);

#endif
