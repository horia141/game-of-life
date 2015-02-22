#include <stdlib.h>

#include "color.h"
#include "utils.h"

color
color_make_rgb(
  float r,
  float g,
  float b)
{
  assert(r >= 0 && r <= 1);
  assert(g >= 0 && g <= 1);
  assert(b >= 0 && b <= 1);

  color  new_c;

  new_c.r = r;
  new_c.g = g;
  new_c.b = b;
  new_c.a = 1;

  return new_c;
}

color
color_make_rgba(
  float r,
  float g,
  float b,
  float a)
{
  assert(r >= 0 && r <= 1);
  assert(g >= 0 && g <= 1);
  assert(b >= 0 && b <= 1);
  assert(a >= 0 && a <= 1);

  color  new_c;

  new_c.r = r;
  new_c.g = g;
  new_c.b = b;
  new_c.a = a;

  return new_c;
}

color
color_make_copy(
  const color* c)
{
  assert(color_is_valid(c));

  color  new_c;

  new_c.r = c->r;
  new_c.g = c->g;
  new_c.b = c->b;
  new_c.a = c->a;

  return new_c;
}

void
color_free(
  color* c)
{
  assert(color_is_valid(c));

  c->r = -1;
  c->g = -1;
  c->b = -1;
  c->a = -1;
}


bool
color_is_valid(
  const color* c)
{
  if (c == NULL) {
    return true;
  }

  if (c->r < 0 || c->r > 1) {
    return false;
  }

  if (c->g < 0 || c->g > 1) {
    return false;
  }

  if (c->b < 0 || c->b > 1) {
    return false;
  }

  if (c->a < 0 || c->a > 1) {
    return false;
  }

  return true;
}


bool
color_equal(
  const color* c1,
  const color* c2)
{
  assert(color_is_valid(c1));
  assert(color_is_valid(c2));

  return c1->r == c2->r && c1->g == c2->g && c1->b == c2->b && c1->a == c2->a;
}


color*
color_overwrite(
  color* dst,
  const color* src)
{
  assert(color_is_valid(dst));
  assert(color_is_valid(src));

  dst->r = src->r;
  dst->g = src->g;
  dst->b = src->b;
  dst->a = src->a;

  return dst;
}

float
color_distance(
  const color* a,
  const color* b)
{
  assert(color_is_valid(a));
  assert(color_is_valid(b));

  float  dst_r;
  float  dst_g;
  float  dst_b;
  float  dst_a;

  dst_r = a->r - b->r;
  dst_g = a->g - b->g;
  dst_b = a->b - b->b;
  dst_a = a->a - b->a;

  return dst_r*dst_r + dst_g*dst_g + dst_b*dst_b + dst_a*dst_a;
}
