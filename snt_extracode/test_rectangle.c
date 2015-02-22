#include <stdio.h>
#include <assert.h>

#include "rectangle.h"

int
main(
  int argc,
  char** argv)
{
  {
    rectangle  r;

    r = rectangle_make_xywh(1,2,3,4);

    assert(r.x == 1.0f);
    assert(r.y == 2.0f);
    assert(r.w == 3.0f);
    assert(r.h == 4.0f);

    rectangle_free(&r);

    assert(r.x == 0.0f);
    assert(r.y == 0.0f);
    assert(r.w == -1.0f);
    assert(r.h == -1.0f);
  }

  {
    rectangle  r;

    r = rectangle_make_tlbr(1,2,3,4);

    assert(r.x == 1.0f);
    assert(r.y == 2.0f);
    assert(r.w == 2.0f);
    assert(r.h == 2.0f);

    rectangle_free(&r);
  }

  {
    rectangle  r1;
    rectangle  r2;

    r1 = rectangle_make_xywh(1,1,2,2);
    r2 = rectangle_make_copy(&r1);

    assert(r1.x == 1);
    assert(r1.y == 1);
    assert(r1.w == 2);
    assert(r1.h == 2);

    assert(r2.x == r1.x);
    assert(r2.y == r1.y);
    assert(r2.w == r1.w);
    assert(r2.h == r1.h);

    rectangle_free(&r1);
    rectangle_free(&r2);
  }

  {
    rectangle  r;

    r = rectangle_make_xywh(1,3,2,0.2);

    assert(rectangle_is_valid(&r) == true);

    r.w = -1.3;
    assert(rectangle_is_valid(&r) == false);

    r.w = 0.55;
    assert(rectangle_is_valid(&r) == true);

    r.h = -1.3;
    assert(rectangle_is_valid(&r) == false);

    r.h = 0.55;
    assert(rectangle_is_valid(&r) == true);

    rectangle_free(&r);
  }

  {
    rectangle  r1;
    rectangle  r2;

    r1 = rectangle_make_xywh(4,4,3,2);
    r2 = rectangle_make_tlbr(3,4,7,5);

    rectangle_overwrite(&r1,&r2);

    assert(r1.x == 3);
    assert(r1.y == 4);
    assert(r1.w == 4);
    assert(r1.h == 1);

    assert(r1.x == r2.x);
    assert(r1.y == r2.y);
    assert(r1.w == r2.w);
    assert(r1.h == r2.h);

    rectangle_free(&r1);
    rectangle_free(&r2);
  }

  return 0;
}
