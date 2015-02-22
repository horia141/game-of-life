#include <stdio.h>
#include <assert.h>

#include "color.h"

int
main(
  int argc,
  char** argv)
{
  {
    color  c;

    c = color_make_rgb(0,0.5,0.77);

    assert(c.r == 0.0f);
    assert(c.g == 0.5f);
    assert(c.b == 0.77f);
    assert(c.a == 1.0f);

    color_free(&c);

    assert(c.r == -1.0f);
    assert(c.g == -1.0f);
    assert(c.b == -1.0f);
    assert(c.a == -1.0f);
  }

  {
    color  c;

    c = color_make_rgba(0,0.23,0.71,0.44);

    assert(c.r == 0.0f);
    assert(c.g == 0.23f);
    assert(c.b == 0.71f);
    assert(c.a == 0.44f);

    color_free(&c);
  }

  {
    color  c1;
    color  c2;
    
    c1 = color_make_rgba(0,1,0,1);
    c2 = color_make_copy(&c1);

    assert(c1.r == 0);
    assert(c1.g == 1);
    assert(c1.b == 0);
    assert(c1.a == 1);

    assert(c2.r == c1.r);
    assert(c2.g == c1.g);
    assert(c2.b == c1.b);
    assert(c2.a == c1.a);

    color_free(&c1);
    color_free(&c2);
  }
    

  {
    color  c;

    c = color_make_rgba(0,0.11,0.73,0.88);

    assert(color_is_valid(&c) == true);

    c.r = 1.01;
    assert(color_is_valid(&c) == false);

    c.r = 0.44;
    assert(color_is_valid(&c) == true);

    c.r = -2.03;
    assert(color_is_valid(&c) == false);

    c.r = 0.22;
    assert(color_is_valid(&c) == true);

    c.g = 1.01;
    assert(color_is_valid(&c) == false);

    c.g = 0.44;
    assert(color_is_valid(&c) == true);

    c.g = -2.03;
    assert(color_is_valid(&c) == false);

    c.g = 0.22;
    assert(color_is_valid(&c) == true);

    c.b = 1.01;
    assert(color_is_valid(&c) == false);

    c.b = 0.44;
    assert(color_is_valid(&c) == true);

    c.b = -2.03;
    assert(color_is_valid(&c) == false);

    c.b = 0.22;
    assert(color_is_valid(&c) == true);

    c.a = 1.01;
    assert(color_is_valid(&c) == false);

    c.a = 0.44;
    assert(color_is_valid(&c) == true);

    c.a = -2.03;
    assert(color_is_valid(&c) == false);

    c.a = 0.22;
    assert(color_is_valid(&c) == true);

    color_free(&c);
  }

  {
    color  c1;
    color  c2;
    color  c3;
    color  c4;

    c1 = color_make_rgba(1,0,0,1);
    c2 = color_make_rgba(1,0,0,1);
    c3 = color_make_rgba(1,1,0,0);
    c4 = color_make_copy(&c3);

    assert(color_equal(&c1,&c2));
    assert(!color_equal(&c1,&c3));
    assert(color_equal(&c3,&c4));

    color_free(&c1);
    color_free(&c2);
    color_free(&c3);
    color_free(&c4);
  }

  {
    color  c1;
    color  c2;

    c1 = color_make_rgb(1,0,1);
    c2 = color_make_rgba(0.1,0.2,0.3,0.4);

    color_overwrite(&c1,&c2);

    assert(c1.r == 0.1f);
    assert(c1.g == 0.2f);
    assert(c1.b == 0.3f);
    assert(c1.a == 0.4f);

    assert(c1.r == c2.r);
    assert(c1.g == c2.g);
    assert(c1.b == c2.b);
    assert(c1.a == c2.a);

    color_free(&c1);
    color_free(&c2);
  }

  {
    color c1;
    color c2;

    c1 = color_make_rgba(1,0,1,0);
    c2 = color_make_rgba(0,1,0,1);

    assert(color_distance(&c1,&c2) == 4);

    color_overwrite(&c1,&c2);

    assert(color_distance(&c1,&c2) == 0);

    color_free(&c1);
    color_free(&c2);
  }

  return 0;
}
