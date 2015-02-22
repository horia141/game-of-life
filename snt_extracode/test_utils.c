#include <stdio.h>
#include <assert.h>

#include "utils.h"

int
main(
  int argc,
  char** argv)
{
  assert (max_n(10,23) == 23);
  assert (max_n(32,13) == 32);
  assert (max_n(10,12) != 10);

  assert (min_n(10,23) == 10);
  assert (min_n(32,13) == 13);
  assert (min_n(10,12) != 12);

  assert (max_f(1.0,2.3) == 2.3f);
  assert (max_f(1.0,1.0001) == 1.0001f);
  assert (max_f(3.2,1.3) == 3.2f);
  assert (max_f(1.0,1.2) != 1.0f);

  assert (min_f(1.0,2.3) == 1.0f);
  assert (min_f(1.0,1.0001) == 1.0f);
  assert (min_f(3.2,1.3) == 1.3f);
  assert (min_f(1.0,1.2) != 1.2f);

  assert (clamp_i(3,2,5) == 3);
  assert (clamp_i(1,2,5) == 2);
  assert (clamp_i(2,2,5) == 2);
  assert (clamp_i(6,2,5) == 5);
  assert (clamp_i(5,2,5) == 5);

  assert (clamp_f(3,2,5) == 3.0f);
  assert (clamp_f(1,2,5) == 2.0f);
  assert (clamp_f(2,2,5) == 2.0f);
  assert (clamp_f(1.999,2,5) == 2.0f);
  assert (clamp_f(2.001,2,5) == 2.001f);
  assert (clamp_f(6,2,5) == 5.0f);
  assert (clamp_f(5,2,5) == 5.0f);
  assert (clamp_f(4.999,2,5) == 4.999f);
  assert (clamp_f(5.001,2,5) == 5.000f);

  assert (eps_eql(1,1) == true);
  assert (eps_eql(1,3) == false);
  assert (eps_eql(3,1) == false);
  assert ((1.0f == 1.000001f) == false);
  assert (eps_eql(1,1.000001) == true);
  assert (eps_eql(1,1.00001) == false);
  assert ((1.000001f == 1.0f) == false);
  assert (eps_eql(1.000001,1) == true);
  assert (eps_eql(1,1.00001) == false);

  return 0;
}
