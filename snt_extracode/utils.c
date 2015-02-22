#include <stdlib.h>

#include "utils.h"

int
max_n(
  int a,
  int b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

int
min_n(
  int a,
  int b)
{
  if (a < b) {
    return a;
  } else {
    return b;
  }
}

float
max_f(
  float a,
  float b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

float
min_f(
  float a,
  float b)
{
  if (a < b) {
    return a;
  } else {
    return b;
  }
}

int
clamp_i(
  int x,
  int min,
  int max)
{
  assert (min < max);

  if (x < min) {
    return min;
  } else if (x > max) {
    return max;
  } else {
    return x;
  }
}

float
clamp_f(
  float x,
  float min,
  float max)
{
  assert (min < max);

  if (x < min) {
    return min;
  } else if (x > max) {
    return max;
  } else {
    return x;
  }
}

int
unirandom_i(
  int beg,
  int end)
{
  assert(beg < end);

  return beg + (random() % (end - beg));
}


float
unirandom_f(
  float beg,
  float end)
{
  assert(beg < end);

  return beg + (end - beg) * (random() / (float)RAND_MAX);
}

bool
unirandom_b()
{
  int  tmp;

  tmp = random() % 2;

  if (tmp == 0) {
    return true;
  } else {
    return false;
  }
}

bool
eps_eql(
  float a,
  float b)
{
  float  max;
  float  min;

  if (a > b) {
    max = a;
    min = b;
  } else {
    max = b;
    min = a;
  }

  if ((max - min) < EPS) {
    return true;
  } else {
    return false;
  }
}
