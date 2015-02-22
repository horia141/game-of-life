#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "image.h"

struct _image
{
  int     rows;
  int     cols;
  color   data[];
};

image*
image_make_blank(
  int rows,
  int cols,
  const color* c)
{
  assert(rows > 0);
  assert(cols > 0);
  assert(color_is_valid(c));

  image*  new_img;
  int     i;
  int     j;

  new_img = (image*)malloc(sizeof(image) + sizeof(color) * rows * cols);

  new_img->rows = rows;
  new_img->cols = cols;

  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      new_img->data[i * cols + j].r = c->r;
      new_img->data[i * cols + j].g = c->g;
      new_img->data[i * cols + j].b = c->b;
      new_img->data[i * cols + j].a = c->a;
    }
  }
		   
  return new_img;
}

image*
image_make_copy(
  const image* src)
{
  assert(image_is_valid(src));

  image*  new_img;

  new_img = image_make_blank(src->rows,src->cols,&(color){0,0,0,1});

  return image_overwrite(new_img,src);
}

void
image_free(
  image* img)
{
  assert(image_is_valid(img));

  #ifndef _NDEBUG
  int  i;
  int  j;

  for (i = 0; i < img->rows; i++) {
    for (j = 0; j < img->cols; j++) {
      color_free(&img->data[i * img->cols + j]);
    }
  }
  #endif

  img->rows = -1;
  img->cols = -1;

  free(img);
}


bool
image_is_valid(
  const image* img)
{
  int  i;
  int  j;

  if (img == NULL) {
    return false;
  }

  if (img->rows < 1) {
    return false;
  }

  if (img->cols < 1) {
    return false;
  }

  for (i = 0; i < img->rows; i++) {
    for (j = 0; j < img->cols; j++) {
      if (!color_is_valid(&img->data[i * img->cols + j]));
    }
  }

  return true;
}


bool
image_equal(
  const image* img1,
  const image* img2,
  float epsilon)
{
  assert(image_is_valid(img1));
  assert(image_is_valid(img2));
  assert(epsilon > 0);
  assert(img1->rows == img2->rows);
  assert(img1->cols == img2->cols);

  int  i;
  int  j;

  for (i = 0; i < img1->rows; i++) {
    for (j = 0; j < img1->cols; j++) {
      if (color_distance(&img1->data[i * img1->cols + j],&img2->data[i * img1->cols +j]) > epsilon) {
	return false;
      }
    }
  }

  return true;  
}


image*
image_overwrite(
  image* dst,
  const image* src)
{
  assert(image_is_valid(dst));
  assert(image_is_valid(src));
  assert(dst->rows == src->rows);
  assert(dst->cols == src->cols);

  int  i;
  int  j;

  for (i = 0; i < dst->rows; i++) {
    for (j = 0; j < dst->cols; j++) {
      color_overwrite(&dst->data[i * dst->cols + j],&src->data[i * dst->cols + j]);
    }
  }

  return dst;
}

float*
image_make_texture(
  const image* img)
{
  assert(image_is_valid(img));

  float*  new_texture;

  new_texture = (float*)malloc(sizeof(float) * img->rows * img->cols * 4);

  return image_make_texture_a(img,new_texture);
}

float*
image_make_texture_a(
  const image* img,
  float* texture)
{
  assert(image_is_valid(img));
  assert(texture != NULL);

  const color*  tmp;
  int           i;
  int           j;

  for (i = 0; i < img->rows; i++) {
    for (j = 0; j < img->cols; j++) {
      tmp = image_get(img,i,j);
      texture[(img->rows - i - 1) * img->cols * 4 + j * 4 + 0] = tmp->r;
      texture[(img->rows - i - 1) * img->cols * 4 + j * 4 + 1] = tmp->g;
      texture[(img->rows - i - 1) * img->cols * 4 + j * 4 + 2] = tmp->b;
      texture[(img->rows - i - 1) * img->cols * 4 + j * 4 + 3] = tmp->a;
    }
  }

  return texture;
}


int
image_get_rows(
  const image* img)
{
  assert(image_is_valid(img));

  return img->rows;
}

int
image_get_cols(
  const image* img)
{
  assert(image_is_valid(img));

  return img->cols;
}

const color*
image_get_pixels(
  const image* img)
{
  assert(image_is_valid(img));

  return img->data;
}

color*
image_get_pixels_a(
  image* img)
{
  assert(image_is_valid(img));

  return img->data;
}

const color*
image_get(
  const image* img,
  int row,
  int col)
{
  assert(image_is_valid(img));
  assert(row >= 0 && row < img->rows);
  assert(col >= 0 && col < img->cols);

  return &img->data[row * img->cols + col];
}

void
image_set(
  image* img,
  int row,
  int col,
  const color* c)
{
  assert(image_is_valid(img));
  assert(row >= 0 && row < img->rows);
  assert(col >= 0 && col < img->cols);
  assert(color_is_valid(c));

  img->data[row * img->cols + col].r = c->r;
  img->data[row * img->cols + col].g = c->g;
  img->data[row * img->cols + col].b = c->b;
  img->data[row * img->cols + col].a = c->a;
}


image*
image_from_ppm_t(
  const char* ppm_path)
{
  assert(ppm_path != NULL);
  assert(ppm_path[0] != '\0');

  #define TEXT_LINE_MAX_SIZE 1024

  image*  new_img;
  FILE*   ppm_file;
  char*   text_line;
  int     width;
  int     height;
  int     maximum;
  int     color_r;
  int     color_g;
  int     color_b;
  int     i;
  int     j;

  ppm_file = fopen(ppm_path,"rt");
  text_line = (char*)malloc(sizeof(char) * TEXT_LINE_MAX_SIZE);

  /* Jump first two ines */

  memset(text_line,0,TEXT_LINE_MAX_SIZE);
  fgets(text_line,TEXT_LINE_MAX_SIZE,ppm_file);
  memset(text_line,0,TEXT_LINE_MAX_SIZE);
  fgets(text_line,TEXT_LINE_MAX_SIZE,ppm_file);
  fscanf(ppm_file,"%d %d",&width,&height);
  fscanf(ppm_file,"%d",&maximum);

  new_img = image_make_blank(height,width,&(color){0,0,0,1});

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      fscanf(ppm_file,"%d %d %d",&color_r,&color_g,&color_b);
      image_set(new_img,i,j,&(color){color_r/(float)maximum,color_g/(float)maximum,color_b/(float)maximum,1});
    }
  }

  free(text_line);
  fclose(ppm_file);

  return new_img;
}

void
image_print_t(
  const image* img)
{
  assert(image_is_valid(img));

  int  i;
  int  j;

  for (i = 0; i < img->rows; i++) {
    for (j = 0; j < img->cols; j++) {
      printf("[%03d:%03d:%03d:%03d] ",
	     (int)(img->data[i * img->cols + j].r * 255),
	     (int)(img->data[i * img->cols + j].g * 255),
	     (int)(img->data[i * img->cols + j].b * 255),
	     (int)(img->data[i * img->cols + j].a * 255));
    }
    printf("\n");
  }
}
