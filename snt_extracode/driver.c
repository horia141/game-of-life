#include <setjmp.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "driver.h"

static void  _glut_display_cb();
static void  _glut_frame_cb();
static void  _glut_start(driver* drv);

static void  _driver_init(driver* drv);
static void  _driver_display(driver* drv);
static int   _driver_frame(driver* drv);
static void  _driver_init_tquad(driver* drv, tquad* tquad);
static void  _driver_clean_tquad(driver* drv, tquad* tquad);

struct _driver
{
  driver_frame_cb  frame_cb;
  char*            title;
  rectangle        geometry;
  int              ms_per_frame;
  int              quad_cnt;
  tquad*           quads;
};

struct _tquad
{
  const driver*  drv;
  bool           show;
  bool           dirty;
  rectangle      geometry;
  image*         texture;
  unsigned int   tid;
  tquad*         next;
  tquad*         prev;
};

static void  _driver_display(driver* drv);
static int   _driver_frame(driver* drv);


driver*
driver_make(
  driver_frame_cb frame_cb,
  const char* title,
  const rectangle* geometry,
  int ms_per_frame)
{
  assert(title != NULL && title[0] != '\0');
  assert(rectangle_is_valid(geometry));
  assert(geometry->x >= 0 && geometry->y >= 0);
  assert(ms_per_frame > 0);

  driver*  new_drv;

  new_drv = malloc(sizeof(driver));

  new_drv->frame_cb = frame_cb;
  new_drv->title = strdup(title);
  new_drv->geometry = *geometry;
  new_drv->ms_per_frame = ms_per_frame;
  new_drv->quad_cnt = 0;
  new_drv->quads = malloc(sizeof(tquad));
  new_drv->quads->drv = new_drv;
  new_drv->quads->show = false;
  new_drv->quads->dirty = false;
  new_drv->quads->geometry = rectangle_make_xywh(0,0,1,1);
  new_drv->quads->texture = NULL;
  new_drv->quads->tid = 0;
  new_drv->quads->next = malloc(sizeof(tquad));
  new_drv->quads->prev = new_drv->quads->next;
  new_drv->quads->prev->drv = new_drv;
  new_drv->quads->prev->show = false;
  new_drv->quads->prev->dirty = false;
  new_drv->quads->prev->geometry = rectangle_make_xywh(0,0,1,1);
  new_drv->quads->prev->texture = NULL;
  new_drv->quads->prev->tid = 0;
  new_drv->quads->prev->next = new_drv->quads;
  new_drv->quads->prev->prev = new_drv->quads;

  return new_drv;
}

void
driver_free(
  driver* drv)
{
  assert(driver_is_valid(drv));
  
  while (drv->quad_cnt > 0) {
    driver_tquad_free(drv,drv->quads->next);
  }

  drv->quads->prev->drv = NULL;
  drv->quads->prev->show = false;
  drv->quads->prev->dirty = false;
  rectangle_free(&drv->quads->prev->geometry);
  drv->quads->prev->texture = NULL;
  drv->quads->prev->tid = 0;
  drv->quads->prev->next = NULL;
  drv->quads->prev->prev = NULL;

  free(drv->quads->next);

  drv->quads->drv = NULL;
  drv->quads->show = false;
  drv->quads->dirty = false;
  rectangle_free(&drv->quads->geometry);
  drv->quads->texture = NULL;
  drv->quads->tid = 0;
  drv->quads->next = NULL;
  drv->quads->prev = NULL;

  free(drv->quads);
  free(drv->title);
  rectangle_free(&drv->geometry);

  drv->frame_cb = NULL;
  drv->title = NULL;
  drv->ms_per_frame = -1;
  drv->quad_cnt = -1;
  drv->quads = NULL;

  free(drv);
}


bool
driver_is_valid(
  const driver* drv)
{
  const tquad*  iter;
  int           count;

  if (drv == NULL) {
    return false;
  }

  if (drv->title == NULL || drv->title[0] == '\0') {
    return false;
  }

  if (!rectangle_is_valid(&drv->geometry)) {
    return false;
  }

  if (drv->geometry.x < 0 || drv->geometry.y < 0) {
    return false;
  }

  if (drv->ms_per_frame < 1) {
    return false;
  }

  if (drv->quad_cnt < 0) {
    return false;
  }

  if (drv->quads == NULL) {
    return false;
  }

  if (drv->quads->drv != drv) {
    return false;
  }

  if (drv->quads->show == true) {
    return false;
  }

  if (drv->quads->dirty == true) {
    return false;
  }
  
  if (!rectangle_is_valid(&drv->quads->geometry)) {
    return false;
  }

  if (drv->quads->texture != NULL) {
    return false;
  }

  if (drv->quads->tid != 0) {
    return false;
  }

  if (drv->quads->next == NULL) {
    return false;
  }

  if (drv->quads->prev == NULL) {
    return false;
  }

  if (drv->quads->prev->drv != drv) {
    return false;
  }

  if (drv->quads->prev->show == true) {
    return false;
  }

  if (drv->quads->prev->dirty == true) {
    return false;
  }

  if (!rectangle_is_valid(&drv->quads->prev->geometry)) {
    return false;
  }

  if (drv->quads->prev->texture != NULL) {
    return false;
  }

  if (drv->quads->prev->tid != 0) {
    return false;
  }

  if (drv->quads->prev->next != drv->quads) {
    return false;
  }

  if (drv->quads->prev->prev == NULL) {
    return false;
  }

  iter = drv->quads->next;
  count = 0;

  while (iter != drv->quads->prev) {
    if (!tquad_is_valid(iter)) {
      return false;
    }

    iter = iter->next;
    count = count + 1;
  }

  if (drv->quad_cnt != count) {
    return false;
  }

  return true;
}


driver*
driver_start(
  driver* drv)
{
  assert(driver_is_valid(drv));

  _glut_start(drv);

  return drv;
}

static void
_driver_init_tquad(
  driver* drv,
  tquad* tq)
{
  assert(driver_is_valid(drv));
  assert(tquad_is_valid(tq));
  assert(tq->drv == drv);
  assert(drv->quads != tq);
  assert(drv->quads->prev != tq);
  assert(tq->dirty);
  assert(tq->tid == 0);

  glGenTextures(1,&tq->tid);
  glBindTexture(GL_TEXTURE_2D,tq->tid);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
  glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
}

static void
_driver_clean_tquad(
  driver* drv,
  tquad* tq)
{
  assert(driver_is_valid(drv));
  assert(tquad_is_valid(tq));
  assert(tq->drv == drv);
  assert(drv->quads != tq);
  assert(drv->quads->prev != tq);
  assert(tq->dirty);
  assert(tq->tid > 0);

  float*  work_texture;
  int     rows;
  int     cols;

  rows = image_get_rows(tq->texture);
  cols = image_get_cols(tq->texture);
  work_texture = image_make_texture(tq->texture);
  glBindTexture(GL_TEXTURE_2D,tq->tid);
  glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,cols,rows,0,GL_RGBA,GL_FLOAT,work_texture);
  free(work_texture);

  tq->dirty = false;
}

static void
_driver_init(
  driver* drv)
{
  assert(driver_is_valid(drv));

  tquad*  iter;

  iter = drv->quads->next;

  while (iter != drv->quads->prev) {
    if (iter->dirty) {
      if (iter->tid == 0) {
	_driver_init_tquad(drv,iter);
      }

      _driver_clean_tquad(drv,iter);
    }

    iter = iter->next;
  }
}


static void
_driver_display(
  driver* drv)
{
  assert(driver_is_valid(drv));

  tquad*  iter;

  glutSetWindowTitle(drv->title);
  glutPositionWindow(drv->geometry.x,drv->geometry.y);
  glutReshapeWindow(drv->geometry.w,drv->geometry.h);

  glClearColor(0,0,0,0);
  glLoadIdentity();
  glOrtho(-1,1,-1,1,-1,1);
  glScalef(2,2,1);
  glTranslatef(-0.5,0.5,0);
  glRotatef(180,1,0,0);

  iter = drv->quads->next;

  while (iter != drv->quads->prev) {
    if (iter->dirty) {
      if (iter->tid == 0) {
	_driver_init_tquad(drv,iter);
      }

      _driver_clean_tquad(drv,iter);
    }

    if (iter->show) {
      glBindTexture(GL_TEXTURE_2D,iter->tid);

      glBegin(GL_QUADS);
      glColor3f(1,1,1);
      glTexCoord2f(0,1);
      glVertex2f(iter->geometry.x,iter->geometry.y);
      glTexCoord2f(1,1);
      glVertex2f(iter->geometry.x + iter->geometry.w,iter->geometry.y);
      glTexCoord2f(1,0);
      glVertex2f(iter->geometry.x + iter->geometry.w,iter->geometry.y + iter->geometry.h);
      glTexCoord2f(0,0);
      glVertex2f(iter->geometry.x,iter->geometry.y + iter->geometry.h);
      glEnd();
    }
    
    iter = iter->next;
  }
}

static int
_driver_frame(
  driver* drv)
{
  assert(driver_is_valid(drv));

  if (drv->frame_cb) {
    return drv->frame_cb();
  } else {
    return 1;
  }
}


tquad*
driver_tquad_make_color(
  driver* drv,
  const rectangle* geometry,
  int rows,
  int cols,
  const color* color)
{
  assert(driver_is_valid(drv));
  assert(rectangle_is_valid(geometry));
  assert(rows > 0);
  assert(cols > 0);
  assert(color_is_valid(color));

  tquad*  new_tq;

  new_tq = malloc(sizeof(tquad));

  new_tq->drv = drv;
  new_tq->show = true;
  new_tq->dirty = true;
  new_tq->geometry = *geometry;
  new_tq->texture = image_make_blank(rows,cols,color);
  new_tq->tid = 0;
  new_tq->next = drv->quads->next;
  new_tq->prev = drv->quads;

  drv->quad_cnt = drv->quad_cnt + 1;
  drv->quads->next->prev = new_tq;
  drv->quads->next = new_tq;

  return new_tq;
}

tquad*
driver_tquad_make_image(
  driver* drv,
  const rectangle* geometry,
  const image* texture)
{
  assert(driver_is_valid(drv));
  assert(rectangle_is_valid(geometry));
  assert(image_is_valid(texture));

  tquad*  new_tq;

  new_tq = driver_tquad_make_color(drv,geometry,image_get_rows(texture),image_get_cols(texture),&(color){0,0,0,1});

  image_overwrite(new_tq->texture,texture);

  return new_tq;
}

tquad*
driver_tquad_make_copy(
  driver* drv,
  const tquad* tq)
{
  assert(driver_is_valid(drv));
  assert(tquad_is_valid(tq));
  assert(tq->drv == drv);
  assert(drv->quads != tq);
  assert(drv->quads->prev != tq);

  return driver_tquad_make_image(drv,&tq->geometry,tq->texture);
}

void
driver_tquad_free(
  driver* drv,
  tquad* tq)
{
  assert(driver_is_valid(drv));
  assert(tquad_is_valid(tq));
  assert(tq->drv == drv);
  assert(drv->quads != tq);
  assert(drv->quads->prev != tq);

  tq->next->prev = tq->prev;
  tq->prev->next = tq->next;

  tq->drv = NULL;
  tq->show = false;
  tq->dirty = false;
  rectangle_free(&tq->geometry);
  image_free(tq->texture);
  tq->texture = NULL;
  tq->tid = 0;
  tq->next = NULL;
  tq->prev = NULL;

  drv->quad_cnt = drv->quad_cnt - 1;

  free(tq);
}


const char*
driver_get_title(
  const driver* drv)
{
  assert(driver_is_valid(drv));

  return drv->title;
}

void
driver_set_title(
  driver* drv,
  const char* format,
  ...)
{
  assert(driver_is_valid(drv));
  assert(format != NULL && format[0] != '\0');

  va_list  args;

  free(drv->title);
  va_start(args,format);
  vasprintf(&drv->title,format,args);
  va_end(args);
}

void
driver_move_to(
  driver* drv,
  float x,
  float y)
{
  assert(driver_is_valid(drv));
  assert(x >= 0);
  assert(y >= 0);

  drv->geometry.x = x;
  drv->geometry.y = y;
}

void
driver_move_by(
  driver* drv,
  float x,
  float y)
{
  assert(driver_is_valid(drv));
  assert(drv->geometry.x + x >= 0);
  assert(drv->geometry.y + y >= 0);

  drv->geometry.x = drv->geometry.x + x;
  drv->geometry.y = drv->geometry.y + y;
}

void
driver_resize_to(
  driver* drv,
  float w,
  float h)
{
  assert(driver_is_valid(drv));
  assert(w > 0);
  assert(h > 0);

  drv->geometry.w = w;
  drv->geometry.h = h;
}

void
driver_resize_by(
  driver* drv,
  float w,
  float h)
{
  assert(driver_is_valid(drv));
  assert(drv->geometry.w + w > 0);
  assert(drv->geometry.h + h > 0);

  drv->geometry.w = drv->geometry.w + w;
  drv->geometry.h = drv->geometry.h + h;
}


bool
tquad_is_valid(
  const tquad* tq)
{
  if (tq == NULL) {
    return false;
  }

  if (tq->drv == NULL) {
    return false;
  }

  if (!rectangle_is_valid(&tq->geometry)) {
    return false;
  }

  if (!image_is_valid(tq->texture)) {
    return false;
  }

  if (!tq->dirty && tq->tid == 0) {
    return false;
  }

  if (tq->next == NULL) {
    return false;
  }

  if (tq->prev == NULL) {
    return false;
  }

  return true;
}


void
tquad_update_texture(
  tquad* tq,
  const image* img)
{
  assert(tquad_is_valid(tq));
  assert(image_is_valid(img));
  assert(image_get_rows(tq->texture) == image_get_rows(img));
  assert(image_get_cols(tq->texture) == image_get_cols(img));

  image_overwrite(tq->texture,img);
  tq->dirty = true;
}

void
tquad_move_to(
  tquad* tq,
  float x,
  float y)
{
  assert(tquad_is_valid(tq));

  tq->geometry.x = x;
  tq->geometry.y = y;
}

void
tquad_move_by(
  tquad* tq,
  float dx,
  float dy)
{
  assert(tquad_is_valid(tq));

  tq->geometry.x = tq->geometry.x + dx;
  tq->geometry.y = tq->geometry.y + dy;
}

void
tquad_resize_to(
  tquad* tq,
  float w,
  float h)
{
  assert(tquad_is_valid(tq));
  assert(w > 0);
  assert(h > 0);

  tq->geometry.w = w;
  tq->geometry.h = h;
}

void
tquad_resize_by(
  tquad* tq,
  float dw,
  float dh)
{
  assert(tquad_is_valid(tq));
  assert(tq->geometry.w + dw > 0);
  assert(tq->geometry.h + dh > 0);

  tq->geometry.w = tq->geometry.w + dw;
  tq->geometry.h = tq->geometry.h + dh;
}

void
tquad_show(
  tquad* tq)
{
  assert(tquad_is_valid(tq));

  tq->show = true;
}

void
tquad_hide(
  tquad* tq)
{
  assert(tquad_is_valid(tq));

  tq->show = false;
}

void
tquad_visiflip(
  tquad* tq)
{
  assert(tquad_is_valid(tq));

  tq->show = !tq->show;
}

const rectangle*
tquad_get_rectangle(
  const tquad* tq)
{
  assert(tquad_is_valid(tq));

  return &tq->geometry;
}

const image*
tquad_get_texture(
  const tquad* tq)
{
  assert(tquad_is_valid(tq));

  return tq->texture;
}

const color*
tquad_texture_get(
  const tquad* tq,
  int row,
  int col)
{
  assert(tquad_is_valid(tq));
  assert(row >= 0 && row < image_get_rows(tq->texture));
  assert(col >= 0 && col < image_get_cols(tq->texture));

  return image_get(tq->texture,row,col);
}

void
tquad_texture_set(
  tquad* tq,
  int row,
  int col,
  const color* c)
{
  assert(tquad_is_valid(tq));
  assert(row >= 0 && row < image_get_rows(tq->texture));
  assert(col >= 0 && col < image_get_cols(tq->texture));
  assert(color_is_valid(c));

  tq->dirty = true;
  image_set(tq->texture,row,col,c);
}


static struct {
  int      started;
  driver*  drv;
  int      w_id;
  jmp_buf  return_env;
} _glut_state = {
  .started = false,
  .drv = NULL,
  .w_id = -1
};

static void
_glut_display_cb()
{
  assert(_glut_state.started);
  assert(driver_is_valid(_glut_state.drv));
  assert(_glut_state.w_id > 0);

  glClear(GL_COLOR_BUFFER_BIT);
  _driver_display(_glut_state.drv);
  glutSwapBuffers();
}

static void
_glut_frame_cb()
{
  assert(_glut_state.started);
  assert(driver_is_valid(_glut_state.drv));
  assert(_glut_state.w_id > 0);

  int  next;

  next = _driver_frame(_glut_state.drv);
  glutPostRedisplay();

  if (next != 0) {
    glutTimerFunc(_glut_state.drv->ms_per_frame,_glut_frame_cb,0);
  } else {
    glutHideWindow();
    longjmp(_glut_state.return_env,1);
  }
}

static void
_glut_start(
  driver* drv)
{
  volatile int  jmp_ret;

  if (_glut_state.started) {
    _glut_state.drv = drv;
  } else {
    int           argc;
    char**        argv;

    argc = 1;
    argv = malloc(sizeof(char*) * 1);
    argv[0] = malloc(sizeof(char) * 32);
    strcpy(argv[0],"ShowNTell");

    glutInit(&argc,argv);

    free(argv[0]);
    free(argv);

    glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE);
    glutInitWindowSize(drv->geometry.w,drv->geometry.h);
    glutInitWindowPosition(drv->geometry.x,drv->geometry.y);

    _glut_state.started = true;
    _glut_state.drv = drv;
    _glut_state.w_id = glutCreateWindow(drv->title);

    glutDisplayFunc(_glut_display_cb);
  }

  _driver_init(drv);

  jmp_ret = setjmp(_glut_state.return_env);
    
  if (jmp_ret == 0) {
    glutTimerFunc(drv->ms_per_frame,_glut_frame_cb,0);
    glEnable(GL_TEXTURE_2D);
    glutShowWindow();
    glutMainLoop();
  } else {
    _glut_state.drv = NULL;
  }
}
