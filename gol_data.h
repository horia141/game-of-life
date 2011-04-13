#ifndef _GOL_DATA_H
#define _GOL_DATA_H

#include <stdbool.h>

#define ALIVE 1
#define DEAD 0

typedef struct _gol_data  gol_data;

#ifdef GPU
extern "C" {
#endif

gol_data*  gol_data_from_l(const char* path);
void       gol_data_free(gol_data* gol);

bool       gol_data_is_valid(const gol_data* gol);

gol_data*  gol_data_evolve(gol_data* gol, int iters);
int        gol_data_neigh_count(const gol_data* gol, int i, int j);
int        gol_data_get_rows(const gol_data* gol);
int        gol_data_get_cols(const gol_data* gol);
int        gol_data_get(const gol_data* gol, int i, int j);
void       gol_data_set(gol_data* gol, int i, int j, int cell_state);

void       gol_data_save_l(const gol_data* gol, const char* path);
  
#ifdef GPU
}
#endif

#endif
