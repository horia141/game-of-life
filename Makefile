LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):./lib

GOL_SRC_C = gol_main.c gol_data.c
GOL_SRC_H = gol_data.h
GOL_SRC = $(GOL_SRC_C) $(GOL_SRC_H)

gol-cpu-debug: $(GOL_SRC)
	gcc -g -Wall -o gol-cpu-debug -DCPU $(GOL_SRC_C) -L./lib -lsnt.debug

gol-cpu-final: $(GOL_SRC)
	gcc -O2 -DNDEBUG -Wall -o gol-cpu-final -DCPU $(GOL_SRC_C) -L./lib -lsnt

gol-gpu: $(SRC_C) $(SRC_H)
	nvcc -g -G -o gol-cpu -DCPU $(GOL_SRC_C) -L./lib -lsnt

gol-cpu-debug-run: gol-cpu-debug
	./gol-cpu-debug

gol-cpu-final-run: gol-cpu-final
	./gol-cpu-final

gol-gpu-run: gol-gpu
	./gol-gpu
