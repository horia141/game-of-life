LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):./lib

GOL_SRC_C = game_of_life.c
GOL_SRC_H =
GOL_SRC = $(GOL_SRC_C) $(GOL_SRC_H)

gol-cpu: $(GOL_SRC)
	gcc -g -Wall -o gol-cpu -DCPU $(GOL_SRC_C) -L./lib -lsnt

gol-gpu: $(SRC_C) $(SRC_H)
	nvcc -g -G -o gol-cpu -DCPU $(GOL_SRC_C) -L./lib -lsnt

gol-cpu-run: gol-cpu
	./gol-cpu

gol-gpu-run: gol-gpu
	./gol-gpu
