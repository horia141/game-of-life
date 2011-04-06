gol-cpu: GameOfLife.c
	gcc -g -Wall --pedantic -DCPU -o gol-cpu GameOfLife.c

gol-gpu: GameOfLife.c
	nvcc -g -G -DGPU -o gol-gpu -x cu GameOfLife.c
