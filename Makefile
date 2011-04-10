LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):./lib

GOL_SRC_C = gol_main.c gol_data.c
GOL_SRC_H = gol_data.h
GOL_SRC = $(GOL_SRC_C) $(GOL_SRC_H)

gol-cpu-debug: $(GOL_SRC)
	gcc -g -Wall -DCPU -o gol-cpu-debug $(GOL_SRC_C) -L./lib -lsnt.debug

gol-cpu-final: $(GOL_SRC)
	gcc -O2 -Wall -DCPU -DNDEBUG -o gol-cpu-final $(GOL_SRC_C) -L./lib -lsnt

gol-gpu-debug: $(GOL_SRC)
	gcc -g -Wall -DCPU -o gol_main.o -c gol_main.c
	nvcc -g -G -DGPU -o gol_data.o -c -x cu gol_data.c
	nvcc -g -G -DGPU -o gol-gpu-debug gol_data.o gol_main.o -L./lib -lsnt.debug
	rm gol_main.o
	rm gol_data.o

gol-gpu-final: $(GOL_SRC)
	gcc -O2 -DCPU -DNDEBUG -o gol_main.o -c gol_main.c
	nvcc -O2 -DGPU -DNDEBUG -o gol_data.o -c -x cu gol_data.c
	nvcc -O2 -DGPU -DNDEBUG -o gol-gpu-final gol_data.o gol_main.o -L./lib -lsnt
	rm gol_main.o
	rm gol_data.o

gol-cpu-debug-run: gol-cpu-debug
	./gol-cpu-debug

gol-cpu-final-run: gol-cpu-final
	./gol-cpu-final

gol-gpu-debug-run: gol-gpu-debug
	./gol-gpu-debug

gol-gpu-final-run: gol-gpu-final
	./gol-gpu-final
