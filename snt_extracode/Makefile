UTILS_SRC_C = utils.c
UTILS_SRC_H = utils.h
UTILS_SRC = $(UTILS_SRC_C) $(UTILS_SRC_H)

COLOR_SRC_C = color.c
COLOR_SRC_H = color.h
COLOR_SRC = $(COLOR_SRC_C) $(COLOR_SRC_H)

RECTANGLE_SRC_C = rectangle.c
RECTANGLE_SRC_H = rectangle.h
RECTANGLE_SRC = $(RECTANGLE_SRC_C) $(RECTANGLE_SRC_H)

IMAGE_SRC_C = image.c
IMAGE_SRC_H = image.h
IMAGE_SRC = $(IMAGE_SRC_C) $(IMAGE_SRC_H)

DRIVER_SRC_C = driver.c
DRIVER_SRC_H = driver.h
DRIVER_SRC = $(DRIVER_SRC_C) $(DRIVER_SRC_H)
DRIVER_LIB = -lglut

SRC = $(UTILS_SRC) $(COLOR_SRC) $(RECTANGLE_SRC) $(IMAGE_SRC) $(DRIVER_SRC)
LIB = $(UTILS_LIB) $(COLOR_LIB) $(RECTANGLE_LIB) $(IMAGE_LIB) $(DRIVER_LIB)

test-all: test-utils test-color test-rectangle test-image test-driver

test-utils: dirs $(UTILS_SRC) test_utils.c
	gcc -g -Wall -o ./out/test/utils $(UTILS_SRC_C) test_utils.c
	valgrind --leak-check=full --show-reachable=yes ./out/test/utils

test-color: dirs $(COLOR_SRC) test_color.c
	gcc -g -Wall -o ./out/test/color $(COLOR_SRC_C) test_color.c
	valgrind --leak-check=full --show-reachable=yes ./out/test/color

test-rectangle: dirs $(RECTANGLE_SRC) test_rectangle.c
	gcc -g -Wall -o ./out/test/rectangle $(RECTANGLE_SRC_C) test_rectangle.c
	valgrind --leak-check=full --show-reachable=yes ./out/test/rectangle

test-image: dirs $(UTILS_SRC) $(COLOR_SRC) $(IMAGE_SRC) test_image.c
	gcc -g -Wall -o ./out/test/image $(UTILS_SRC_C) $(COLOR_SRC_C) $(IMAGE_SRC_C) test_image.c
	valgrind --leak-check=full --show-reachable=yes ./out/test/image

test-driver: dirs $(UTILS_SRC) $(COLOR_SRC) $(RECTANGLE_SRC) $(IMAGE_SR) $(DRIVER_SRC) test_driver.c
	gcc -g -Wall -o ./out/test/driver -D_GNU_SOURCE $(DRIVER_LIB) $(UTILS_SRC_C) $(COLOR_SRC_C) $(RECTANGLE_SRC_C) $(IMAGE_SRC_C) $(DRIVER_SRC_C) test_driver.c
	valgrind --leak-check=full ./out/test/driver

dist-debug: dirs $(SRC)
	mkdir -p ./out/dist
	gcc -g -D_GNU_SOURCE -Wall -shared -Wl,-soname,libsnt.debug.so -o ./out/dist/libsnt.debug.so $(LIB) $(SRC)
	cp utils.h ./out/dist
	cp color.h ./out/dist
	cp rectangle.h ./out/dist
	cp image.h ./out/dist
	cp driver.h ./out/dist

dist-final: dirs $(SRC)
	mkdir -p ./out/dist
	gcc -O2 -DNDEBUG -D_GNU_SOURCE -Wall -shared -Wl,-soname,libsnt.so -o ./out/dist/libsnt.so $(LIB) $(SRC)
	cp utils.h ./out/dist
	cp color.h ./out/dist
	cp rectangle.h ./out/dist
	cp image.h ./out/dist
	cp driver.h ./out/dist

dirs:
	mkdir -p ./out
	mkdir -p ./out/test
	mkdir -p ./out/dist
	mkdir -p ./out/tmp
