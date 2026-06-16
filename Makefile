CC     = gcc
CFLAGS = -O3 -march=native -funroll-loops -ffast-math -fopenmp -mavx2 -mfma -Wall -Wextra -Wno-unused-result
LIBS   = -lm -fopenmp

.PHONY: all clean

all: mandelbrot

mandelbrot: mandelbrot.c orbit.c orbit.h
	$(CC) $(CFLAGS) -o $@ mandelbrot.c orbit.c $(LIBS)

clean:
	rm -f mandelbrot
