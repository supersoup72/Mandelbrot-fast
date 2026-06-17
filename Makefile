CC     = gcc
CFLAGS = -O3 -march=native -funroll-loops -ffast-math -fopenmp -mavx2 -mfma -Wall -Wextra -Wno-unused-result
LIBS   = -lm -lz -lmpfr -lgmp -fopenmp

.PHONY: all clean

all: mandelbrot

mandelbrot: mandelbrot.c orbit.c orbit.h png_writer.c png_writer.h
	$(CC) $(CFLAGS) -o $@ mandelbrot.c orbit.c png_writer.c $(LIBS)

clean:
	rm -f mandelbrot
