CC      = gcc
CFLAGS  = -O3 -march=native -funroll-loops -ffast-math -fopenmp -Wall -Wextra -Wno-unused-result
LIBS    = -lm -fopenmp
TARGET  = mandelbrot

.PHONY: all clean

all: $(TARGET)

$(TARGET): mandelbrot.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET)
