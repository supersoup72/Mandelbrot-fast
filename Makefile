CC     = gcc
# Targets AMD EPYC "Genoa" (Zen4) explicitly rather than -march=native: native
# bakes in whatever ISA the *build* machine happens to expose, which is wrong
# whenever the binary is compiled somewhere other than the actual target CPU
# (e.g. this would silently enable AMX/AVX512FP16/AVXVNNI if built on an
# Intel Sapphire-Rapids-class box, none of which exist on Zen4 -- the
# resulting binary would SIGILL on the real target). znver4 still gets the
# full AVX-512 (F/BW/DQ/VL/CD/IFMA/VBMI2/VNNI/BITALG/VPOPCNTDQ/BF16) + GFNI/
# VAES/VPCLMULQDQ feature set Genoa actually has, plus correct Zen4
# scheduling/cost tables for instruction selection.
CFLAGS = -O3 -march=znver4 -mtune=znver4 -funroll-loops -ffast-math -fopenmp -Wall -Wextra -Wno-unused-result
LIBS   = -lm -lz -lmpfr -lgmp -fopenmp

.PHONY: all clean

all: mandelbrot

mandelbrot: mandelbrot.c orbit.c orbit.h png_writer.c png_writer.h
	$(CC) $(CFLAGS) -o $@ mandelbrot.c orbit.c png_writer.c $(LIBS)

clean:
	rm -f mandelbrot
