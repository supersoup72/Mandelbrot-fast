/* png_writer.c — minimal 8-bit RGB PNG writer (uses zlib for deflate/CRC) */
#include "png_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static void put_u32(unsigned char *p, unsigned long v) {
    p[0] = (unsigned char)(v >> 24); p[1] = (unsigned char)(v >> 16);
    p[2] = (unsigned char)(v >> 8);  p[3] = (unsigned char)v;
}

static void write_chunk(FILE *f, const char *type, const unsigned char *data, unsigned long len) {
    unsigned char hdr[8];
    put_u32(hdr, len);
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (len) fwrite(data, 1, len, f);

    unsigned long crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, (const unsigned char *)type, 4);
    if (len) crc = crc32(crc, data, len);
    unsigned char crcbuf[4];
    put_u32(crcbuf, crc);
    fwrite(crcbuf, 1, 4, f);
}

int write_png(const char *filename, int width, int height, const unsigned char *rgb)
{
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    static const unsigned char sig[8] = { 0x89,'P','N','G','\r','\n',0x1a,'\n' };
    fwrite(sig, 1, 8, f);

    unsigned char ihdr[13];
    put_u32(ihdr,     (unsigned long)width);
    put_u32(ihdr + 4, (unsigned long)height);
    ihdr[8]  = 8; /* bit depth       */
    ihdr[9]  = 2; /* color type: RGB */
    ihdr[10] = 0; /* compression     */
    ihdr[11] = 0; /* filter          */
    ihdr[12] = 0; /* interlace       */
    write_chunk(f, "IHDR", ihdr, sizeof ihdr);

    /* Raw scanlines: a leading filter-type byte (0 = none) per row. */
    unsigned long raw_len = (unsigned long)height * (1 + (unsigned long)width * 3);
    unsigned char *raw = malloc(raw_len);
    if (!raw) { fclose(f); return -1; }
    unsigned char *rp = raw;
    for (int y = 0; y < height; y++) {
        *rp++ = 0;
        memcpy(rp, rgb + (size_t)y * width * 3, (size_t)width * 3);
        rp += (size_t)width * 3;
    }

    unsigned long comp_cap = compressBound(raw_len);
    unsigned char *comp = malloc(comp_cap);
    if (!comp) { free(raw); fclose(f); return -1; }
    unsigned long comp_len = comp_cap;
    int zr = compress2(comp, &comp_len, raw, raw_len, 6);
    free(raw);
    if (zr != Z_OK) { free(comp); fclose(f); return -1; }

    write_chunk(f, "IDAT", comp, comp_len);
    free(comp);

    write_chunk(f, "IEND", NULL, 0);
    fclose(f);
    return 0;
}
