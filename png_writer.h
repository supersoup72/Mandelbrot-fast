/* png_writer.h — minimal 8-bit RGB PNG writer (uses zlib for deflate/CRC) */
#ifndef PNG_WRITER_H
#define PNG_WRITER_H

/* rgb holds width*height*3 bytes, row-major, top to bottom.
 * Returns 0 on success, -1 on failure (file or memory error). */
int write_png(const char *filename, int width, int height, const unsigned char *rgb);

#endif /* PNG_WRITER_H */
