/* png_writer.h — minimal 8-bit RGB PNG writer (uses zlib for deflate/CRC) */
#ifndef PNG_WRITER_H
#define PNG_WRITER_H

/* rgb holds width*height*3 bytes, row-major, top to bottom. level is the
 * zlib deflate level (0-9, or Z_DEFAULT_COMPRESSION); lower is faster but
 * produces bigger files — useful for disposable animation frames destined
 * for ffmpeg, where write speed matters more than size.
 * Returns 0 on success, -1 on failure (file or memory error). */
int write_png(const char *filename, int width, int height, const unsigned char *rgb, int level);

#endif /* PNG_WRITER_H */
