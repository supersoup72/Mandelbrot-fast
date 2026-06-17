/* mandelbrot.c — fast terminal Mandelbrot explorer with perturbation theory
 *
 *  wasd    pan              z / x   zoom in / out
 *  i / o   more/fewer iter  k / l   color density up / down
 *  p       save PNG         t       toggle truecolor / 16-color
 *  q       quit
 *
 *  Rendering uses Unicode upper-half-block glyphs (▀) with independent
 *  foreground/background colors, packing two vertical pixels per
 *  terminal cell for roughly double the effective resolution. Colors are
 *  24-bit truecolor by default, or the basic ANSI 16-color palette (for
 *  terminals without truecolor support) via the 't' toggle.
 *
 *  Build:
 *    gcc -O3 -march=native -funroll-loops -ffast-math -fopenmp -mavx2 -mfma \
 *        -o mandelbrot mandelbrot.c orbit.c png_writer.c -lm -lz -fopenmp
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include "orbit.h"
#include "png_writer.h"

/* ─── terminal ──────────────────────────────────────────────────────────── */

static struct termios        g_orig;
static volatile sig_atomic_t g_resize;

static void on_winch(int s) { (void)s; g_resize = 1; }

static void term_restore(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig);
    (void)write(STDOUT_FILENO, "\033[?25h\033[?1049l", 14);
}

static void term_init(void) {
    tcgetattr(STDIN_FILENO, &g_orig);
    atexit(term_restore);

    struct sigaction sa = { .sa_handler = on_winch };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGWINCH, &sa, NULL);

    struct termios raw = g_orig;
    raw.c_iflag &= ~(IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    /* Leave c_oflag (and ONLCR) untouched: render() writes rows terminated
     * by a bare '\n', relying on the kernel's normal LF->CRLF translation
     * to return the cursor to column 1. Clearing OPOST turns that off and
     * corrupts every row after the first via column drift.                */
    raw.c_cflag |=  CS8;
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    (void)write(STDOUT_FILENO, "\033[?1049h\033[?25l", 14);
}

static void term_size(int *w, int *h) {
    struct winsize ws = {0};
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
    *w = ws.ws_col ? ws.ws_col : 80;
    *h = ws.ws_row ? ws.ws_row : 24;
}

/* ─── view state ────────────────────────────────────────────────────────── */

static double g_cx   = -0.5;
static double g_cy   =  0.0;
static double g_zoom =  1.0;
static int    g_iter =  128;
static double g_dens =  8.0;
static double g_render_ms = 0.0;
static char   g_msg[160] = "";
static int    g_truecolor = 1;

/* ─── view geometry ──────────────────────────────────────────────────────
 * Coordinates are sampled on a grid of square "sub-pixels": W columns by
 * ph = 2*h sub-rows, two sub-rows packed per terminal row via half-block
 * glyphs (one as foreground color, one as background). Because a terminal
 * cell is about twice as tall as wide, splitting its height in two makes
 * each sub-pixel roughly square, so a single step size sx == sy applies to
 * both axes — no separate aspect-ratio correction needed.                 */

static void view_geometry(int *Wp, int *hp, int *php,
                           double *sxp, double *syp, double *x0p, double *y0p) {
    int W, H;
    term_size(&W, &H);
    int h = H - 1;
    if (h < 1) h = 1;
    if (W < 1) W = 1;
    int ph = 2 * h;

    double sx = 1.25 / g_zoom / h;
    double sy = sx;
    double x0 = g_cx - W  * 0.5 * sx;
    double y0 = g_cy - ph * 0.5 * sy;

    *Wp = W; *hp = h; *php = ph;
    *sxp = sx; *syp = sy; *x0p = x0; *y0p = y0;
}

/* ─── scalar fallback Mandelbrot (used only if orbit allocation fails) ──── */

static int scalar_fallback(double cr, double ci, int max_iter) {
    double p = cr - 0.25;
    double q = p*p + ci*ci;
    if (q*(q+p) <= 0.25*ci*ci || (cr+1.0)*(cr+1.0) + ci*ci <= 0.0625)
        return max_iter;
    double zr=0, zi=0, zr2=0, zi2=0;
    int it=0;
    while (zr2+zi2 < 4.0 && it < max_iter) {
        zi  = 2.0*zr*zi + ci;
        zr  = zr2-zi2 + cr;
        zr2 = zr*zr; zi2 = zi*zi;
        it++;
    }
    return it;
}

/* ─── continuous truecolor iteration→RGB table ──────────────────────────── */

typedef struct { unsigned char r, g, b; } RGB;
static RGB *g_tab    = NULL;
static int  g_tabcap = 0;

static void build_itable(void) {
    int need = g_iter + 1;
    if (g_tabcap < need) {
        free(g_tab);
        g_tab    = malloc(need * sizeof *g_tab);
        g_tabcap = need;
    }
    g_tab[g_iter].r = g_tab[g_iter].g = g_tab[g_iter].b = 0; /* inside set: black */

    for (int i = 0; i < g_iter; i++) {
        double t = fmod((double)i / g_iter * g_dens, 1.0);
        double a = t * 6.283185307179586;
        double r = (sin(a)               + 1.0) * 0.5;
        double g = (sin(a + 2.094395102) + 1.0) * 0.5;
        double b = (sin(a + 4.188790205) + 1.0) * 0.5;
        g_tab[i].r = (unsigned char)(r * 255.0 + 0.5);
        g_tab[i].g = (unsigned char)(g * 255.0 + 0.5);
        g_tab[i].b = (unsigned char)(b * 255.0 + 0.5);
    }
}

/* ─── 16-color ANSI fallback (for terminals without truecolor) ──────────── */

static const RGB ANSI16[16] = {
    {  0,  0,  0}, {205,  0,  0}, {  0,205,  0}, {205,205,  0},
    {  0,  0,205}, {205,  0,205}, {  0,205,205}, {229,229,229},
    {127,127,127}, {255,  0,  0}, {  0,255,  0}, {255,255,  0},
    {  0,  0,255}, {255,  0,255}, {  0,255,255}, {255,255,255},
};

static int nearest_ansi16(RGB c) {
    int best = 0;
    long bestd = -1;
    for (int i = 0; i < 16; i++) {
        long dr = (long)c.r - ANSI16[i].r;
        long dg = (long)c.g - ANSI16[i].g;
        long db = (long)c.b - ANSI16[i].b;
        long d = dr*dr + dg*dg + db*db;
        if (bestd < 0 || d < bestd) { bestd = d; best = i; }
    }
    return best;
}

/* ─── output buffers ────────────────────────────────────────────────────── */

static char **g_rows  = NULL;
static int   *g_rlens = NULL;
static int    g_rcap  = 0;
static char  *g_frame = NULL;
static int    g_fcap  = 0;

/* worst case per cell: fg + bg 24-bit SGR codes ("\033[38;2;255;255;255m"
 * is 19 bytes, same for bg) plus the 3-byte UTF-8 "▀" glyph.              */
#define PX_MAX 64
#define STATUS_RESERVE 1024

static void ensure_bufs(int w, int h) {
    if (h > g_rcap) {
        g_rows  = realloc(g_rows,  h * sizeof *g_rows);
        g_rlens = realloc(g_rlens, h * sizeof *g_rlens);
        for (int i = g_rcap; i < h; i++) g_rows[i] = NULL;
        g_rcap = h;
    }
    int rw = w * PX_MAX + 4;
    for (int i = 0; i < h; i++) g_rows[i] = realloc(g_rows[i], rw);
    int fw = h * rw + STATUS_RESERVE;
    if (fw > g_fcap) { free(g_frame); g_frame = malloc(fw); g_fcap = fw; }
}

/* ─── render ────────────────────────────────────────────────────────────── */

static void render(void) {
    int W, h, ph;
    double sx, sy, x0, y0;
    view_geometry(&W, &h, &ph, &sx, &sy, &x0, &y0);
    ensure_bufs(W, h);

    double half_w = sx * W  * 0.5;
    double half_h = sy * ph * 0.5;
    double max_delta = sqrt(half_w*half_w + half_h*half_h);

    int mi = g_iter;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Orbit *orb = orbit_new(g_cx, g_cy, mi, max_delta);
    int sa_skip  = orb ? orb->sa_skip    : 0;
    int bla_levs = orb ? orb->bla_levels : 0;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < h; y++) {
        char *rb = g_rows[y];
        int   rp = 0;
        RGB   prev_fg = {0,0,0}, prev_bg = {0,0,0};
        int   have_fg = 0, have_bg = 0;

        double ci_top = y0 + (2*y)     * sy;
        double ci_bot = y0 + (2*y + 1) * sy;

        double *dcr     = malloc(W * sizeof(double));
        double *dci_top = malloc(W * sizeof(double));
        double *dci_bot = malloc(W * sizeof(double));
        int    *irow_top= malloc(W * sizeof(int));
        int    *irow_bot= malloc(W * sizeof(int));

        if (!dcr || !dci_top || !dci_bot || !irow_top || !irow_bot) {
            free(dcr); free(dci_top); free(dci_bot); free(irow_top); free(irow_bot);
            g_rlens[y] = 0;
            continue;
        }

        for (int x = 0; x < W; x++) {
            dcr[x]     = x0 + x * sx - g_cx;
            dci_top[x] = ci_top - g_cy;
            dci_bot[x] = ci_bot - g_cy;
        }

        if (orb) {
            orbit_render_row(orb, dcr, dci_top, W, mi, irow_top);
            orbit_render_row(orb, dcr, dci_bot, W, mi, irow_bot);
        } else {
            for (int x = 0; x < W; x++) {
                irow_top[x] = scalar_fallback(g_cx + dcr[x], g_cy + dci_top[x], mi);
                irow_bot[x] = scalar_fallback(g_cx + dcr[x], g_cy + dci_bot[x], mi);
            }
        }

        for (int x = 0; x < W; x++) {
            RGB fg = g_tab[irow_top[x] < mi ? irow_top[x] : mi];
            RGB bg = g_tab[irow_bot[x] < mi ? irow_bot[x] : mi];

            if (!have_fg || fg.r!=prev_fg.r || fg.g!=prev_fg.g || fg.b!=prev_fg.b) {
                if (g_truecolor) {
                    rp += snprintf(rb+rp, 20, "\033[38;2;%d;%d;%dm", fg.r, fg.g, fg.b);
                } else {
                    int idx = nearest_ansi16(fg);
                    rp += snprintf(rb+rp, 12, "\033[%dm", idx < 8 ? 30+idx : 90+(idx-8));
                }
                prev_fg = fg; have_fg = 1;
            }
            if (!have_bg || bg.r!=prev_bg.r || bg.g!=prev_bg.g || bg.b!=prev_bg.b) {
                if (g_truecolor) {
                    rp += snprintf(rb+rp, 20, "\033[48;2;%d;%d;%dm", bg.r, bg.g, bg.b);
                } else {
                    int idx = nearest_ansi16(bg);
                    rp += snprintf(rb+rp, 12, "\033[%dm", idx < 8 ? 40+idx : 100+(idx-8));
                }
                prev_bg = bg; have_bg = 1;
            }
            rb[rp++] = '\xe2'; rb[rp++] = '\x96'; rb[rp++] = '\x80'; /* ▀ */
        }
        rb[rp++] = '\n';
        g_rlens[y] = rp;

        free(dcr); free(dci_top); free(dci_bot); free(irow_top); free(irow_bot);
    }

    orbit_free(orb);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_render_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Assemble frame and single write() */
    int fp = 0;
    memcpy(g_frame+fp, "\033[H", 3); fp += 3;
    for (int y = 0; y < h; y++) {
        if (g_rlens[y] <= 0) continue;
        memcpy(g_frame+fp, g_rows[y], g_rlens[y]);
        fp += g_rlens[y];
    }

    /* Status bar with SA/BLA debug info, render time, and last message */
    fp += snprintf(g_frame+fp, STATUS_RESERVE,
        "\033[0m\033[%d;1H\033[K"
        "  \033[1mcx\033[0m=%-14.8g"
        "  \033[1mcy\033[0m=%-14.8g"
        "  \033[1mzoom\033[0m=%-11.5g"
        "  \033[1miter\033[0m=%-4d"
        "  SA=%-4d BLA=%-2d"
        "  render=%.1fms"
        "  color=%s"
        "%s%s"
        "  \033[2m[wasd]move [z/x]zoom [i/o]iter [k/l]color [p]png [t]color-mode [q]quit\033[0m",
        h+1, g_cx, g_cy, g_zoom, g_iter,
        sa_skip, bla_levs, g_render_ms,
        g_truecolor ? "truecolor" : "16-color",
        g_msg[0] ? "  " : "", g_msg);

    (void)write(STDOUT_FILENO, g_frame, fp);
    g_resize = 0;
}

/* ─── PNG export ────────────────────────────────────────────────────────── */

static void save_png(void) {
    int W, h, ph;
    double sx, sy, x0, y0;
    view_geometry(&W, &h, &ph, &sx, &sy, &x0, &y0);

    /* Oversample the same view bounds onto a higher-resolution grid
     * (target ~1920 px wide) rather than just upscaling terminal cells. */
    int scale = 1920 / W;
    if (scale < 1) scale = 1;
    int pw  = W  * scale;
    int phh = ph * scale;
    double psx = sx / scale;
    double psy = sy / scale;

    double half_w = psx * pw  * 0.5;
    double half_h = psy * phh * 0.5;
    double max_delta = sqrt(half_w*half_w + half_h*half_h);

    int mi = g_iter;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Orbit *orb = orbit_new(g_cx, g_cy, mi, max_delta);

    unsigned char *rgb = malloc((size_t)pw * phh * 3);
    if (!rgb) { orbit_free(orb); snprintf(g_msg, sizeof g_msg, "png save failed: out of memory"); return; }

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < phh; y++) {
        double ci = y0 + y * psy - g_cy;

        double *dcr = malloc(pw * sizeof(double));
        double *dci = malloc(pw * sizeof(double));
        int    *irow= malloc(pw * sizeof(int));
        if (!dcr || !dci || !irow) { free(dcr); free(dci); free(irow); continue; }

        for (int x = 0; x < pw; x++) {
            dcr[x] = x0 + x * psx - g_cx;
            dci[x] = ci;
        }

        if (orb) {
            orbit_render_row(orb, dcr, dci, pw, mi, irow);
        } else {
            for (int x = 0; x < pw; x++)
                irow[x] = scalar_fallback(g_cx + dcr[x], g_cy + dci[x], mi);
        }

        unsigned char *row = rgb + (size_t)y * pw * 3;
        for (int x = 0; x < pw; x++) {
            RGB c = g_tab[irow[x] < mi ? irow[x] : mi];
            row[x*3+0] = c.r; row[x*3+1] = c.g; row[x*3+2] = c.b;
        }

        free(dcr); free(dci); free(irow);
    }

    orbit_free(orb);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    /* Sequence number guards against collisions when multiple exports
     * land within the same wall-clock second (timestamp alone isn't
     * unique enough since a render can finish in single-digit ms).    */
    static int seq = 0;
    time_t now = time(NULL);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", localtime(&now));
    char fname[64];
    snprintf(fname, sizeof fname, "mandelbrot_%s_%03d.png", stamp, seq++);

    int ok = write_png(fname, pw, phh, rgb) == 0;
    free(rgb);

    if (ok)
        snprintf(g_msg, sizeof g_msg, "saved %s (%dx%d, %.0fms)", fname, pw, phh, ms);
    else
        snprintf(g_msg, sizeof g_msg, "png save failed: %s", fname);
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    term_init();
    build_itable();
    render();

    for (;;) {
        char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n < 0) {
            if (errno == EINTR) { if (g_resize) render(); }
            continue;
        }
        if (n == 0) break;

        if (c != 'p' && c != 'P') g_msg[0] = '\0';

        int dirty = 1, need_iet = 0;
        double vs = 0.25 / g_zoom;
        double hs = 0.50 / g_zoom;

        switch (c) {
        case 'q': case 'Q': goto quit;

        case 'w': case 'W': g_cy -= vs; break;
        case 's': case 'S': g_cy += vs; break;
        case 'a': case 'A': g_cx -= hs; break;
        case 'd': case 'D': g_cx += hs; break;

        case 'z': case 'Z':
            if (g_zoom < 1e14) g_zoom *= 1.5;
            break;
        case 'x': case 'X':
            if (g_zoom > 1e-4) g_zoom /= 1.5;
            break;

        case 'i': case 'I':
            g_iter = g_iter < 4096 ? g_iter * 3 / 2 + 1 : g_iter;
            need_iet = 1; break;
        case 'o': case 'O':
            g_iter = g_iter > 8 ? g_iter * 2 / 3 : g_iter;
            need_iet = 1; break;

        case 'k': case 'K': g_dens *= 1.5; need_iet = 1; break;
        case 'l': case 'L': g_dens /= 1.5; need_iet = 1; break;

        case 'p': case 'P': save_png(); break;

        case 't': case 'T': g_truecolor = !g_truecolor; break;

        default: dirty = 0; break;
        }

        if (need_iet) build_itable();
        if (dirty)    render();
    }
quit:
    return 0;
}
