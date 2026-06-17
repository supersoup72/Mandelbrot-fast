/* mandelbrot.c — fast terminal Mandelbrot explorer with perturbation theory
 *
 *  wasd    pan              z / x   zoom in / out
 *  i / o   more/fewer iter  k / l   color density up / down
 *  q       quit
 *
 *  Build:
 *    gcc -O3 -march=native -funroll-loops -ffast-math -fopenmp -mavx2 -mfma \
 *        -o mandelbrot mandelbrot.c orbit.c -lm -fopenmp
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
#include <unistd.h>
#include "orbit.h"

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
    raw.c_oflag &= ~OPOST;
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

/* ─── 256-colour palette (6×6×6 cube, indices 16–231) ──────────────────── */

typedef struct { char s[12]; int n; } ColEsc;
static ColEsc g_pal[216];

static void build_palette(void) {
    for (int i = 0; i < 216; i++)
        g_pal[i].n = snprintf(g_pal[i].s, 12, "\033[38;5;%dm", 16 + i);
}

/* ─── per-iteration lookup table ────────────────────────────────────────── */

typedef struct { int pi; char ch; } IterEntry;
static IterEntry *g_tab    = NULL;
static int        g_tabcap = 0;

static const char GLYPHS[] = " .:!|=+*#%@";
#define NG ((int)(sizeof(GLYPHS) - 1))

static void build_itable(void) {
    int need = g_iter + 2;
    if (g_tabcap < need) {
        free(g_tab);
        g_tab    = malloc(need * sizeof *g_tab);
        g_tabcap = need;
    }
    g_tab[g_iter].pi = -1;
    g_tab[g_iter].ch = ' ';

    for (int i = 0; i < g_iter; i++) {
        double t = fmod((double)i / g_iter * g_dens, 1.0);
        double a = t * 6.283185307179586;
        int r = (int)(sin(a)            * 2.5 + 2.5); if (r<0)r=0; if (r>5)r=5;
        int g = (int)(sin(a + 2.094395) * 2.5 + 2.5); if (g<0)g=0; if (g>5)g=5;
        int b = (int)(sin(a + 4.188790) * 2.5 + 2.5); if (b<0)b=0; if (b>5)b=5;
        g_tab[i].pi = 36*r + 6*g + b;
        g_tab[i].ch = GLYPHS[(int)(t * NG) % NG];
    }
}

/* ─── output buffers ────────────────────────────────────────────────────── */

static char **g_rows  = NULL;
static int   *g_rlens = NULL;
static int    g_rcap  = 0;
static char  *g_frame = NULL;
static int    g_fcap  = 0;

#define PX_MAX 13   /* max bytes per pixel: 11 (esc) + 1 (char) + spare */

static void ensure_bufs(int w, int h) {
    if (h > g_rcap) {
        g_rows  = realloc(g_rows,  h * sizeof *g_rows);
        g_rlens = realloc(g_rlens, h * sizeof *g_rlens);
        for (int i = g_rcap; i < h; i++) g_rows[i] = NULL;
        g_rcap = h;
    }
    int rw = w * PX_MAX + 4;
    for (int i = 0; i < h; i++) g_rows[i] = realloc(g_rows[i], rw);
    int fw = h * rw + 512;
    if (fw > g_fcap) { free(g_frame); g_frame = malloc(fw); g_fcap = fw; }
}

static const char BLK[]  = "\033[38;5;232m";
#define BLK_N 11

/* ─── render ────────────────────────────────────────────────────────────── */

static void render(void) {
    int W, H;
    term_size(&W, &H);
    int h = H - 1;           /* reserve last row for status bar */
    if (h < 1 || W < 1) return;
    ensure_bufs(W, h);

    /* coordinate step sizes (terminal cells are ~2x taller than wide,
     * so a column must cover half the coordinate distance a row does) */
    double sy = 2.5 / g_zoom / h;
    double sx = sy * 0.5;
    double x0 = g_cx - W * 0.5 * sx;
    double y0 = g_cy - h * 0.5 * sy;

    /* max_delta: half-diagonal of the view in coordinate space.
     * This is used by the SA validity criterion.                  */
    double max_delta = sqrt((sx * W) * (sx * W) + (sy * h) * (sy * h)) * 0.5;

    int mi = g_iter;

    /* Build reference orbit at view center */
    Orbit *orb = orbit_new(g_cx, g_cy, mi, max_delta);

    int sa_skip   = orb ? orb->sa_skip   : 0;
    int bla_levs  = orb ? orb->bla_levels: 0;

    /* Per-row iteration buffer — allocate once, reused per thread */
    /* Each thread needs its own scratch; use thread-private or alloc per row */

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < h; y++) {
        char *rb  = g_rows[y];
        int   rp  = 0;
        int   ppi = -2;
        double ci = y0 + y * sy;

        /* Build Δc arrays for this row */
        double *dcr = malloc(W * sizeof(double));
        double *dci = malloc(W * sizeof(double));
        int    *irow= malloc(W * sizeof(int));

        if (!dcr || !dci || !irow) {
            free(dcr); free(dci); free(irow);
            g_rlens[y] = 0;
            continue;
        }

        for (int x = 0; x < W; x++) {
            dcr[x] = x0 + x * sx - g_cx;
            dci[x] = ci - g_cy;
        }

        if (orb) {
            orbit_render_row(orb, dcr, dci, W, mi, irow);
        } else {
            /* orbit allocation failed: scalar fallback */
            for (int x = 0; x < W; x++) {
                double cr = g_cx + dcr[x];
                double cci= g_cy + dci[x];
                double p  = cr - 0.25;
                double q  = p*p + cci*cci;
                if (q * (q + p) <= 0.25 * cci*cci ||
                    (cr+1.0)*(cr+1.0) + cci*cci <= 0.0625) {
                    irow[x] = mi;
                } else {
                    double zr=0,zi=0,zr2=0,zi2=0;
                    int it=0;
                    while (zr2+zi2 < 4.0 && it < mi) {
                        zi  = 2.0*zr*zi + cci;
                        zr  = zr2-zi2 + cr;
                        zr2 = zr*zr; zi2=zi*zi;
                        it++;
                    }
                    irow[x] = it;
                }
            }
        }

        /* Encode row into escape sequences */
        for (int x = 0; x < W; x++) {
            int iter = irow[x];
            IterEntry *e = &g_tab[iter < mi ? iter : mi];

            if (e->pi != ppi) {
                if (e->pi < 0) {
                    memcpy(rb+rp, BLK, BLK_N); rp += BLK_N;
                } else {
                    ColEsc *ce = &g_pal[e->pi];
                    memcpy(rb+rp, ce->s, ce->n); rp += ce->n;
                }
                ppi = e->pi;
            }
            rb[rp++] = e->ch;
        }
        rb[rp++] = '\n';
        g_rlens[y] = rp;

        free(dcr);
        free(dci);
        free(irow);
    }

    orbit_free(orb);

    /* Assemble frame and single write() */
    int fp = 0;
    memcpy(g_frame+fp, "\033[H", 3); fp += 3;
    for (int y = 0; y < h; y++) {
        if (g_rlens[y] <= 0) continue;
        memcpy(g_frame+fp, g_rows[y], g_rlens[y]);
        fp += g_rlens[y];
    }

    /* Status bar with SA and BLA debug info */
    fp += snprintf(g_frame+fp, 512,
        "\033[0m\033[%d;1H\033[K"
        "  \033[1mcx\033[0m=%-14.8g"
        "  \033[1mcy\033[0m=%-14.8g"
        "  \033[1mzoom\033[0m=%-11.5g"
        "  \033[1miter\033[0m=%-4d"
        "  \033[1mdens\033[0m=%.2f"
        "  SA skip=%-4d BLA levels=%-2d"
        "  \033[2m[wasd]move [z/x]zoom [i/o]iter [k/l]color [q]quit\033[0m",
        H, g_cx, g_cy, g_zoom, g_iter, g_dens,
        sa_skip, bla_levs);

    (void)write(STDOUT_FILENO, g_frame, fp);
    g_resize = 0;
}

/* ─── main ──────────────────────────────────────────────────────────────── */

int main(void) {
    build_palette();
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

        default: dirty = 0; break;
        }

        if (need_iet) build_itable();
        if (dirty)    render();
    }
quit:
    return 0;
}
