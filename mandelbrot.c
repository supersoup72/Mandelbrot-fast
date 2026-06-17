/* mandelbrot.c — fast terminal Mandelbrot explorer with perturbation theory
 *
 *  wasd    pan              z / x   zoom in / out
 *  i / o   more/fewer iter  k / l   color density up / down
 *  p       save PNG         t       toggle truecolor / 16-color
 *  M       zoom-out animation: saves numbered PNG frames (prompts for
 *          per-frame zoom factor); concat with ffmpeg afterwards
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
#include <poll.h>
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

/* Deliberately does NOT return an absolute view origin (g_cx/g_cy - half
 * extent): per-pixel deltas are derived purely from pixel index and step
 * size (see render()/render_view_to_png()), never added to and subtracted
 * back from the absolute center. That round-trip used to be exactly how
 * dcr/dci were computed and it throws away precision for no reason — once
 * sx is small relative to g_cx's own ULP, "g_cx - half_w" already rounds,
 * and the subsequent "+ x*sx - g_cx" compounds it further. Avoiding the
 * round-trip buys several extra usable bits of zoom depth for free.       */
static void view_geometry(int *Wp, int *hp, int *php,
                           double *sxp, double *syp) {
    int W, H;
    term_size(&W, &H);
    int h = H - 1;
    if (h < 1) h = 1;
    if (W < 1) W = 1;
    int ph = 2 * h;

    double sx = 1.25 / g_zoom / h;
    double sy = sx;

    *Wp = W; *hp = h; *php = ph;
    *sxp = sx; *syp = sy;
}

/* ─── cached reference orbit ─────────────────────────────────────────────
 * The reference orbit, SA arrays, and BLA pyramid depend only on (cx, cy,
 * max_iter) — not on zoom — so a pure zoom animation (the common case)
 * can reuse the same orbit across every frame and pay only the cheap
 * orbit_select() rescan, instead of rebuilding the whole BLA pyramid from
 * scratch on every single frame.                                          */

static Orbit *g_orbit     = NULL;
static double g_orbit_cx  = 0.0/0.0;
static double g_orbit_cy  = 0.0/0.0;
static int    g_orbit_iter = -1;

static Orbit *get_orbit(double cx, double cy, int mi, double max_delta) {
    if (!g_orbit || cx != g_orbit_cx || cy != g_orbit_cy || mi != g_orbit_iter) {
        orbit_free(g_orbit);
        g_orbit     = orbit_new(cx, cy, mi);
        g_orbit_cx  = cx;
        g_orbit_cy  = cy;
        g_orbit_iter = mi;
    }
    if (g_orbit) orbit_select(g_orbit, max_delta);
    return g_orbit;
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
    double sx, sy;
    view_geometry(&W, &h, &ph, &sx, &sy);
    ensure_bufs(W, h);

    double half_w = sx * W  * 0.5;
    double half_h = sy * ph * 0.5;
    double max_delta = sqrt(half_w*half_w + half_h*half_h);

    int mi = g_iter;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Orbit *orb = get_orbit(g_cx, g_cy, mi, max_delta);
    int sa_skip  = orb ? orb->sa_skip    : 0;
    int bla_levs = orb ? orb->bla_levels : 0;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < h; y++) {
        char *rb = g_rows[y];
        int   rp = 0;
        RGB   prev_fg = {0,0,0}, prev_bg = {0,0,0};
        int   have_fg = 0, have_bg = 0;

        double ci_top = (2*y)     * sy - half_h;
        double ci_bot = (2*y + 1) * sy - half_h;

        /* Per-thread scratch buffers, grown (never shrunk) and reused
         * across calls — render() can run thousands of times during a
         * long animation, so per-row malloc/free churn adds up fast.    */
        static __thread double *dcr = NULL, *dci_top = NULL, *dci_bot = NULL;
        static __thread int    *irow_top = NULL, *irow_bot = NULL;
        static __thread int     cap = 0;
        if (cap < W) {
            free(dcr); free(dci_top); free(dci_bot); free(irow_top); free(irow_bot);
            dcr      = malloc(W * sizeof(double));
            dci_top  = malloc(W * sizeof(double));
            dci_bot  = malloc(W * sizeof(double));
            irow_top = malloc(W * sizeof(int));
            irow_bot = malloc(W * sizeof(int));
            cap = (dcr && dci_top && dci_bot && irow_top && irow_bot) ? W : 0;
        }

        if (cap < W) {
            g_rlens[y] = 0;
            continue;
        }

        for (int x = 0; x < W; x++) {
            dcr[x]     = x * sx - half_w;
            dci_top[x] = ci_top;
            dci_bot[x] = ci_bot;
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
    }

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
        "  \033[2m[wasd]move [z/x]zoom [i/o]iter [k/l]color [p]png [t]color-mode [M]anim [q]quit\033[0m",
        h+1, g_cx, g_cy, g_zoom, g_iter,
        sa_skip, bla_levs, g_render_ms,
        g_truecolor ? "truecolor" : "16-color",
        g_msg[0] ? "  " : "", g_msg);

    (void)write(STDOUT_FILENO, g_frame, fp);
    g_resize = 0;
}

/* ─── PNG export ────────────────────────────────────────────────────────── */

/* Renders the current view at oversampled resolution (target ~1920px
 * wide, rather than just upscaling terminal cells) and writes it to
 * `fname` using zlib level `png_level` (lower = faster, bigger files —
 * see write_png()). Returns 0 on success; pw_out / ph_out / ms_out
 * receive the pixel dimensions used and the render time, for status
 * messages.                                                              */
static int render_view_to_png(const char *fname, int png_level,
                               int *pw_out, int *ph_out, double *ms_out) {
    int W, h, ph;
    double sx, sy;
    view_geometry(&W, &h, &ph, &sx, &sy);

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

    Orbit *orb = get_orbit(g_cx, g_cy, mi, max_delta);

    unsigned char *rgb = malloc((size_t)pw * phh * 3);
    if (!rgb) return -1;

    #pragma omp parallel for schedule(dynamic, 2)
    for (int y = 0; y < phh; y++) {
        double ci = y * psy - half_h;

        /* Per-thread scratch buffers, grown (never shrunk) and reused
         * across calls — an animation can call this hundreds of times,
         * each with up to ~2x the row count of the terminal view.       */
        static __thread double *dcr = NULL, *dci = NULL;
        static __thread int    *irow = NULL;
        static __thread int     cap = 0;
        if (cap < pw) {
            free(dcr); free(dci); free(irow);
            dcr  = malloc(pw * sizeof(double));
            dci  = malloc(pw * sizeof(double));
            irow = malloc(pw * sizeof(int));
            cap = (dcr && dci && irow) ? pw : 0;
        }
        if (cap < pw) continue;

        for (int x = 0; x < pw; x++) {
            dcr[x] = x * psx - half_w;
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
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);

    int ok = write_png(fname, pw, phh, rgb, png_level) == 0;
    free(rgb);

    *pw_out = pw; *ph_out = phh;
    *ms_out = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return ok ? 0 : -1;
}

static void save_png(void) {
    /* Sequence number guards against collisions when multiple exports
     * land within the same wall-clock second (timestamp alone isn't
     * unique enough since a render can finish in single-digit ms).    */
    static int seq = 0;
    time_t now = time(NULL);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", localtime(&now));
    char fname[64];
    snprintf(fname, sizeof fname, "mandelbrot_%s_%03d.png", stamp, seq++);

    int pw, ph;
    double ms;
    /* A deliberate single "keeper" save: use good compression (zlib 6). */
    int ok = render_view_to_png(fname, 6, &pw, &ph, &ms) == 0;

    if (ok)
        snprintf(g_msg, sizeof g_msg, "saved %s (%dx%d, %.0fms)", fname, pw, ph, ms);
    else
        snprintf(g_msg, sizeof g_msg, "png save failed: %s", fname);
}

/* ─── zoom-out animation ─────────────────────────────────────────────────
 * 'M' saves a numbered sequence of PNG frames zooming out from the
 * current zoom level down to 1.0 ("e0"), dividing g_zoom by a user-chosen
 * per-frame factor each step (e.g. 1.085 for a slow, smooth descent) —
 * meant to be concatenated into a video afterwards, e.g.:
 *   ffmpeg -framerate 30 -i mandelbrot_anim_<stamp>_%05d.png \
 *          -c:v libx264 -pix_fmt yuv420p out.mp4
 * A cheap terminal preview is also drawn each frame so progress is
 * visible live. Since the center (cx, cy) and iteration count stay fixed
 * throughout, every frame hits the orbit cache in get_orbit() — only the
 * cheap orbit_select() rescan runs per frame, not a full
 * reference-orbit/BLA rebuild.                                            */

static void draw_status_line(const char *text) {
    int W, H;
    term_size(&W, &H);
    char buf[600];
    int n = snprintf(buf, sizeof buf, "\033[%d;1H\033[K  %s", H, text);
    if (n > 0) (void)write(STDOUT_FILENO, buf, n);
}

/* Reads digits/'.' for a per-frame zoom-out factor (> 1.0). Returns 1
 * with *out set on Enter, 0 (no change) if cancelled with Esc.           */
static int prompt_zoom_factor(double *out) {
    char buf[32] = "";
    int  len = 0;
    for (;;) {
        char line[96];
        snprintf(line, sizeof line,
            "zoom-out factor per frame, e.g. 1.085 (Enter=go, Esc=cancel): %s", buf);
        draw_status_line(line);

        char c;
        ssize_t n = read(STDIN_FILENO, &c, 1);
        if (n <= 0) return 0;

        if (c == '\r' || c == '\n') {
            double f = atof(buf);
            if (f > 1.0) { *out = f; return 1; }
            continue; /* empty/invalid entry: keep prompting */
        }
        if (c == 27) return 0;
        if ((c == 127 || c == 8) && len > 0) { buf[--len] = '\0'; continue; }
        if ((c == '.' || (c >= '0' && c <= '9')) && len < (int)sizeof(buf) - 1) {
            buf[len++] = c; buf[len] = '\0';
        }
    }
}

/* Any keypress aborts the animation; 'q'/'Q' also requests a full quit. */
static void play_zoom_out(double factor, int *quit) {
    *quit = 0;
    int aborted = 0;

    char prefix[48];
    time_t now = time(NULL);
    char stamp[32];
    strftime(stamp, sizeof stamp, "%Y%m%d_%H%M%S", localtime(&now));
    snprintf(prefix, sizeof prefix, "mandelbrot_anim_%s", stamp);

    int frame = 0;
    for (;;) {
        frame++;
        char fname[80];
        snprintf(fname, sizeof fname, "%s_%05d.png", prefix, frame);

        int pw = 0, ph = 0;
        double ms = 0.0;
        /* Animation frames are disposable inputs to ffmpeg — favor fast
         * writes (zlib level 1) over file size.                         */
        int ok = render_view_to_png(fname, 1, &pw, &ph, &ms) == 0;

        if (ok)
            snprintf(g_msg, sizeof g_msg, "saving frame %d (zoom=%.4g, %dx%d, %.0fms): %s",
                     frame, g_zoom, pw, ph, ms, fname);
        else
            snprintf(g_msg, sizeof g_msg, "frame %d save FAILED (zoom=%.4g): %s",
                     frame, g_zoom, fname);
        render();

        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        if (poll(&pfd, 1, 0) > 0) {
            char c;
            if (read(STDIN_FILENO, &c, 1) == 1) {
                if (c == 'q' || c == 'Q') *quit = 1;
                aborted = 1;
                break;
            }
        }

        if (g_zoom <= 1.0) break;
        g_zoom /= factor;
        if (g_zoom < 1.0) g_zoom = 1.0;
    }

    if (*quit)
        snprintf(g_msg, sizeof g_msg, "saved %d frames (interrupted): %s_%%05d.png", frame, prefix);
    else if (aborted)
        snprintf(g_msg, sizeof g_msg, "saved %d frames (aborted): %s_%%05d.png", frame, prefix);
    else
        snprintf(g_msg, sizeof g_msg, "saved %d frames: %s_%%05d.png", frame, prefix);
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
            /* Per-pixel deltas are computed without ever round-tripping
             * through the absolute center (see view_geometry()), so
             * double precision holds up far past the old 1e14 cap that
             * was masking that cancellation bug. 1e17 leaves headroom
             * below where sx itself would underflow.                   */
            if (g_zoom < 1e17) g_zoom *= 1.5;
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

        case 'm': case 'M':
            if (g_zoom <= 1.0) {
                snprintf(g_msg, sizeof g_msg, "already at base zoom (1.0)");
            } else {
                double factor;
                if (prompt_zoom_factor(&factor)) {
                    int quit_req = 0;
                    play_zoom_out(factor, &quit_req);
                    if (quit_req) goto quit;
                } else {
                    g_msg[0] = '\0';
                }
            }
            break;

        default: dirty = 0; break;
        }

        if (need_iet) build_itable();
        if (dirty)    render();
    }
quit:
    return 0;
}
