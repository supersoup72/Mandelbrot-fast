/* orbit.h — perturbation theory types and interface */
#ifndef ORBIT_H
#define ORBIT_H

typedef struct { double r, i; } cx_t;

/* ─── double-double helper, for fallback paths needing an absolute coord ──
 * Adding a tiny per-pixel delta to an O(1) absolute coordinate in plain
 * double loses precision once the delta drops below the coordinate's own
 * ULP (~1e-16 relative) — exactly the regime deep zoom puts us in. That
 * loss doesn't shrink with zoom, so it quantizes nearby pixels onto the
 * same few representable doubles, which is what shows up as flat blocky
 * rectangles. two_sum recovers the *exact* mathematical sum of two
 * doubles as a (hi, lo) pair — no approximation, just split across two
 * doubles instead of rounding into one — giving ~106 bits of headroom for
 * the handful of pixels that actually need a direct (non-perturbed)
 * absolute-coordinate iteration.                                          */
typedef struct { double hi, lo; } dd_t;

/* -ffast-math's -fassociative-math will happily "simplify" this exact
 * error-recovery arithmetic back to lo=0 (algebraically true for reals,
 * false for IEEE doubles — which is the entire point), so it's compiled
 * with fast-math explicitly off regardless of the including TU's flags. */
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")
static inline dd_t dd_two_sum(double a, double b) {
    double s   = a + b;
    double bb  = s - a;
    double err = (a - (s - bb)) + (b - bb);
    return (dd_t){ s, err };
}
#pragma GCC pop_options

/* Direct (non-perturbed) Mandelbrot iteration at double-double precision —
 * the extended-precision counterpart to a plain scalar fallback, for the
 * rare per-pixel recompute that needs to trust an absolute coordinate far
 * past where a single double can represent it accurately.                */
int scalar_mandelbrot_dd(dd_t cr, dd_t ci, int max_iter);

/* BLA entry: δ_{n+step} ≈ A*δ + B*Δc, valid when |δ|² < r2 */
typedef struct { cx_t A, B; double r2; } BlaEntry;

#define BLA_LEVELS 12

typedef struct {
    double cx, cy;     /* reference point (absolute coordinates) */
    cx_t *Z;         /* reference orbit Z[0..len-1]          */
    cx_t *sA, *sB, *sC; /* SA coefficients (incl. cubic term) per iter */
    int   len;        /* orbit length (including escape point) */
    int   sa_skip;    /* SA-skippable iterations               */
    cx_t  sa_A, sa_B; /* A, B at sa_skip step                  */
    BlaEntry *bla[BLA_LEVELS];
    int bla_levels;
} Orbit;

/* Builds the reference orbit, SA coefficient arrays, and BLA pyramid —
 * all independent of the view's zoom level, so the result can be cached
 * and reused across frames whenever (cx, cy, max_iter) are unchanged
 * (e.g. a pure zoom animation). Call orbit_select() before rendering. */
Orbit *orbit_new(double cx, double cy, int max_iter);

/* Picks sa_skip (and sa_A/sa_B) for the current view's max_delta — a
 * cheap O(len) scan over the cached SA arrays, meant to be called every
 * frame even when the underlying orbit_new() result is being reused.   */
void   orbit_select(Orbit *o, double max_delta);
void   orbit_free(Orbit *o);

/* Render one row: dcr[w], dci[w] are per-pixel Δc components.
 * out[w] receives iteration counts (max_iter = inside set).      */
void orbit_render_row(const Orbit *o, const double *dcr, const double *dci,
                      int w, int max_iter, int *out);

#endif /* ORBIT_H */
