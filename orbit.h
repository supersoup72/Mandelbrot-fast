/* orbit.h — perturbation theory types and interface */
#ifndef ORBIT_H
#define ORBIT_H

#include <mpfr.h>

typedef struct { double r, i; } cx_t;

/* ─── arbitrary-precision reference center ───────────────────────────────
 * A plain double can only pin down a coordinate to ~53 bits (~1e15-16 of
 * relative precision). Past that, distinct deep-zoom locations round to
 * the same double, the reference orbit itself drifts from the location
 * the user actually asked for, and rendering degrades into glitches/flat
 * blocks no per-pixel fallback can repair (the orbit shared by every
 * pixel is simply wrong). Storing the center as an MPFR float with
 * precision scaled to the current zoom depth removes that ceiling
 * entirely — there's no fixed bit budget to run out of.                   */

/* BLA entry: δ_{n+step} ≈ A*δ + B*Δc, valid when |δ|² < r2 */
typedef struct { cx_t A, B; double r2; } BlaEntry;

#define BLA_LEVELS 12

typedef struct {
    mpfr_t cx, cy;    /* reference point, at full target precision     */
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
 * (e.g. a pure zoom animation). Call orbit_select() before rendering.
 * cx/cy are read (mpfr_set, not stored by reference) at their current
 * precision — the caller is responsible for giving them enough precision
 * bits for the zoom depth in play (see bits_for_zoom() in mandelbrot.c)
 * before calling.                                                        */
Orbit *orbit_new(const mpfr_t cx, const mpfr_t cy, int max_iter);

/* Picks sa_skip (and sa_A/sa_B) for the current view's max_delta — a
 * cheap O(len) scan over the cached SA arrays, meant to be called every
 * frame even when the underlying orbit_new() result is being reused.   */
void   orbit_select(Orbit *o, double max_delta);
void   orbit_free(Orbit *o);

/* Direct (non-perturbed) Mandelbrot iteration at full MPFR precision —
 * for the rare per-pixel recompute (glitch detection, or a too-short
 * reference orbit) that needs to trust an absolute coordinate past where
 * perturbation from the reference orbit can be relied on. cr/ci must
 * already be initialized by the caller (mpfr_init2 at the desired
 * precision, typically matching the orbit's cx/cy) and hold the absolute
 * coordinate to test; they are read but not modified.                   */
int scalar_mandelbrot_mp(const mpfr_t cr, const mpfr_t ci, int max_iter);

/* Render one row: dcr[w], dci[w] are per-pixel Δc components.
 * out[w] receives iteration counts (max_iter = inside set).      */
void orbit_render_row(const Orbit *o, const double *dcr, const double *dci,
                      int w, int max_iter, int *out);

#endif /* ORBIT_H */
