/* orbit.h — perturbation theory types and interface */
#ifndef ORBIT_H
#define ORBIT_H

typedef struct { double r, i; } cx_t;

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
