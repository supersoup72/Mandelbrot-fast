/* orbit.h — perturbation theory types and interface */
#ifndef ORBIT_H
#define ORBIT_H

typedef struct { double r, i; } cx_t;

/* BLA entry: δ_{n+step} ≈ A*δ + B*Δc, valid when |δ|² < r2 */
typedef struct { cx_t A, B; double r2; } BlaEntry;

#define BLA_LEVELS 12

typedef struct {
    cx_t *Z;         /* reference orbit Z[0..len-1]          */
    cx_t *sA, *sB;   /* SA coefficients at each iteration     */
    int   len;        /* orbit length (including escape point) */
    int   sa_skip;    /* SA-skippable iterations               */
    cx_t  sa_A, sa_B; /* A, B at sa_skip step                  */
    BlaEntry *bla[BLA_LEVELS];
    int bla_levels;
} Orbit;

Orbit *orbit_new(double cx, double cy, int max_iter, double max_delta);
void   orbit_free(Orbit *o);

/* Render one row: dcr[w], dci[w] are per-pixel Δc components.
 * out[w] receives iteration counts (max_iter = inside set).      */
void orbit_render_row(const Orbit *o, const double *dcr, const double *dci,
                      int w, int max_iter, int *out);

#endif /* ORBIT_H */
