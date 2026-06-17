/* orbit.c — perturbation theory, SA, BLA, AVX2 render row */
#define _GNU_SOURCE
#include "orbit.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>   /* AVX2 */

/* ─── helpers ────────────────────────────────────────────────────────────── */

static inline double cx_abs2(cx_t z) { return z.r*z.r + z.i*z.i; }

static inline cx_t cx_mul(cx_t a, cx_t b) {
    return (cx_t){ a.r*b.r - a.i*b.i, a.r*b.i + a.i*b.r };
}

static inline cx_t cx_add(cx_t a, cx_t b) {
    return (cx_t){ a.r+b.r, a.i+b.i };
}

static inline cx_t cx_scale(cx_t a, double s) {
    return (cx_t){ a.r*s, a.i*s };
}

/* ─── glitch detection (Pauldelbrot criterion) ──────────────────────────────
 * A perturbed point W = Z_n + δ_n that lands anomalously close to zero
 * relative to the reference's own scale signals catastrophic cancellation:
 * δ_n has grown large enough that the linear perturbation model is no
 * longer trustworthy and the pixel orbit may have drifted onto the wrong
 * branch. Flagged pixels are recomputed directly (see scalar_mandelbrot_mp). */
#define GLITCH_EPS2 1e-12   /* (1e-6)^2 */
#define DELTA_GROWTH_FRAC 1e-2  /* δ grown to >1% of |Z_n|^2: perturbation's
                                 * small-delta accuracy assumption has broken
                                 * down, regardless of whether W passed near
                                 * zero -- force the MPFR fallback instead of
                                 * trusting a now-meaningless double δ.      */

/* Below this many elements, OpenMP's thread spawn/join overhead exceeds
 * the work being parallelized — measured break-even is ~20k elements for
 * the simple per-entry BLA composition below, so smaller loops are left
 * to run serially via the `if()` clause rather than forking threads.    */
#define OMP_PAR_THRESHOLD 20000

/* ─── MPFR Mandelbrot, for glitch/short-orbit fallback at any zoom depth ──
 * The fallback paths below need to trust an absolute coordinate built
 * from the reference center plus a per-pixel delta. At deep zoom the
 * center itself needs more than 53 bits to even specify the location, so
 * this iterates entirely in MPFR at whatever precision the orbit's own
 * center was built with — there's no fixed bit budget to run out of, only
 * a (rare, per-glitched-pixel) performance cost.                         */
int scalar_mandelbrot_mp(const mpfr_t cr, const mpfr_t ci, int max_iter)
{
    /* Cardioid/bulb quick-reject only needs O(1) precision, so a plain
     * double snapshot of cr/ci is enough for this test.                 */
    double crd = mpfr_get_d(cr, MPFR_RNDN);
    double cid = mpfr_get_d(ci, MPFR_RNDN);
    double p = crd - 0.25;
    double q = p*p + cid*cid;
    if (q*(q+p) <= 0.25*cid*cid || (crd+1.0)*(crd+1.0)+cid*cid <= 0.0625)
        return max_iter;

    mpfr_prec_t prec = mpfr_get_prec(cr);
    mpfr_t zr, zi, zr2, zi2, zrzi, tmp;
    mpfr_inits2(prec, zr, zi, zr2, zi2, zrzi, tmp, (mpfr_ptr)0);
    mpfr_set_zero(zr, 1);
    mpfr_set_zero(zi, 1);

    int iter = 0;
    while (iter < max_iter) {
        mpfr_sqr(zr2, zr, MPFR_RNDN);
        mpfr_sqr(zi2, zi, MPFR_RNDN);
        mpfr_add(tmp, zr2, zi2, MPFR_RNDN);
        /* Escape test only needs double precision: once |z| actually
         * exceeds 2, it diverges within a couple more steps regardless
         * of any correction far below a double's ULP.                  */
        if (mpfr_get_d(tmp, MPFR_RNDN) >= 4.0) break;

        mpfr_mul(zrzi, zr, zi, MPFR_RNDN);
        mpfr_mul_2ui(zrzi, zrzi, 1, MPFR_RNDN);   /* 2*zr*zi */
        mpfr_add(zi, zrzi, ci, MPFR_RNDN);        /* new zi */

        mpfr_sub(tmp, zr2, zi2, MPFR_RNDN);
        mpfr_add(zr, tmp, cr, MPFR_RNDN);         /* new zr */

        iter++;
    }

    mpfr_clears(zr, zi, zr2, zi2, zrzi, tmp, (mpfr_ptr)0);
    return iter;
}

/* ─── reference orbit ────────────────────────────────────────────────────── */

Orbit *orbit_new(const mpfr_t cx, const mpfr_t cy, int max_iter)
{
    Orbit *o = calloc(1, sizeof *o);
    if (!o) return NULL;

    mpfr_prec_t prec = mpfr_get_prec(cx);
    if (mpfr_get_prec(cy) > prec) prec = mpfr_get_prec(cy);
    if (prec < 53) prec = 53;

    mpfr_init2(o->cx, prec);
    mpfr_init2(o->cy, prec);
    mpfr_set(o->cx, cx, MPFR_RNDN);
    mpfr_set(o->cy, cy, MPFR_RNDN);

    /* allocate with +2 spare */
    o->Z  = malloc((max_iter + 2) * sizeof *o->Z);
    o->sA = malloc((max_iter + 2) * sizeof *o->sA);
    o->sB = malloc((max_iter + 2) * sizeof *o->sB);
    o->sC = malloc((max_iter + 2) * sizeof *o->sC);
    if (!o->Z || !o->sA || !o->sB || !o->sC) { orbit_free(o); return NULL; }

    /* ── reference orbit, iterated at full MPFR precision ──
     * The chaotic map amplifies any rounding error introduced at step k
     * by roughly the local derivative (|2*Z_k|) at every later step, so
     * the *iteration itself* — not just the input c — has to be carried
     * at the precision the zoom depth demands. Only each step's already-
     * computed result is well-conditioned enough to downcast to a plain
     * double for the (unchanged) double-precision SA/BLA/perturbation
     * math below — that's the whole point of perturbation theory: pay
     * for precision once here, not on every pixel.                      */
    mpfr_t zr, zi, zr2, zi2, zrzi, tmp;
    mpfr_inits2(prec, zr, zi, zr2, zi2, zrzi, tmp, (mpfr_ptr)0);
    mpfr_set_zero(zr, 1);
    mpfr_set_zero(zi, 1);

    int n;
    for (n = 0; n < max_iter; n++) {
        o->Z[n].r = mpfr_get_d(zr, MPFR_RNDN);
        o->Z[n].i = mpfr_get_d(zi, MPFR_RNDN);
        if (cx_abs2(o->Z[n]) > 4.0) break;

        /* Z_{n+1} = Z_n^2 + c */
        mpfr_sqr(zr2, zr, MPFR_RNDN);
        mpfr_sqr(zi2, zi, MPFR_RNDN);
        mpfr_mul(zrzi, zr, zi, MPFR_RNDN);
        mpfr_mul_2ui(zrzi, zrzi, 1, MPFR_RNDN);
        mpfr_add(zi, zrzi, cy, MPFR_RNDN);
        mpfr_sub(tmp, zr2, zi2, MPFR_RNDN);
        mpfr_add(zr, tmp, cx, MPFR_RNDN);
    }
    o->len = n; /* orbit has Z[0..n-1], plus Z[n] below: escaped, or the
                 * value after exactly max_iter iterations.              */

    /* If the loop broke out early (escape), Z[n] was already stored by
     * the body above before the break. If it ran to completion instead,
     * Z[max_iter] was never stored — the loop body's store happens at
     * the *start* of each iteration, so the value computed by the final
     * iteration (now sitting in zr/zi) was never written back. Store it
     * here so every "next reference point" lookup below has real data
     * even at the very end of the orbit, instead of silently reusing
     * the previous iteration's (wrong) value.                          */
    if (n == max_iter) {
        o->Z[n].r = mpfr_get_d(zr, MPFR_RNDN);
        o->Z[n].i = mpfr_get_d(zi, MPFR_RNDN);
    }

    mpfr_clears(zr, zi, zr2, zi2, zrzi, tmp, (mpfr_ptr)0);

    /* ── SA coefficients ── */
    /* A_{n+1} = 2*Z_n*A_n + 1,  B_{n+1} = 2*Z_n*B_n + A_n^2
     * Stored at every k (cheap, independent of zoom level) so that
     * orbit_select() can later pick sa_skip for any max_delta without
     * recomputing this recurrence.                                    */
    cx_t An = { 0.0, 0.0 };
    cx_t Bn = { 0.0, 0.0 };
    cx_t Cn = { 0.0, 0.0 };

    for (int k = 0; k < o->len; k++) {
        o->sA[k] = An;
        o->sB[k] = Bn;
        o->sC[k] = Cn;

        cx_t two_Zk = cx_scale(o->Z[k], 2.0);

        /* recurrence */
        cx_t An1 = cx_add(cx_mul(two_Zk, An), (cx_t){1.0, 0.0});
        cx_t Bn1 = cx_add(cx_mul(two_Zk, Bn), cx_mul(An, An));
        cx_t Cn1 = cx_add(cx_mul(two_Zk, Cn),
                          cx_scale(cx_mul(An, Bn), 2.0));
        An = An1;
        Bn = Bn1;
        Cn = Cn1;
    }
    o->sa_skip = 0;
    o->sa_A = (cx_t){0.0, 0.0};
    o->sa_B = (cx_t){0.0, 0.0};

    /* ── BLA table ── */
    /* Level 0: step = 1.  For position n:
     *   A = 2*Z[n],  B = {1,0},  r2 = (1e-6)^2 * |2*Z[n]|^2          */
    int levels = 0;
    if (o->len > 1) {
        o->bla[0] = malloc(o->len * sizeof *o->bla[0]);
        if (!o->bla[0]) { orbit_free(o); return NULL; }
        #pragma omp parallel for schedule(static) if(o->len > OMP_PAR_THRESHOLD)
        for (int k = 0; k < o->len; k++) {
            cx_t twoZ = cx_scale(o->Z[k], 2.0);
            double r2_twoZ = cx_abs2(twoZ);
            o->bla[0][k].A  = twoZ;
            o->bla[0][k].B  = (cx_t){1.0, 0.0};
            o->bla[0][k].r2 = 1e-12 * r2_twoZ;  /* (1e-6)^2 */
        }
        levels = 1;

        /* Level j from level j-1: step = 2^j.
         * Compose bla[j-1][n] with bla[j-1][n + 2^(j-1)]. Each entry only
         * reads the fully-built level j-1 array, so the k-loop within a
         * fixed j is embarrassingly parallel.                            */
        for (int j = 1; j < BLA_LEVELS; j++) {
            int half_step = 1 << (j - 1);
            int step      = 1 << j;
            if (step >= o->len) break;

            o->bla[j] = malloc(o->len * sizeof *o->bla[j]);
            if (!o->bla[j]) break;

            /* trip count is len - step + 1, always >= 2 here since the
             * step >= o->len check above already ruled out an empty range */
            int trip = o->len - step + 1;
            #pragma omp parallel for schedule(static) if(trip > OMP_PAR_THRESHOLD)
            for (int k = 0; k < trip; k++) {
                BlaEntry *e1 = &o->bla[j-1][k];
                BlaEntry *e2 = &o->bla[j-1][k + half_step];
                /* compose: δ_1 = A1*δ + B1*Δc
                 *           δ_2 = A2*δ_1 + B2*Δc = A2*A1*δ + (A2*B1+B2)*Δc */
                cx_t A = cx_mul(e2->A, e1->A);
                cx_t B = cx_add(cx_mul(e2->A, e1->B), e2->B);
                /* validity: min(r2_1, r2_2 / |A1|^2) */
                double absA1_2 = cx_abs2(e1->A);
                double r2;
                if (absA1_2 < 1e-300)
                    r2 = e1->r2;
                else
                    r2 = e2->r2 / absA1_2;
                if (r2 > e1->r2) r2 = e1->r2;

                o->bla[j][k].A  = A;
                o->bla[j][k].B  = B;
                o->bla[j][k].r2 = r2;
            }
            levels = j + 1;
        }
    }
    o->bla_levels = levels;

    return o;
}

void orbit_select(Orbit *o, double max_delta)
{
    if (!o || o->len == 0) return;

    /* Validity requires the truncated quadratic AND cubic terms to be
     * negligible next to the retained linear term:
     *   |B_k| * max_delta     < tol * |A_k|   (quadratic term tiny)
     *   |C_k| * max_delta^2   < tol * |A_k|   (cubic term tiny)
     * Checking only the cubic term (as a single coefficient ratio) is
     * not sufficient: C_k can be coincidentally ~0 early on (e.g. an
     * artifact of Z_0=0) while B_k*Δc is still comparable to A_k*Δc,
     * meaning the series has NOT actually converged — this produced
     * widespread wrong iteration counts at low zoom (large Δc) where
     * the 2-term truncation is simply invalid.                       */
    double md2 = max_delta * max_delta;
    int sa_skip = 0;
    for (int k = 0; k < o->len; k++) {
        double absA = sqrt(cx_abs2(o->sA[k]));
        double absB = sqrt(cx_abs2(o->sB[k]));
        double absC = sqrt(cx_abs2(o->sC[k]));
        if (absB * max_delta < 1e-3 * absA && absC * md2 < 1e-3 * absA)
            sa_skip = k;
    }
    o->sa_skip = sa_skip;
    o->sa_A    = o->sA[sa_skip];
    o->sa_B    = o->sB[sa_skip];
}

void orbit_free(Orbit *o)
{
    if (!o) return;
    mpfr_clear(o->cx);
    mpfr_clear(o->cy);
    free(o->Z);
    free(o->sA);
    free(o->sB);
    free(o->sC);
    for (int j = 0; j < BLA_LEVELS; j++) free(o->bla[j]);
    free(o);
}

/* (scalar fallback is inlined in orbit_render_row remainder loop) */

/* ─── orbit_render_row ───────────────────────────────────────────────────── */

void orbit_render_row(const Orbit *o, const double *dcr, const double *dci,
                      int w, int max_iter, int *out)
{
    /* If reference orbit is very short (reference escapes almost
     * immediately), perturbation has nothing useful to skip — go direct. */
    if (o->len < 2) {
        mpfr_prec_t prec = mpfr_get_prec(o->cx);
        mpfr_t cr, ci;
        mpfr_init2(cr, prec);
        mpfr_init2(ci, prec);
        for (int x = 0; x < w; x++) {
            mpfr_add_d(cr, o->cx, dcr[x], MPFR_RNDN);
            mpfr_add_d(ci, o->cy, dci[x], MPFR_RNDN);
            out[x] = scalar_mandelbrot_mp(cr, ci, max_iter);
        }
        mpfr_clear(cr);
        mpfr_clear(ci);
        return;
    }

    /* ── Phase 1: SA skip ── */
    /* Each pixel starts at n = sa_skip with initial δ = A*Δc + B*Δc^2 */
    int sa_n = o->sa_skip;

    /* Per-pixel delta arrays */
    double *dr = malloc(w * sizeof(double));
    double *di = malloc(w * sizeof(double));
    if (!dr || !di) { free(dr); free(di); return; }

    cx_t sA = o->sa_A;
    cx_t sB = o->sa_B;
    {
        __m256d vAr = _mm256_set1_pd(sA.r), vAi = _mm256_set1_pd(sA.i);
        __m256d vBr = _mm256_set1_pd(sB.r), vBi = _mm256_set1_pd(sB.i);
        __m256d two = _mm256_set1_pd(2.0);
        int xv = 0;
        for (; xv + 4 <= w; xv += 4) {
            __m256d Dcr = _mm256_loadu_pd(&dcr[xv]);
            __m256d Dci = _mm256_loadu_pd(&dci[xv]);

            /* Δc^2 */
            __m256d Dc2r = _mm256_fmsub_pd(Dcr, Dcr, _mm256_mul_pd(Dci, Dci));
            __m256d Dc2i = _mm256_mul_pd(two, _mm256_mul_pd(Dcr, Dci));

            /* δ = A*Δc + B*Δc^2 */
            __m256d accR = _mm256_fmsub_pd(vBr, Dc2r, _mm256_mul_pd(vBi, Dc2i));
            accR = _mm256_fmadd_pd(vAr, Dcr, accR);
            accR = _mm256_fnmadd_pd(vAi, Dci, accR);

            __m256d accI = _mm256_fmadd_pd(vBr, Dc2i, _mm256_mul_pd(vBi, Dc2r));
            accI = _mm256_fmadd_pd(vAr, Dci, accI);
            accI = _mm256_fmadd_pd(vAi, Dcr, accI);

            _mm256_storeu_pd(&dr[xv], accR);
            _mm256_storeu_pd(&di[xv], accI);
        }
        for (; xv < w; xv++) {
            double Dcr = dcr[xv], Dci = dci[xv];
            /* δ = A*Δc + B*Δc^2 */
            double Ar = sA.r*Dcr - sA.i*Dci;
            double Ai = sA.r*Dci + sA.i*Dcr;
            double Dc2r = Dcr*Dcr - Dci*Dci;
            double Dc2i = 2.0*Dcr*Dci;
            double Br = sB.r*Dc2r - sB.i*Dc2i;
            double Bi = sB.r*Dc2i + sB.i*Dc2r;
            dr[xv] = Ar + Br;
            di[xv] = Ai + Bi;
        }
    }

    /* ── Phase 2: Global BLA loop ── */
    /* Find largest BLA level valid for ALL pixels, apply it to ALL pixels.
     * This keeps the orbit position n synchronized across the row.       */
    int n = sa_n;
    /* iter_count[x] < 0 means pixel is still active (not yet escaped) */
    int  *iter_count = malloc(w * sizeof(int));
    unsigned char *glitched = calloc(w, 1);
    if (!iter_count || !glitched) {
        free(dr); free(di); free(iter_count); free(glitched);
        return;
    }
    for (int x = 0; x < w; x++) iter_count[x] = -1; /* sentinel = not done */

    /* The SA seed jumps straight to n = sa_n without validating any
     * intermediate iteration in between — the validity criterion only
     * bounds polynomial truncation error, not whether an escape happens
     * somewhere in [1, sa_n). δ_1 = Δc exactly (Z_0 = 0), so checking n=1
     * is free (no polynomial evaluation) and catches the single most
     * common "escapes immediately" case (|c| > 2) cheaply.               */
    if (sa_n > 1) {
        int idx1 = (1 < o->len) ? 1 : o->len;
        double Znr = o->Z[idx1].r, Zni = o->Z[idx1].i;
        for (int x = 0; x < w; x++) {
            double Wr = Znr + dcr[x];
            double Wi = Zni + dci[x];
            if (Wr*Wr + Wi*Wi > 4.0) iter_count[x] = 1;
        }
    }

    /* A pixel can also already be escaped (or glitched) exactly AT the
     * seeded point n = sa_n — if that isn't checked here, the loops below
     * would iterate one step further before noticing, reporting an
     * escape iteration one too high.                                     */
    {
        int idx = (sa_n < o->len) ? sa_n : o->len;
        double Znr = o->Z[idx].r, Zni = o->Z[idx].i;
        double Zmag2 = Znr*Znr + Zni*Zni;
        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            double Wr = Znr + dr[x];
            double Wi = Zni + di[x];
            double W2 = Wr*Wr + Wi*Wi;
            double d2 = dr[x]*dr[x] + di[x]*di[x];
            if (W2 > 4.0) {
                iter_count[x] = sa_n;
            } else if (W2 < GLITCH_EPS2 * Zmag2 || d2 > DELTA_GROWTH_FRAC * Zmag2) {
                glitched[x]   = 1;
                iter_count[x] = max_iter; /* placeholder; overridden later */
            }
        }
    }

    /* Compute max |δ|² across active pixels for BLA validity check */
    while (n < o->len && o->bla_levels > 0) {
        /* find best BLA level where all active pixels are valid */
        double max_d2 = 0.0;
        int active = 0;
        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            active++;
            double d2 = dr[x]*dr[x] + di[x]*di[x];
            if (d2 > max_d2) max_d2 = d2;
        }
        if (active == 0) break;

        /* Pick largest level j where bla[j][n].r2 > max_d2 and n+step <= len */
        int best_j = -1;
        for (int j = o->bla_levels - 1; j >= 0; j--) {
            if (!o->bla[j]) continue;
            int step = 1 << j;
            if (n + step > o->len) continue;
            if (o->bla[j][n].r2 > max_d2) {
                best_j = j;
                break;
            }
        }
        if (best_j < 0) break; /* no BLA step valid; exit to per-pixel AVX2 */

        BlaEntry *be   = &o->bla[best_j][n];
        int       step = 1 << best_j;
        double    Ar = be->A.r, Ai = be->A.i;
        double    Br = be->B.r, Bi = be->B.i;

        /* δ_new = A*δ + B*Δc, applied SIMD-wide to the whole row (à la
         * fraktaler-3's batched BLA application) rather than branching
         * per pixel on iter_count. Already-resolved lanes get garbage
         * values here, but nothing downstream ever reads dr/di for a
         * pixel once iter_count[x] >= 0, so this is safe and lets the
         * compiler/AVX2 process 4 lanes per step unconditionally.       */
        __m256d vAr = _mm256_set1_pd(Ar), vAi = _mm256_set1_pd(Ai);
        __m256d vBr = _mm256_set1_pd(Br), vBi = _mm256_set1_pd(Bi);
        int xx = 0;
        for (; xx + 4 <= w; xx += 4) {
            __m256d old_dr = _mm256_loadu_pd(&dr[xx]);
            __m256d old_di = _mm256_loadu_pd(&di[xx]);
            __m256d Dcr_v  = _mm256_loadu_pd(&dcr[xx]);
            __m256d Dci_v  = _mm256_loadu_pd(&dci[xx]);

            /* δ_new = A*δ + B*Δc, fused into a chain of FMAs */
            __m256d new_dr = _mm256_fmsub_pd(vBr, Dcr_v, _mm256_mul_pd(vBi, Dci_v));
            new_dr = _mm256_fmadd_pd(vAr, old_dr, new_dr);
            new_dr = _mm256_fnmadd_pd(vAi, old_di, new_dr);

            __m256d new_di = _mm256_fmadd_pd(vBr, Dci_v, _mm256_mul_pd(vBi, Dcr_v));
            new_di = _mm256_fmadd_pd(vAr, old_di, new_di);
            new_di = _mm256_fmadd_pd(vAi, old_dr, new_di);

            _mm256_storeu_pd(&dr[xx], new_dr);
            _mm256_storeu_pd(&di[xx], new_di);
        }
        for (; xx < w; xx++) {
            double old_dr = dr[xx], old_di = di[xx];
            double Dcr_x = dcr[xx],  Dci_x = dci[xx];
            dr[xx] = Ar*old_dr - Ai*old_di + Br*Dcr_x - Bi*Dci_x;
            di[xx] = Ar*old_di + Ai*old_dr + Br*Dci_x + Bi*Dcr_x;
        }
        n += step;

        /* After applying BLA step, check escape at new position n.
         * Use Z[n] if available. If n lands exactly at the end of the
         * recorded orbit, Z[o->len] still holds a valid value — either
         * the reference's escaped point, or (if the reference ran the
         * full max_iter without escaping) the value after exactly
         * max_iter iterations, stored by orbit_new() for this purpose.
         * Using anything else here corrupts W = Z+δ and causes wrong
         * escape iterations near the orbit's own endpoint (the single
         * most common perturbation glitch).                            */
        int Zn_idx = (n < o->len) ? n : o->len;
        double Znr = o->Z[Zn_idx].r, Zni = o->Z[Zn_idx].i;
        double Zmag2 = Znr*Znr + Zni*Zni;
        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            double Wr = Znr + dr[x];
            double Wi = Zni + di[x];
            double W2 = Wr*Wr + Wi*Wi;
            double d2 = dr[x]*dr[x] + di[x]*di[x];
            if (W2 > 4.0) {
                iter_count[x] = n;
            } else if (W2 < GLITCH_EPS2 * Zmag2 || d2 > DELTA_GROWTH_FRAC * Zmag2) {
                glitched[x]   = 1;
                iter_count[x] = max_iter;  /* placeholder; overridden later */
            }
        }
    }

    /* ── Phase 3: AVX2 4-wide perturbation loop ── */
    /* Process groups of 4 pixels from position n to orbit end */
    int x = 0;
    for (; x + 4 <= w; x += 4) {
        /* Check if all 4 are already done */
        if (iter_count[x]   >= 0 && iter_count[x+1] >= 0 &&
            iter_count[x+2] >= 0 && iter_count[x+3] >= 0)
            continue;

        /* Load current deltas */
        __m256d vdr = _mm256_set_pd(dr[x+3], dr[x+2], dr[x+1], dr[x]);
        __m256d vdi = _mm256_set_pd(di[x+3], di[x+2], di[x+1], di[x]);
        __m256d vDcr= _mm256_set_pd(dcr[x+3],dcr[x+2],dcr[x+1],dcr[x]);
        __m256d vDci= _mm256_set_pd(dci[x+3],dci[x+2],dci[x+1],dci[x]);

        /* Active mask: lane is active if iter_count < 0 */
        /* We track iteration count with integer counters */
        int cnt[4];
        int active_mask[4];
        for (int k = 0; k < 4; k++) {
            cnt[k]         = (iter_count[x+k] >= 0) ? iter_count[x+k] : n;
            active_mask[k] = (iter_count[x+k] < 0) ? 1 : 0;
        }

        /* Freeze already-escaped lanes' deltas */
        /* (they won't be written back, so this is fine) */

        /* AVX2 iteration loop */
        /* active lanes as a bitmask for early exit */
        int alive = active_mask[0] | (active_mask[1]<<1) |
                    (active_mask[2]<<2) | (active_mask[3]<<3);

        /* We need per-lane iteration counters; simplest: use int array */
        __m256d four = _mm256_set1_pd(4.0);

        for (int nn = n; nn < o->len && nn < max_iter && alive; nn++) {
            double Znr = o->Z[nn].r, Zni = o->Z[nn].i;
            __m256d vZr = _mm256_set1_pd(Znr);
            __m256d vZi = _mm256_set1_pd(Zni);

            /* 2*Z_n*δ:  real = 2*(Zr*dr - Zi*di),  imag = 2*(Zr*di + Zi*dr) */
            __m256d two  = _mm256_set1_pd(2.0);
            __m256d tZr  = _mm256_mul_pd(two, vZr);
            __m256d tZi  = _mm256_mul_pd(two, vZi);

            /* δ^2 */
            __m256d dr2  = _mm256_fmsub_pd(vdr, vdr, _mm256_mul_pd(vdi, vdi));
            __m256d di2  = _mm256_mul_pd(_mm256_mul_pd(two, vdr), vdi);

            /* new δ = 2*Z_n*δ + δ^2 + Δc */
            /* real: tZr*dr - tZi*di + dr2 + Dcr */
            __m256d new_dr = _mm256_add_pd(
                               _mm256_add_pd(
                                 _mm256_fmsub_pd(tZr, vdr,
                                               _mm256_mul_pd(tZi, vdi)),
                                 dr2),
                               vDcr);
            /* imag: tZr*di + tZi*dr + di2 + Dci  (use FMA) */
            __m256d new_di = _mm256_add_pd(
                               _mm256_add_pd(
                                 _mm256_fmadd_pd(tZi, vdr,
                                   _mm256_mul_pd(tZr, vdi)),
                                 di2),
                               vDci);

            /* Escape check: |Z_{nn+1} + δ_{nn+1}|² > 4
             * Z_{nn+1} is Z[nn+1] if available; if nn+1 lands exactly at
             * the recorded orbit's end, Z[o->len] is still valid (the
             * escaped value, or the value after exactly max_iter
             * iterations — see orbit_new()) and must be used instead of
             * stale Z[len-1].                                           */
            int next_idx = (nn + 1 < o->len) ? nn + 1 : o->len;
            double Zr1 = o->Z[next_idx].r, Zi1 = o->Z[next_idx].i;
            __m256d vZr1 = _mm256_set1_pd(Zr1);
            __m256d vZi1 = _mm256_set1_pd(Zi1);
            __m256d Wr   = _mm256_add_pd(vZr1, new_dr);
            __m256d Wi   = _mm256_add_pd(vZi1, new_di);
            __m256d W2   = _mm256_add_pd(_mm256_mul_pd(Wr, Wr),
                                          _mm256_mul_pd(Wi, Wi));
            /* escape_mask: lane is escaped if W2 > 4 AND was active */
            __m256d esc_mask = _mm256_cmp_pd(W2, four, _CMP_GT_OQ);
            int     esc_bits = _mm256_movemask_pd(esc_mask);

            /* glitch_mask: W anomalously small vs reference scale (catastrophic
             * cancellation), OR δ itself has grown to a non-negligible
             * fraction of |Z_n| (the small-perturbation assumption that
             * perturbation theory's accuracy relies on has broken down) --
             * AND did not escape this step                                    */
            double  Zmag2_next = Zr1*Zr1 + Zi1*Zi1;
            __m256d gthresh    = _mm256_set1_pd(GLITCH_EPS2 * Zmag2_next);
            __m256d raw_glitch = _mm256_cmp_pd(W2, gthresh, _CMP_LT_OQ);
            __m256d d2v        = _mm256_add_pd(_mm256_mul_pd(new_dr, new_dr),
                                                _mm256_mul_pd(new_di, new_di));
            __m256d gdthresh   = _mm256_set1_pd(DELTA_GROWTH_FRAC * Zmag2_next);
            __m256d raw_dgrow  = _mm256_cmp_pd(d2v, gdthresh, _CMP_GT_OQ);
            __m256d glitch_mask = _mm256_andnot_pd(esc_mask,
                                       _mm256_or_pd(raw_glitch, raw_dgrow));
            int     glitch_bits = _mm256_movemask_pd(glitch_mask);

            /* Freeze escaped lanes: keep old delta */
            new_dr = _mm256_blendv_pd(new_dr, vdr, esc_mask);
            new_di = _mm256_blendv_pd(new_di, vdi, esc_mask);
            vdr = new_dr;
            vdi = new_di;

            /* Record escape iteration for newly escaped lanes */
            for (int k = 0; k < 4; k++) {
                if (!active_mask[k]) continue;
                if (esc_bits & (1 << k)) {
                    cnt[k]         = nn + 1;
                    active_mask[k] = 0;
                } else if (glitch_bits & (1 << k)) {
                    glitched[x+k]  = 1;
                    cnt[k]         = max_iter; /* placeholder; overridden later */
                    active_mask[k] = 0;
                }
            }
            alive = active_mask[0] | (active_mask[1]<<1) |
                    (active_mask[2]<<2) | (active_mask[3]<<3);
        }

        /* Write results */
        for (int k = 0; k < 4; k++) {
            if (iter_count[x+k] < 0) {
                if (!active_mask[k]) {
                    iter_count[x+k] = cnt[k];
                } else if (o->len >= max_iter) {
                    /* Reference orbit ran the full max_iter without
                     * escaping: a still-active pixel is genuinely inside. */
                    iter_count[x+k] = max_iter;
                } else {
                    /* Reference orbit escaped before max_iter and ran out
                     * of data while this pixel was still unresolved — its
                     * true fate is unknown to perturbation; verify directly
                     * rather than assuming it's inside the set.            */
                    glitched[x+k]   = 1;
                    iter_count[x+k] = max_iter; /* placeholder; overridden later */
                }
            }
        }
    }

    /* ── Handle remainder (< 4 pixels) with scalar perturbation ── */
    for (; x < w; x++) {
        if (iter_count[x] >= 0) continue;  /* already set by BLA phase */

        double pdr = dr[x], pdi = di[x];
        int nn;
        int glitch = 0;
        for (nn = n; nn < o->len && nn < max_iter; nn++) {
            double Znr = o->Z[nn].r, Zni = o->Z[nn].i;
            double dr2  = pdr*pdr - pdi*pdi;
            double di2  = 2.0*pdr*pdi;
            double ndr  = 2.0*(Znr*pdr - Zni*pdi) + dr2 + dcr[x];
            double ndi  = 2.0*(Znr*pdi + Zni*pdr) + di2 + dci[x];
            pdr = ndr; pdi = ndi;
            int ni = (nn+1 < o->len) ? nn+1 : o->len;
            double Zr1 = o->Z[ni].r, Zi1 = o->Z[ni].i;
            double Wr = Zr1 + pdr;
            double Wi = Zi1 + pdi;
            double W2 = Wr*Wr + Wi*Wi;
            if (W2 > 4.0) { nn++; break; }
            double Zmag2 = Zr1*Zr1 + Zi1*Zi1;
            double d2 = pdr*pdr + pdi*pdi;
            if (W2 < GLITCH_EPS2 * Zmag2 || d2 > DELTA_GROWTH_FRAC * Zmag2) { glitch = 1; break; }
        }
        if (glitch) {
            glitched[x]   = 1;
            iter_count[x] = max_iter; /* placeholder; overridden later */
        } else if (nn >= max_iter) {
            /* nn can only reach max_iter if the reference orbit covered
             * the full range (o->len >= max_iter), so the pixel is
             * genuinely inside the set.                                */
            iter_count[x] = max_iter;
        } else {
            /* Loop stopped because the reference orbit ran out of data
             * (nn == o->len) before this pixel resolved — its true fate
             * is unknown to perturbation; verify directly.             */
            glitched[x]   = 1;
            iter_count[x] = max_iter; /* placeholder; overridden later */
        }
    }

    /* Copy results to output, recomputing glitched pixels directly.
     * The MPFR scratch pair is only set up if this row actually has a
     * glitched pixel — the common case has none, and mpfr_init2 isn't
     * free.                                                            */
    {
        int any_glitched = 0;
        for (int xx = 0; xx < w; xx++) if (glitched[xx]) { any_glitched = 1; break; }

        mpfr_t cr, ci;
        if (any_glitched) {
            mpfr_prec_t prec = mpfr_get_prec(o->cx);
            mpfr_init2(cr, prec);
            mpfr_init2(ci, prec);
        }
        for (int xx = 0; xx < w; xx++) {
            if (glitched[xx]) {
                mpfr_add_d(cr, o->cx, dcr[xx], MPFR_RNDN);
                mpfr_add_d(ci, o->cy, dci[xx], MPFR_RNDN);
                out[xx] = scalar_mandelbrot_mp(cr, ci, max_iter);
            } else {
                out[xx] = (iter_count[xx] < 0) ? max_iter : iter_count[xx];
            }
        }
        if (any_glitched) { mpfr_clear(cr); mpfr_clear(ci); }
    }

    free(dr);
    free(di);
    free(iter_count);
    free(glitched);
}
