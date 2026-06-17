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
 * branch. Flagged pixels are recomputed directly (see scalar_mandelbrot). */
#define GLITCH_EPS2 1e-12   /* (1e-6)^2 */

/* ─── direct (non-perturbed) scalar Mandelbrot, for fallback/glitch fix ──── */

static int scalar_mandelbrot(double cr, double ci, int max_iter)
{
    double p = cr - 0.25;
    double q = p*p + ci*ci;
    if (q*(q+p) <= 0.25*ci*ci || (cr+1.0)*(cr+1.0)+ci*ci <= 0.0625)
        return max_iter;
    double zr=0, zi=0, zr2=0, zi2=0;
    int iter=0;
    while (zr2+zi2 < 4.0 && iter < max_iter) {
        zi  = 2.0*zr*zi + ci;
        zr  = zr2-zi2+cr;
        zr2 = zr*zr; zi2 = zi*zi;
        iter++;
    }
    return iter;
}

/* ─── reference orbit ────────────────────────────────────────────────────── */

Orbit *orbit_new(double cx, double cy, int max_iter, double max_delta)
{
    Orbit *o = calloc(1, sizeof *o);
    if (!o) return NULL;

    o->cx = cx;
    o->cy = cy;

    /* allocate with +2 spare */
    o->Z  = malloc((max_iter + 2) * sizeof *o->Z);
    o->sA = malloc((max_iter + 2) * sizeof *o->sA);
    o->sB = malloc((max_iter + 2) * sizeof *o->sB);
    if (!o->Z || !o->sA || !o->sB) { orbit_free(o); return NULL; }

    /* ── reference orbit ── */
    cx_t c  = { cx, cy };
    cx_t Zn = { 0.0, 0.0 };
    int  n;
    for (n = 0; n < max_iter; n++) {
        o->Z[n] = Zn;
        if (cx_abs2(Zn) > 4.0) break;
        /* Z_{n+1} = Z_n^2 + c */
        Zn = cx_add(cx_mul(Zn, Zn), c);
    }
    o->len = n; /* orbit has Z[0..n-1]; Z[n] would be escaped or == Z_{max} */

    /* ── SA coefficients ── */
    /* A_{n+1} = 2*Z_n*A_n + 1,  B_{n+1} = 2*Z_n*B_n + A_n^2 */
    cx_t An = { 0.0, 0.0 };
    cx_t Bn = { 0.0, 0.0 };
    cx_t Cn = { 0.0, 0.0 };
    o->sa_skip = 0;

    double md2 = max_delta * max_delta;

    for (int k = 0; k < o->len; k++) {
        o->sA[k] = An;
        o->sB[k] = Bn;

        cx_t two_Zk = cx_scale(o->Z[k], 2.0);

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
        double absB = sqrt(cx_abs2(Bn));
        double absC = sqrt(cx_abs2(Cn));
        double absA = sqrt(cx_abs2(An));
        if (absB * max_delta < 1e-3 * absA && absC * md2 < 1e-3 * absA) {
            o->sa_skip = k;
        }

        /* recurrence */
        cx_t An1 = cx_add(cx_mul(two_Zk, An), (cx_t){1.0, 0.0});
        cx_t Bn1 = cx_add(cx_mul(two_Zk, Bn), cx_mul(An, An));
        cx_t Cn1 = cx_add(cx_mul(two_Zk, Cn),
                          cx_scale(cx_mul(An, Bn), 2.0));
        An = An1;
        Bn = Bn1;
        Cn = Cn1;
    }
    /* store A,B at sa_skip */
    if (o->len > 0 && o->sa_skip < o->len) {
        o->sa_A = o->sA[o->sa_skip];
        o->sa_B = o->sB[o->sa_skip];
    } else {
        o->sa_A = (cx_t){0.0, 0.0};
        o->sa_B = (cx_t){0.0, 0.0};
        o->sa_skip = 0;
    }

    /* ── BLA table ── */
    /* Level 0: step = 1.  For position n:
     *   A = 2*Z[n],  B = {1,0},  r2 = (1e-6)^2 * |2*Z[n]|^2          */
    int levels = 0;
    if (o->len > 1) {
        o->bla[0] = malloc(o->len * sizeof *o->bla[0]);
        if (!o->bla[0]) { orbit_free(o); return NULL; }
        for (int k = 0; k < o->len; k++) {
            cx_t twoZ = cx_scale(o->Z[k], 2.0);
            double r2_twoZ = cx_abs2(twoZ);
            o->bla[0][k].A  = twoZ;
            o->bla[0][k].B  = (cx_t){1.0, 0.0};
            o->bla[0][k].r2 = 1e-12 * r2_twoZ;  /* (1e-6)^2 */
        }
        levels = 1;

        /* Level j from level j-1: step = 2^j.
         * Compose bla[j-1][n] with bla[j-1][n + 2^(j-1)].              */
        for (int j = 1; j < BLA_LEVELS; j++) {
            int half_step = 1 << (j - 1);
            int step      = 1 << j;
            if (step >= o->len) break;

            o->bla[j] = malloc(o->len * sizeof *o->bla[j]);
            if (!o->bla[j]) break;

            int valid_count = 0;
            for (int k = 0; k + step <= o->len; k++) {
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
                valid_count++;
            }
            if (valid_count == 0) {
                free(o->bla[j]);
                o->bla[j] = NULL;
                break;
            }
            levels = j + 1;
        }
    }
    o->bla_levels = levels;

    return o;
}

void orbit_free(Orbit *o)
{
    if (!o) return;
    free(o->Z);
    free(o->sA);
    free(o->sB);
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
        for (int x = 0; x < w; x++)
            out[x] = scalar_mandelbrot(o->cx + dcr[x], o->cy + dci[x], max_iter);
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
    for (int x = 0; x < w; x++) {
        double Dcr = dcr[x], Dci = dci[x];
        /* δ = A*Δc + B*Δc^2 */
        /* A*Δc */
        double Ar = sA.r*Dcr - sA.i*Dci;
        double Ai = sA.r*Dci + sA.i*Dcr;
        /* Δc^2 */
        double Dc2r = Dcr*Dcr - Dci*Dci;
        double Dc2i = 2.0*Dcr*Dci;
        /* B*Δc^2 */
        double Br = sB.r*Dc2r - sB.i*Dc2i;
        double Bi = sB.r*Dc2i + sB.i*Dc2r;
        dr[x] = Ar + Br;
        di[x] = Ai + Bi;
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
        int idx1 = (1 < o->len) ? 1 : (o->len < max_iter ? o->len : o->len - 1);
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
        int idx = (sa_n < o->len) ? sa_n
                : (o->len < max_iter ? o->len : o->len - 1);
        double Znr = o->Z[idx].r, Zni = o->Z[idx].i;
        double Zmag2 = Znr*Znr + Zni*Zni;
        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            double Wr = Znr + dr[x];
            double Wi = Zni + di[x];
            double W2 = Wr*Wr + Wi*Wi;
            if (W2 > 4.0) {
                iter_count[x] = sa_n;
            } else if (W2 < GLITCH_EPS2 * Zmag2) {
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

        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            double old_dr = dr[x], old_di = di[x];
            double Dcr_x = dcr[x],  Dci_x = dci[x];
            /* δ_new = A*δ + B*Δc */
            dr[x] = Ar*old_dr - Ai*old_di + Br*Dcr_x - Bi*Dci_x;
            di[x] = Ar*old_di + Ai*old_dr + Br*Dci_x + Bi*Dcr_x;
        }
        n += step;

        /* After applying BLA step, check escape at new position n.
         * Use Z[n] if available. If n lands exactly at the end of the
         * recorded orbit and the reference itself escaped (o->len <
         * max_iter), Z[o->len] still holds that valid escaped value —
         * use it instead of stale Z[len-1], which corrupts W = Z+δ and
         * causes widespread wrong escape iterations near the orbit's
         * own escape point (the single most common perturbation glitch). */
        int Zn_idx = (n < o->len) ? n
                   : (o->len < max_iter ? o->len : o->len - 1);
        double Znr = o->Z[Zn_idx].r, Zni = o->Z[Zn_idx].i;
        double Zmag2 = Znr*Znr + Zni*Zni;
        for (int x = 0; x < w; x++) {
            if (iter_count[x] >= 0) continue;
            double Wr = Znr + dr[x];
            double Wi = Zni + di[x];
            double W2 = Wr*Wr + Wi*Wi;
            if (W2 > 4.0) {
                iter_count[x] = n;
            } else if (W2 < GLITCH_EPS2 * Zmag2) {
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
            __m256d dr2  = _mm256_sub_pd(_mm256_mul_pd(vdr, vdr),
                                          _mm256_mul_pd(vdi, vdi));
            __m256d di2  = _mm256_mul_pd(_mm256_mul_pd(two, vdr), vdi);

            /* new δ = 2*Z_n*δ + δ^2 + Δc */
            /* real: tZr*dr - tZi*di + dr2 + Dcr */
            __m256d new_dr = _mm256_add_pd(
                               _mm256_add_pd(
                                 _mm256_sub_pd(_mm256_mul_pd(tZr, vdr),
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
             * the recorded orbit's end and the reference escaped early
             * (o->len < max_iter), Z[o->len] is still the valid escaped
             * value and must be used instead of stale Z[len-1].         */
            int next_idx = (nn + 1 < o->len) ? nn + 1
                          : (o->len < max_iter ? o->len : o->len - 1);
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
             * cancellation) AND did not escape this step                       */
            double  Zmag2_next = Zr1*Zr1 + Zi1*Zi1;
            __m256d gthresh    = _mm256_set1_pd(GLITCH_EPS2 * Zmag2_next);
            __m256d raw_glitch = _mm256_cmp_pd(W2, gthresh, _CMP_LT_OQ);
            __m256d glitch_mask = _mm256_andnot_pd(esc_mask, raw_glitch);
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
            int ni = (nn+1 < o->len) ? nn+1
                   : (o->len < max_iter ? o->len : o->len-1);
            double Zr1 = o->Z[ni].r, Zi1 = o->Z[ni].i;
            double Wr = Zr1 + pdr;
            double Wi = Zi1 + pdi;
            double W2 = Wr*Wr + Wi*Wi;
            if (W2 > 4.0) { nn++; break; }
            double Zmag2 = Zr1*Zr1 + Zi1*Zi1;
            if (W2 < GLITCH_EPS2 * Zmag2) { glitch = 1; break; }
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

    /* Copy results to output, recomputing glitched pixels directly */
    for (int xx = 0; xx < w; xx++) {
        if (glitched[xx]) {
            out[xx] = scalar_mandelbrot(o->cx + dcr[xx], o->cy + dci[xx], max_iter);
        } else {
            out[xx] = (iter_count[xx] < 0) ? max_iter : iter_count[xx];
        }
    }

    free(dr);
    free(di);
    free(iter_count);
    free(glitched);
}
