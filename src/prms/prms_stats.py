"""Rank statistics reported in the journal article (Section IV).

Reproduces, from prms_models.csv:
  1. Kendall's tau_b between the four PRMS dimensions (P, D, T, C) over the 19 models,
     with two-sided permutation p-values (Table "Kendall's tau_b Between the PRMS Dimensions").
  2. Kendall's tau_b between log10(#params) and PRMS for the models whose papers report a
     parameter count, with a permutation p-value.
  3. Kendall's tau_b between the PRMS rankings under the three weighting schemes (ablation).
  4. AHP weights, consistency ratio and lexicographic check for the pairwise judgements of
     Section III (P vs D = 3, D vs S = 3, P vs S = 9) and two alternatives.

tau_b accounts for ties, which are frequent with ordinal scores. p-values come from
permutation tests (20,000 random permutations of one variable, fixed seed), which do not rely
on large-sample approximations. Only the Python standard library is required.

Usage:
  python prms_stats.py
"""
import itertools
import math
import random

from prms_ablation import SCHEMES, load, prms

N_PERM = 20000
SEED = 1


def kendall_tau_b(x, y):
    concordant = discordant = ties_x = ties_y = 0
    for i, j in itertools.combinations(range(len(x)), 2):
        dx, dy = x[i] - x[j], y[i] - y[j]
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            continue
        if abs(dx) < 1e-12:
            ties_x += 1
        elif abs(dy) < 1e-12:
            ties_y += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else float("nan")


def permutation_p(x, y, n_perm=N_PERM, seed=SEED):
    """Two-sided permutation p-value for tau_b (add-one estimator)."""
    observed = abs(kendall_tau_b(x, y))
    rng = random.Random(seed)
    y = list(y)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(kendall_tau_b(x, y)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def parse_params(text):
    """'28.3M' -> 2.83e7, '1011M' -> 1.011e9, '6B' -> 6e9; storage sizes (MB) and n/r -> None."""
    text = text.strip()
    if "MB" in text or text == "n/r":
        return None
    if text.endswith("M"):
        return float(text[:-1]) * 1e6
    if text.endswith("B"):
        return float(text[:-1]) * 1e9
    return None


def ahp(a_pd, a_ps, a_ds, iters=500):
    """Principal eigenvector (power iteration) and consistency ratio of a 3x3 AHP matrix (RI = 0.58)."""
    A = [[1, a_pd, a_ps], [1 / a_pd, 1, a_ds], [1 / a_ps, 1 / a_ds, 1]]
    w = [1 / 3] * 3
    for _ in range(iters):
        v = [sum(A[i][j] * w[j] for j in range(3)) for i in range(3)]
        total = sum(v)
        w = [x / total for x in v]
    lam = sum(sum(A[i][j] * w[j] for j in range(3)) / w[i] for i in range(3)) / 3
    cr = ((lam - 3) / 2) / 0.58
    return w, max(cr, 0.0)


def main():
    rows = load()
    dims = {k: [r[k] for r in rows] for k in "PDTC"}

    print(f"1. Kendall tau_b between dimensions (N={len(rows)}, {N_PERM} permutations)")
    for a, b in itertools.combinations("PDTC", 2):
        t = kendall_tau_b(dims[a], dims[b])
        p = permutation_p(dims[a], dims[b])
        print(f"   {a}-{b}: tau_b = {t:+.2f}  p = {p:.3f}")

    sized = [(r, parse_params(r["params"])) for r in rows]
    sized = [(r, n) for r, n in sized if n]
    x = [math.log10(n) for _, n in sized]
    y = [prms(r["P"], r["D"], r["T"], r["C"]) for r, _ in sized]
    print(f"\n2. log10(#params) vs PRMS (N={len(sized)}: {', '.join(r['model'] for r, _ in sized)})")
    print(f"   tau_b = {kendall_tau_b(x, y):+.2f}  p = {permutation_p(x, y):.3f}")

    scores = {k: [prms(r["P"], r["D"], r["T"], r["C"], w) for r in rows] for k, w in SCHEMES.items()}
    print("\n3. Rank agreement between weighting schemes (tau_b)")
    for other in ("deployment_centric", "balanced"):
        print(f"   privacy_centric vs {other}: {kendall_tau_b(scores['privacy_centric'], scores[other]):.2f}")

    print("\n4. AHP (judgements a_PD, a_PS, a_DS)")
    base = scores["privacy_centric"]
    for judg in ((3, 9, 3), (3, 7, 2), (4, 9, 3)):
        (wp, wd, ws), cr = ahp(*judg)
        w = (wp, wd, ws / 2, ws / 2)
        lex = wp / 2 > wd + ws and wd / 2 > ws
        tau = kendall_tau_b(base, [prms(r["P"], r["D"], r["T"], r["C"], w) for r in rows])
        print(f"   {judg}: w = ({wp:.3f}, {wd:.3f}, {ws:.3f})  CR = {cr:.3f}  "
              f"lexicographic = {lex}  tau_b vs reported ranking = {tau:.2f}")


if __name__ == "__main__":
    main()
