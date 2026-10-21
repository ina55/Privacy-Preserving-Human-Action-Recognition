"""Monte Carlo sensitivity of PRMS to the exact weight values (journal article, Section IV).

Draws weight vectors with w_P ~ U(0.65, 0.75) and w_D ~ U(0.15, 0.25), splits the remainder
equally between T and C, and keeps only draws that satisfy both lexicographic conditions
(w_P/2 > w_D + w_T + w_C and w_D/2 > w_T + w_C). For the accepted draws it reports:
  - Kendall's tau_b between the perturbed and the reported PRMS ranking,
  - the share of draws in which all 19 class labels are unchanged with the fixed class
    intervals of the article (A >= 0.90, B >= 0.70, C >= 0.10),
  - which models change label when they do.
With --plot it also saves the figure MonteCarloPRMS.pdf (needs numpy and matplotlib).

Usage:
  python prms_montecarlo.py [--plot]
"""
import os
import random
import sys
from collections import Counter

from prms_ablation import HERE, load, prms, prms_class
from prms_stats import kendall_tau_b

N_ACCEPTED = 10000
SEED = 2026
BASE = (0.70, 0.20, 0.04999, 0.04999)


def lexicographic(wp, wd, wt, wc):
    return wp / 2 > wd + wt + wc and wd / 2 > wt + wc


def draw(rng):
    while True:
        wp = rng.uniform(0.65, 0.75)
        wd = rng.uniform(0.15, 0.25)
        rest = 1.0 - wp - wd
        w = (wp, wd, rest / 2, rest / 2)
        if rest > 0 and lexicographic(*w):
            return w


def main():
    rows = load()
    names = [r["model"] for r in rows]
    base = [prms(r["P"], r["D"], r["T"], r["C"], BASE) for r in rows]
    base_cls = [prms_class(s) for s in base]

    rng = random.Random(SEED)
    weights, scores, unchanged, flips = [], [], [], Counter()
    taus = []
    for _ in range(N_ACCEPTED):
        w = draw(rng)
        s = [prms(r["P"], r["D"], r["T"], r["C"], w) for r in rows]
        cls = [prms_class(x) for x in s]
        same = cls == base_cls
        weights.append(w)
        scores.append(s)
        unchanged.append(same)
        taus.append(kendall_tau_b(base, s))
        if not same:
            for n, a, b in zip(names, base_cls, cls):
                if a != b:
                    flips[f"{n}: {a}->{b}"] += 1

    print(f"accepted draws: {N_ACCEPTED} (seed {SEED})")
    print(f"tau_b vs reported ranking: min {min(taus):.3f}, identical ranking in "
          f"{100 * sum(t > 0.9999 for t in taus) / N_ACCEPTED:.1f}% of draws")
    print(f"all class labels unchanged (fixed intervals): {100 * sum(unchanged) / N_ACCEPTED:.1f}% of draws")
    for k, v in flips.most_common():
        print(f"   {k}: {100 * v / N_ACCEPTED:.1f}% of draws")

    if "--plot" in sys.argv:
        plot(names, base, base_cls, weights, scores, unchanged)


def plot(names, base, base_cls, weights, scores, unchanged):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    S = np.array(scores)
    order = np.argsort(-np.array(base), kind="stable")
    colors = {"A": "#2e7d32", "B": "#1f5fa8", "C": "#e08a00", "D": "#c62828"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.7, 1]})

    # (a) score range per model across draws
    for x, i in enumerate(order):
        lo, q1, med, q3, hi = np.percentile(S[:, i], [0, 25, 50, 75, 100])
        c = colors[base_cls[i]]
        ax1.vlines(x, lo, hi, color=c, lw=1.0)
        ax1.vlines(x, q1, q3, color=c, lw=4.0)
        ax1.plot(x, base[i], marker="_", color="black", ms=7, mew=1.2)
    for t in (0.10, 0.65, 0.70, 0.90):
        ax1.axhline(t, color="gray", lw=0.6, ls="--")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([names[i] for i in order], rotation=65, ha="right", fontsize=6)
    ax1.set_ylabel("PRMS", fontsize=8)
    ax1.set_ylim(-0.02, 1.0)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.set_title("(a) PRMS range over 10,000 weight draws", fontsize=8)
    for k, c in colors.items():
        ax1.plot([], [], color=c, lw=4, label=f"Class {k}")
    ax1.legend(fontsize=6, loc="upper right", frameon=False)
    ax1.spines[["top", "right"]].set_visible(False)

    # (b) weight space
    W = np.array(weights)
    ok = np.array(unchanged)
    from matplotlib.lines import Line2D
    ax2.scatter(W[ok, 0], W[ok, 1], s=2, color="#9ccc9c", rasterized=True)
    ax2.scatter(W[~ok, 0], W[~ok, 1], s=2, color="#e57373", rasterized=True)
    ax2.plot(0.70, 0.20, marker="*", color="black", ms=9, ls="none")
    ax2.plot(0.692, 0.231, marker="D", color="black", ms=4, mfc="white", ls="none")
    handles = [Line2D([], [], marker="o", ls="none", color="#9ccc9c", ms=4, label="all labels unchanged"),
               Line2D([], [], marker="o", ls="none", color="#e57373", ms=4, label="some label changes"),
               Line2D([], [], marker="*", ls="none", color="black", ms=7, label="weights used"),
               Line2D([], [], marker="D", ls="none", color="black", mfc="white", ms=4, label="AHP weights")]
    for t in (0.70,):
        ax2.axvline(t, color="gray", lw=0.6, ls="--")
    ax2.set_xlabel("$w_P$", fontsize=8)
    ax2.set_ylabel("$w_D$", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_title("(b) Accepted weight draws", fontsize=8)
    ax2.legend(handles=handles, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(HERE, "MonteCarloPRMS.pdf")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
