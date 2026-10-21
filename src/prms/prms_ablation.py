"""PRMS scores, classes and weighting ablation for the 19 models audited in the journal article
("Beyond Benchmark Accuracy: Privacy, Deployment Readiness, and Model Scale in Human Action
Recognition for Assistive Systems"). Input: prms_models.csv. Evidence for every score:
PRMS_SCORING_EVIDENCE.md.

PRMS(i) = w_P * (P_i - 1)/2 + w_D * D_i/2 + w_T * T_i/2 + w_C * C_i/2
with P in {1,2,3}, D,T,C in {0,1,2}.

Weighting schemes (Table "PRMS Score Ablation"):
  privacy-centric (default): w = (0.70, 0.20, 0.04999, 0.04999)
  deployment-centric:        w = (0.20, 0.70, 0.05, 0.05)
  balanced:                  w = (0.25, 0.25, 0.25, 0.25)

Class intervals: A [0.90, 1.00], B [0.70, 0.90), C [0.10, 0.65), D [0, 0.10).

Usage:
  python prms_ablation.py           # print scores and classes, verify against the article
  python prms_ablation.py --plots   # also save the three ablation bar charts (Fig. 3; needs matplotlib)
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMES = {
    "privacy_centric": (0.70, 0.20, 0.04999, 0.04999),
    "deployment_centric": (0.20, 0.70, 0.05, 0.05),
    "balanced": (0.25, 0.25, 0.25, 0.25),
}


def prms(p, d, t, c, w=SCHEMES["privacy_centric"]):
    wp, wd, wt, wc = w
    return wp * (p - 1) / 2 + wd * d / 2 + wt * t / 2 + wc * c / 2


def prms_class(s):
    s = round(s, 6)
    if s >= 0.90:
        return "A"
    if s >= 0.70:
        return "B"
    if s >= 0.10:
        return "C"
    return "D"


def parse_amount(text):
    """'1.011B' -> 1.011e9, '89M' -> 8.9e7, '45-168' -> geometric mean, '<1.0' -> 1.0, '1000+' -> 1000."""
    text = text.strip().replace("$", "")
    m = re.match(r"^[<>]?\s*([\d.]+)(?:\s*-\s*([\d.]+))?\+?\s*([MB]?)$", text)
    if not m:
        return math.nan
    lo = float(m.group(1))
    val = math.sqrt(lo * float(m.group(2))) if m.group(2) else lo
    return val * {"M": 1e6, "B": 1e9, "": 1.0}[m.group(3)]


def load():
    with open(os.path.join(HERE, "prms_models.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in "PDTC":
            r[k] = int(r[k])
    return rows


def main():
    rows = load()
    print(f"{'Model':<18}{'P':>2}{'D':>3}{'T':>3}{'C':>3}{'PRMS':>8}{'Cls':>5}{'Deploy':>9}{'Balanced':>10}  check")
    mismatches = 0
    for r in rows:
        s = prms(r["P"], r["D"], r["T"], r["C"])
        cls = prms_class(s)
        sd = prms(r["P"], r["D"], r["T"], r["C"], SCHEMES["deployment_centric"])
        sb = prms(r["P"], r["D"], r["T"], r["C"], SCHEMES["balanced"])
        ok = abs(s - float(r["prms_reported"])) < 5e-4 and cls == r["class_reported"]
        mismatches += not ok
        print(f"{r['model']:<18}{r['P']:>2}{r['D']:>3}{r['T']:>3}{r['C']:>3}{s:>8.3f}{cls:>5}{sd:>9.3f}{sb:>10.3f}  {'ok' if ok else 'MISMATCH (reported ' + r['prms_reported'] + ' ' + r['class_reported'] + ')'}")

    counts = {k: sum(prms_class(prms(r['P'], r['D'], r['T'], r['C'])) == k for r in rows) for k in "ABCD"}
    print("\nClass distribution:", ", ".join(f"{k}: {v}/{len(rows)} ({100 * v / len(rows):.1f}%)" for k, v in counts.items()))
    print("Verification:", "all reported PRMS values and classes reproduced" if not mismatches else f"{mismatches} mismatches")

    if "--plots" in sys.argv:
        plots(rows)
    return 1 if mismatches else 0


def plots(rows):
    """Save the three ablation bar charts of the article (models ordered by privacy-centric score)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(rows, key=lambda r: -prms(r["P"], r["D"], r["T"], r["C"]))
    names = [r["model"] for r in order]
    titles = {
        "privacy_centric": "Privacy-centric ($w_P$=0.70, $w_D$=0.20, $w_T$=$w_C$=0.04999)",
        "deployment_centric": "Deployment-centric ($w_P$=0.20, $w_D$=0.70, $w_T$=$w_C$=0.05)",
        "balanced": "Balanced ($w_P$=$w_D$=$w_T$=$w_C$=0.25)",
    }
    files = {"privacy_centric": "PrivacyCentricPRMS.pdf", "deployment_centric": "DeploymentCentricPRMS.pdf",
             "balanced": "BalancedPRMS.pdf"}
    for k, w in SCHEMES.items():
        v = [prms(r["P"], r["D"], r["T"], r["C"], w) for r in order]
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        ax.bar(range(len(v)), v, color=[plt.cm.Blues(0.25 + 0.75 * x) for x in v], edgecolor="black", linewidth=0.5)
        for i, x in enumerate(v):
            ax.text(i, x + 0.015, f"{x:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("PRMS", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(titles[k], fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        out = os.path.join(HERE, files[k])
        fig.savefig(out)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
