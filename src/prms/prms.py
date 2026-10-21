"""PRMS score (Privacy-Conscious Readiness and Modality Semantics).

Implements Eq. (3) of:
  D. Chiroiu-Jitianu, M. Nan, I. G. Mocanu,
  "Ethical and Practical Trade-offs in Human Action Recognition Models for Assistive Systems", SYNASC 2025.

    PRMS(i) = 0.70 * (P_i - 1) / 2 + 0.20 * D_i / 2 + 0.09999 * S_i / 4,   S_i = T_i + C_i

with P in {1, 2, 3} (privacy level), D in {0, 1, 2} (deployment readiness),
T in {0, 1, 2} (temporal modeling) and C in {0, 1, 2} (contextual awareness).

Class intervals: A [0.9, 1], B [0.7, 0.9), C [0.1, 0.65), D [0, 0.1).

Running this file reproduces Table II of the paper.
"""


def prms(p: int, d: int, t: int, c: int) -> float:
    return 0.70 * (p - 1) / 2 + 0.20 * d / 2 + 0.09999 * (t + c) / 4


def prms_class(score: float) -> str:
    if score >= 0.9:
        return "A"
    if score >= 0.7:
        return "B"
    if score >= 0.1:
        return "C"
    return "D"


# Table II, SYNASC 2025: (model, P, D, T, C)
TABLE_II = [
    ("VideoMAE V2", 1, 0, 2, 0),
    ("InternVideo2", 1, 0, 2, 2),
    ("Video-LLaVA", 1, 1, 1, 2),
    ("Learnable-Cube", 3, 0, 1, 0),
    ("PO-GUISE", 3, 2, 1, 1),
]


if __name__ == "__main__":
    print(f"{'Model':<16}{'P':>3}{'D':>3}{'T':>3}{'C':>3}{'PRMS':>9}  Class")
    for name, p, d, t, c in TABLE_II:
        s = prms(p, d, t, c)
        print(f"{name:<16}{p:>3}{d:>3}{t:>3}{c:>3}{s:>9.3f}  {prms_class(s)}")
