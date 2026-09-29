"""FICTITIOUS BOOK DATA: the 20 quarterly restaurant cost residuals of Table 4.2.
Hellwig and Jarque-Bera normality tests,
Applied Regression Analysis for Business, sections 4.3.5-4.3.6."""
import numpy as np
from scipy import stats

e = np.array([-178.78, 9.80, 50.33, 36.99, 0.75, -187.76, 48.92, -47.09, -41.41, 244.71,
              25.48, 250.72, 96.62, -320.80, -41.51, -216.89, -240.18, 74.38, 150.58, 285.14])
n = len(e)
print(f"n = {n}, mean = {e.mean():.4f}, sum e^2 = {np.sum(e**2):,.2f}")

# ---------- Hellwig ----------
S_book_formula = np.sqrt(np.mean((e - e.mean())**2))   # 1/n, as Table 4.3 writes it
S_n1 = e.std(ddof=1)                                   # 1/(n-1)
print(f"\nS with 1/n = {S_book_formula:.2f}   S with 1/(n-1) = {S_n1:.2f}   (book uses 166.88)")

def hellwig_empty_cells(res, ddof):
    z = (np.sort(res) - res.mean()) / res.std(ddof=ddof)
    F = stats.norm.cdf(z)
    cells = np.minimum((F * len(res)).astype(int), len(res) - 1)
    counts = np.bincount(cells, minlength=len(res))
    return z, F, counts, int((counts == 0).sum())

z, F, counts, ke = hellwig_empty_cells(e, ddof=1)
print("sorted  z      F(z)")
for s, zz, ff in zip(np.sort(e), z, F):
    print(f"{s:8.2f} {zz:6.3f} {ff:6.3f}")
print("cell counts:", counts.tolist())
print(f"empty cells k_e = {ke}  (book: 7)")
print(f"k_e with 1/n std = {hellwig_empty_cells(e, ddof=0)[3]}")

# critical values by simulation (book's Appendix A4 gives k1=4, k2=9 at alpha=0.05)
rng = np.random.default_rng(0)
sim = np.array([hellwig_empty_cells(rng.standard_normal(n), 1)[3] for _ in range(100_000)])
print(f"simulated k_e under H0: mean {sim.mean():.2f}, "
      f"P(k<=3)={np.mean(sim<=3):.3f}, P(k>=10)={np.mean(sim>=10):.3f}, "
      f"P(4<=k<=9)={np.mean((sim>=4)&(sim<=9)):.3f}")

# ---------- Jarque-Bera ----------
S = np.sqrt(np.mean(e**2))
M3, M4 = np.mean(e**3), np.mean(e**4)
A = M3 / S**3; B1 = A**2; B2 = M4 / S**4
JB = n * (B1/6 + (B2 - 3)**2 / 24)
crit = stats.chi2.ppf(0.95, 2)
print(f"\nS={S:.2f}  M3={M3:,.2f}  M4={M4:,.2f}")
print(f"A (skew)={A:.4f}  B1={B1:.4f}  B2 (kurtosis)={B2:.4f}")
print(f"JB = {JB:.4f}   critical chi2(2, 0.05) = {crit:.2f}   p = {stats.chi2.sf(JB, 2):.3f}")
print("scipy.stats.jarque_bera:", stats.jarque_bera(e))
print("scipy.stats.shapiro (better for n=20):", stats.shapiro(e))
