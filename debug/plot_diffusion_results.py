#!/usr/bin/env python
"""Generate the diffusion-reduction figures from debug/run_mach_ladder.jl output.

Reads:  $DIFFUSION_OUTDIR/ladder_metrics.csv, proj_ma{Ma}_o{1,2}.txt
Writes: $FIGDIR/fig_density_ma100.png, fig_methods_summary.png   (default FIGDIR=debug)

Kinetic timestep trace: parsed from $KINETIC_LOG (a run_kinetic_vs_hll.jl verbose
log, split on 'Starting time evolution') if set; otherwise the recorded canonical
sequence is used so the figure always builds.

Run with a matplotlib-enabled python, e.g.:
  PYTHONNOUSERSITE=1 DIFFUSION_OUTDIR=debug/reprodata python debug/plot_diffusion_results.py
"""
import os, re, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.environ.get("DIFFUSION_OUTDIR", os.path.join(os.path.dirname(__file__), "reprodata"))
FIG = os.environ.get("FIGDIR", os.path.dirname(__file__))
PMA = int(os.environ.get("PEAK_MA", "100"))   # which Ma to show in the density figure

d = np.genfromtxt(os.path.join(OUT, "ladder_metrics.csv"), delimiter=",", names=True)
Ma = np.unique(d["Ma"])
col = lambda o, n: np.array([d[n][(d["Ma"] == m) & (d["order"] == o)][0] for m in Ma])
peak1, peak2 = col(1, "peak_rho"), col(2, "peak_rho")
grad1, grad2 = col(1, "maxgrad"), col(2, "maxgrad")
s1 = np.loadtxt(os.path.join(OUT, f"proj_ma{PMA}_o1.txt"))
s2 = np.loadtxt(os.path.join(OUT, f"proj_ma{PMA}_o2.txt"))
pk1, gr1 = peak1[Ma == PMA][0], grad1[Ma == PMA][0]
pk2, gr2 = peak2[Ma == PMA][0], grad2[Ma == PMA][0]

# --- kinetic vs hll dt traces ---
def dt_from_log(path):
    txt = open(path).read()
    blocks = txt.split("Starting time evolution")
    seqs = []
    for b in blocks[1:]:
        steps, dts = [], []
        for m in re.finditer(r"Step\s+(\d+):.*?dt = ([0-9.eE+-]+|NaN)", b):
            steps.append(int(m.group(1)))
            dts.append(float("nan") if m.group(2) == "NaN" else float(m.group(2)))
        if steps: seqs.append((np.array(steps), np.array(dts)))
    return seqs
klog = os.environ.get("KINETIC_LOG")
if klog and os.path.exists(klog) and len(dt_from_log(klog)) >= 2:
    (hll_s, hll_dt), (kin_s, kin_dt) = dt_from_log(klog)[:2]
else:  # recorded canonical sequence (Ma=10, 24^3)
    hll_s = np.arange(1, 8); hll_dt = np.array([1.69e-3, 5.84e-4, 4.56e-4, 5.15e-4, 4.97e-4, 5.27e-4, 6.08e-4])
    kin_s = np.arange(1, 7); kin_dt = np.array([1.689e-3, 6.39e-10, 2.93e-13, 1.11e-13, 5.07e-31, np.nan])

# ===== Figure 1: density (peak structure, max along z) =====
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
vmax = max(s1.max(), s2.max())
for ax, sl, ttl, pk, gr in ((axs[0], s1, "First-order (diffusive)", pk1, gr1),
                            (axs[1], s2, "High-order reconstruction (our fix)", pk2, gr2)):
    im = ax.imshow(sl.T, origin="lower", cmap="inferno", vmin=0, vmax=vmax, extent=[0, 1, 0, 1])
    ax.set_title(f"{ttl}\npeak $\\rho$={pk:.2f},  max|$\\nabla\\rho$|={gr:.0f}", fontsize=11)
    ax.set_xlabel("x"); ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
axs[0].set_ylabel("y")
fig.colorbar(im, ax=axs, label="density $\\rho$", shrink=0.85)
fig.suptitle(f"Ma={PMA} crossing jets, peak density (max along z) — same problem, same grid",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_density_ma100.png"), dpi=130); plt.close(fig)
print("wrote fig_density_ma100.png")

# ===== Figure 2: composite summary =====
fig, axs = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
ax = axs[0, 0]
ax.plot(Ma, peak1, "o--", color="tab:gray", label="first-order")
ax.plot(Ma, peak2, "o-", color="tab:red", label="high-order (recon)")
for m, a, b in zip(Ma, peak1, peak2):
    ax.annotate(f"+{100*(b-a)/a:.0f}%", (m, b), textcoords="offset points", xytext=(4, 4), fontsize=8, color="tab:red")
ax.set_xlabel("Mach number"); ax.set_ylabel("peak density retained")
ax.set_title("Sharpness: peak density (higher = less diffusion)"); ax.legend(); ax.grid(alpha=.3)
ax = axs[0, 1]
ax.plot(Ma, grad1, "o--", color="tab:gray", label="first-order")
ax.plot(Ma, grad2, "o-", color="tab:red", label="high-order (recon)")
for m, a, b in zip(Ma, grad1, grad2):
    ax.annotate(f"{b/a:.1f}x", (m, b), textcoords="offset points", xytext=(4, 4), fontsize=8, color="tab:red")
ax.set_xlabel("Mach number"); ax.set_ylabel("max |$\\nabla\\rho$|")
ax.set_title("Sharpness: max density gradient (higher = sharper fronts)"); ax.legend(); ax.grid(alpha=.3)
ax = axs[1, 0]
ax.semilogy(hll_s, hll_dt, "o-", color="tab:green", ms=4, label="HLL high-order: stable")
ax.semilogy(kin_s, kin_dt, "s-", color="tab:purple", ms=6, label="kinetic flux: $\\Delta t\\to$ NaN")
ax.set_xlabel("time step"); ax.set_ylabel("$\\Delta t$")
ax.set_title("Robustness: kinetic flux is unstable (Ma=10 jets)")
ax.set_xlim(0, 20); ax.legend(fontsize=9); ax.grid(alpha=.3, which="both")
ax = axs[1, 1]; ax.axis("off")
rows = [["Method", "Diffusion", "Robust @Ma=100", "Verdict"],
        ["First-order", "high (smears)", "yes", "baseline"],
        ["High-order recon\n(+ realizability gate)", "LOW", "YES", "WIN"],
        ["HLLC flux", "= HLL", "(fallback)", "no gain"],
        ["HLLEM flux", "= HLL", "(inert proj.)", "no gain"],
        ["Kinetic flux\n(in-house nodes)", "--", "NO (NaN)", "unstable"]]
t = ax.table(cellText=rows, loc="center", cellLoc="center")
t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 2.0)
for j in range(4):
    t[0, j].set_facecolor("#333"); t[0, j].set_text_props(color="w", fontweight="bold")
    t[2, j].set_facecolor("#d7f5d7")
t[2, 3].set_text_props(fontweight="bold", color="tab:red")
for i in (4, 5):
    for j in range(4): t[i, j].set_facecolor("#f5e0e0")
ax.set_title("Methods tried (this effort)", fontweight="bold")
fig.suptitle("35-moment HyQMOM crossing jets: reducing numerical diffusion at high Mach",
             fontsize=13, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig_methods_summary.png"), dpi=130); plt.close(fig)
print("wrote fig_methods_summary.png")
