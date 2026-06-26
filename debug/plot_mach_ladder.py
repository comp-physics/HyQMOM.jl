import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCR = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(SCR, "slices")
mas = [10, 25, 50, 100]
tags = [("fo", "first-order"), ("lim", "scaling-limiter"), ("projrec", "projection-triggered")]

def load(ma, tag):
    f = os.path.join(SL, f"sl_ma{ma}_{tag}.txt")
    return np.loadtxt(f) if os.path.isfile(f) else None

fig, axes = plt.subplots(len(mas), len(tags), figsize=(11, 14), constrained_layout=True)
for r, ma in enumerate(mas):
    slices = {t: load(ma, t) for t, _ in tags}
    vmax = max(np.nanmax(s) for s in slices.values() if s is not None)
    for c, (t, label) in enumerate(tags):
        ax = axes[r, c]
        s = slices[t]
        if s is None:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center"); ax.axis("off"); continue
        im = ax.imshow(s.T, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{label}\npeak rho = {s.max():.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(f"Ma = {ma}", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=axes[r, :].tolist(), shrink=0.85, label="density (max over z)")

fig.suptitle("3D crossing jets, Np=64, matched time — density (max-z projection), shared color scale per row\n"
             "first-order HLL  vs  scaling-limiter  vs  projection-triggered (all on HLL flux, vacfloor=0)",
             fontsize=12)
out = os.path.join(SCR, "mach_ladder_density.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)

# Also a compact sharpness summary bar chart (peak density vs Ma per control)
peaks = {t: [ (load(ma,t).max() if load(ma,t) is not None else np.nan) for ma in mas ] for t,_ in tags}
fig2, ax2 = plt.subplots(figsize=(7,4.5), constrained_layout=True)
x = np.arange(len(mas)); w = 0.26
for i,(t,label) in enumerate(tags):
    ax2.bar(x + (i-1)*w, peaks[t], w, label=label)
ax2.set_xticks(x); ax2.set_xticklabels([f"Ma={m}" for m in mas])
ax2.set_ylabel("peak density"); ax2.set_title("Peak density vs Mach (higher = less numerical diffusion)")
ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
out2 = os.path.join(SCR, "mach_ladder_peak_density.png")
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print("wrote", out2)
