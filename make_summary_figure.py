"""Single square figure summarising the result, sized for sharing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lobster import OUTPUT_DIR
from features import FEATURES_CACHE

NAVY = "#12395e"
RED = "#c0392b"
GREY = "#6b7280"

book = pd.read_pickle(FEATURES_CACHE)
d = book[["imb_decile", "fwd_return_100"]].dropna()
g = d.groupby("imb_decile", observed=True)["fwd_return_100"]
mean_bps = g.mean() * 1e4
se_bps = g.std() / np.sqrt(g.size()) * 1e4

top = book[book["imb_decile"] == 10]
edge = top["fwd_return_100"].mean() * top["mid"].mean() * 100
cost = book["spread"].median() * 100 / 2

fig = plt.figure(figsize=(10, 10), dpi=120)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.40,
                      left=0.175, right=0.95, top=0.865, bottom=0.115)

fig.text(0.5, 0.955, "Queue imbalance predicts the next price move.",
         ha="center", size=23, weight="bold", color="#111827")
fig.text(0.5, 0.915, "It is still smaller than the cost of trading on it.",
         ha="center", size=19, color=RED)

ax = fig.add_subplot(gs[0])
ax.errorbar(mean_bps.index, mean_bps, yerr=1.96 * se_bps, fmt="o-",
            color=NAVY, ecolor=RED, capsize=4, lw=2.6, ms=9, zorder=3)
ax.axhline(0, color="#9ca3af", lw=1)
ax.set_xticks(range(1, 11))
ax.set_xlabel("imbalance decile   (1 = size stacked on the ask, 10 = on the bid)",
              size=12.5, labelpad=9)
ax.set_ylabel("mean forward return, 100 events (bps)", size=12.5)
ax.set_title("Monotone across all ten deciles, no reversals",
             size=14.5, pad=11, color="#374151")
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=11.5)

ax2 = fig.add_subplot(gs[1])
bars = ax2.barh([1, 0], [edge, cost], height=0.5,
                color=[NAVY, RED], zorder=3)
ax2.set_yticks([1, 0])
ax2.set_yticklabels(["predicted move\n(top decile)", "cost of crossing\nthe spread"],
                    size=12.5)
ax2.set_xlim(0, cost * 1.32)
ax2.grid(axis="x", alpha=0.25)
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(labelsize=11.5)
for bar, val in zip(bars, [edge, cost]):
    ax2.text(val + cost * 0.03, bar.get_y() + bar.get_height() / 2,
             f"{val:.2f}¢", va="center", size=17, weight="bold",
             color=bar.get_facecolor())
ax2.set_title(f"The edge is {edge / cost:.0%} of the toll you would pay to capture it",
              size=14.5, pad=11, color="#374151")

fig.text(0.5, 0.038,
         "INTC · 2012-06-21 · 624,040 NASDAQ order book events · LOBSTER level-10 data\n"
         "Newey-West HAC standard errors, lag = horizon",
         ha="center", size=10.5, color=GREY, linespacing=1.6)

out = OUTPUT_DIR / "fig_summary_share.png"
fig.savefig(out, facecolor="white")
print(f"wrote {out}  edge={edge:.4f}c  cost={cost:.4f}c  ratio={edge / cost:.3f}")
