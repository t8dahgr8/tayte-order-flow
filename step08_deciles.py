"""Step 8: mean forward return by imbalance decile."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lobster import DATE, OUTPUT_DIR, TICKER
from features import FEATURES_CACHE, HORIZONS


def decile_stats(book, horizon):
    col = f"fwd_return_{horizon}"
    d = book[["imb_decile", col]].dropna()
    g = d.groupby("imb_decile", observed=True)[col]
    out = pd.DataFrame({"n": g.size(), "mean_bps": g.mean() * 1e4,
                        "se_bps": g.std() / np.sqrt(g.size()) * 1e4})
    out["lo"] = out["mean_bps"] - 1.96 * out["se_bps"]
    out["hi"] = out["mean_bps"] + 1.96 * out["se_bps"]
    return out


def main():
    book = pd.read_pickle(FEATURES_CACHE)
    #equal count buckets, so each decile carries the same weight
    book["imb_decile"] = pd.qcut(book["imb"], 10, labels=False, duplicates="drop") + 1

    print(f"\n{'=' * 78}\nSTEP 8 - BUCKETED RELATIONSHIP\n{'=' * 78}")
    edges = pd.qcut(book["imb"], 10, duplicates="drop").cat.categories
    print("decile edges of imb:")
    for i, e in enumerate(edges, 1):
        print(f"  {i:>2}  ({e.left:+.4f}, {e.right:+.4f}]")

    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(14, 4.2), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        stats = decile_stats(book, h)
        print(f"\nhorizon N={h}")
        print(stats.to_string(float_format=lambda v: f"{v:9.4f}"))
        ax.errorbar(stats.index, stats["mean_bps"],
                    yerr=1.96 * stats["se_bps"], fmt="o-", capsize=3,
                    color="#1f4e79", ecolor="#a33", lw=1.2, ms=4)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"N = {h} events")
        ax.set_xlabel("imbalance decile")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean forward return (bps)")
    fig.suptitle(f"{TICKER} {DATE} - mean forward return by imbalance decile "
                 f"(error bars +/-1.96 SE)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig08_deciles.png", dpi=130)
    plt.close(fig)

    book.to_pickle(FEATURES_CACHE)
    print("\nwrote fig08_deciles.png")


if __name__ == "__main__":
    main()
