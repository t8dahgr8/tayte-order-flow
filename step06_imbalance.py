"""Step 6: the signal itself."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lobster import DATE, OUTPUT_DIR, TICKER
from features import FEATURES_CACHE, add_imbalance
from pipeline import TOP_OF_BOOK_CACHE


def main():
    book = add_imbalance(pd.read_pickle(TOP_OF_BOOK_CACHE))
    imb = book["imb"]

    print(f"\n{'=' * 78}\nSTEP 6 - IMBALANCE\n{'=' * 78}")
    assert imb.between(-1, 1).all(), "imb outside [-1, 1]"
    print("[PASS] imb within [-1, +1] for every row")
    assert imb.notna().all(), "imb has NaN"
    assert np.isfinite(imb).all(), "imb has inf"
    print("[PASS] no NaN or infinite values")

    print(f"\nmean              : {imb.mean():+.5f}")
    print(f"std               : {imb.std():.5f}")
    print(f"fraction |imb|>0.8: {(imb.abs() > 0.8).mean():.4%}")
    print(f"median |imb|      : {imb.abs().median():.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(imb, bins=50, color="#1f4e79", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("imbalance  (bid_size - ask_size) / (bid_size + ask_size)")
    ax.set_ylabel("rows")
    ax.set_title(f"{TICKER} {DATE} - queue imbalance at the touch")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig06_imbalance.png", dpi=130)
    plt.close(fig)

    book.to_pickle(FEATURES_CACHE)
    print(f"\nwrote fig06_imbalance.png and {FEATURES_CACHE.name}")


if __name__ == "__main__":
    main()
