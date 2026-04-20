"""Step 5: look at the day before trusting anything computed from it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from lobster import DATE, OUTPUT_DIR, TICKER
from pipeline import TOP_OF_BOOK_CACHE


def main():
    book = pd.read_pickle(TOP_OF_BOOK_CACHE)
    cents = book["spread"] * 100

    print(f"\n{'=' * 78}\nSTEP 5 - DIAGNOSTIC PLOTS\n{'=' * 78}")
    print(f"rows: {len(book):,}")
    print(f"fraction of rows with a one cent spread : {(cents.round() == 1).mean():.4%}")
    print(f"median spread (cents)                   : {cents.median():.2f}")
    print(f"median bid_size (shares)                : {book['bid_size'].median():,.0f}")
    print(f"median ask_size (shares)                : {book['ask_size'].median():,.0f}")
    print(f"mid range                               : {book['mid'].min():.3f} - {book['mid'].max():.3f}")

    #spread by half hour, the open vs rest of day check
    half_hour = book["clock"].dt.floor("30min")
    by_bucket = pd.DataFrame({
        "mean_spread_cents": cents.groupby(half_hour).mean(),
        "pct_one_cent": (cents.round() == 1).groupby(half_hour).mean(),
        "rows": cents.groupby(half_hour).size(),
    })
    print("\nspread by half hour:")
    print(by_bucket.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(book["clock"], book["mid"], lw=0.4, color="#1f4e79")
    ax.set_title(f"{TICKER} {DATE} - mid price")
    ax.set_ylabel("mid ($)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig05_mid.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(book["clock"], cents, lw=0.3, color="#a33", alpha=0.7)
    ax.set_title(f"{TICKER} {DATE} - spread (cents)")
    ax.set_ylabel("spread (cents)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig05_spread.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    lo, hi = int(cents.min()), int(cents.max())
    ax.hist(cents, bins=[b - 0.5 for b in range(lo, hi + 2)],
            color="#1f4e79", edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("spread (cents)")
    ax.set_ylabel("rows (log)")
    ax.set_title(f"{TICKER} {DATE} - spread histogram")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig05_spread_hist.png", dpi=130)
    plt.close(fig)

    print("\nwrote fig05_mid.png, fig05_spread.png, fig05_spread_hist.png")


if __name__ == "__main__":
    main()
