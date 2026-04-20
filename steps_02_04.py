"""Steps 2-4. Writes output/top_of_book.pkl so later steps skip the 158MB parse."""
import sys

import numpy as np
import pandas as pd

from lobster import (
    ASK_PRICE_COLS, BID_PRICE_COLS, LEVELS, OUTPUT_DIR, TICKER, DATE, load_raw,
)
from pipeline import (
    SESSION_CLOSE, SESSION_OPEN, TOP_OF_BOOK_CACHE, build_top_of_book,
    convert_units, restrict_session, sentinel_masks, session_masks,
    top_of_book_flaws,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    messages, orderbook = load_raw()
    n_raw = len(messages)

    #=== step 2 ===
    rule("STEP 2 - CONVERT UNITS")

    by_volume, by_price = sentinel_masks(orderbook)
    n_vol, n_price = int(by_volume.to_numpy().sum()), int(by_price.to_numpy().sum())
    disagree = int((by_volume.to_numpy() != by_price.to_numpy()).sum())

    print("Identifying unoccupied level slots two independent ways:")
    print(f"  volume == 0 rule            : {n_vol:,} cells")
    print(f"  price == +/-9999999999 rule : {n_price:,} cells")
    print(f"  cells where they disagree   : {disagree:,}")
    assert disagree == 0, (
        "The volume rule and the price rule disagree. Investigate before trusting either."
    )
    print("[PASS] the two rules are identical; adopting the volume rule (no magic threshold)")

    total_cells = by_volume.size
    print(f"\n{n_vol:,} of {total_cells:,} level slots are padding "
          f"({n_vol / total_cells:.4%}) - i.e. the 10-level window was "
          f"{'never' if n_vol == 0 else 'sometimes'} deeper than the real book.")

    messages, orderbook = convert_units(messages, orderbook, by_volume)

    print(f"\nprice scale applied: raw integer / 10000 -> dollars")
    print(f"  message price   : {messages['price'].min():>10.4f} .. {messages['price'].max():>10.4f}")
    print(f"  best ask (L1)   : {orderbook['ask_price_1'].min():>10.4f} .. {orderbook['ask_price_1'].max():>10.4f}")
    print(f"  best bid (L1)   : {orderbook['bid_price_1'].min():>10.4f} .. {orderbook['bid_price_1'].max():>10.4f}")
    print(f"\nclock column added for plotting only:")
    print(f"  {messages['clock'].iloc[0]}  ..  {messages['clock'].iloc[-1]}")

    #price sanity for the ticker
    for col in ("ask_price_1", "bid_price_1"):
        vals = orderbook[col].dropna()
        assert vals.between(1.0, 1000.0).all(), (
            f"{col} outside $1-$1000: min={vals.min()}, max={vals.max()}"
        )
    print(f"\n[PASS] all level-1 prices lie in $1-$1000 (plausible for {TICKER} in {DATE[:4]})")

    #=== step 3 ===
    rule("STEP 3 - RESTRICT THE SESSION")

    masks = session_masks(messages)
    keep = masks.pop("_keep")

    print(f"{'reason':<24}{'rows flagged':>14}{'% of file':>12}")
    print("-" * 50)
    for reason, mask in masks.items():
        n = int(mask.sum())
        print(f"{reason:<24}{n:>14,}{n / n_raw:>11.4%}")
    n_dropped = int((~keep).sum())
    print("-" * 50)
    print(f"{'total dropped':<24}{n_dropped:>14,}{n_dropped / n_raw:>11.4%}")
    print(f"{'kept':<24}{int(keep.sum()):>14,}{keep.sum() / n_raw:>11.4%}")

    messages, orderbook = restrict_session(messages, orderbook, keep)

    assert len(messages) == len(orderbook), "row counts diverged during filtering"
    print(f"\n[PASS] row counts still match after filtering: "
          f"{len(messages):,} == {len(orderbook):,}")
    assert messages.index.equals(orderbook.index), "indexes differ after reset"
    print("[PASS] both frames re-indexed identically (alignment preserved)")

    remaining = sorted(messages["event_type"].unique().tolist())
    assert 6 not in remaining and 7 not in remaining, f"event types 6/7 survive: {remaining}"
    print(f"[PASS] no event_type 6 or 7 remain; surviving types: {remaining}")

    tmin, tmax = messages["time"].min(), messages["time"].max()
    assert tmin >= SESSION_OPEN and tmax <= SESSION_CLOSE, f"time out of session: {tmin}, {tmax}"
    print(f"[PASS] time in session: {tmin:.6f} >= {SESSION_OPEN:.0f} and "
          f"{tmax:.6f} <= {SESSION_CLOSE:.0f}")

    #=== step 4 ===
    rule("STEP 4 - BUILD THE TOP OF BOOK")

    book = build_top_of_book(messages, orderbook)
    n_before = len(book)

    print("Working frame columns:", list(book.columns))
    print(f"rows: {n_before:,}\n")
    print(book.head(10).to_string())

    rule("STEP 4 - SANITY OF THE TOUCH (counts BEFORE dropping anything)")
    flaws = top_of_book_flaws(book)
    print(f"{'condition':<40}{'rows':>12}{'% of rows':>13}")
    print("-" * 65)
    for label, mask in flaws.items():
        n = int(mask.sum())
        print(f"{label:<40}{n:>12,}{n / n_before:>12.5%}")

    bad = np.zeros(n_before, dtype=bool)
    for mask in flaws.values():
        bad |= mask.to_numpy()
    n_bad = int(bad.sum())
    print("-" * 65)
    print(f"{'union of all failures':<40}{n_bad:>12,}{n_bad / n_before:>12.5%}")

    print(f"\nspread summary (cents): "
          f"min={book['spread'].min() * 100:.2f}  "
          f"median={book['spread'].median() * 100:.2f}  "
          f"max={book['spread'].max() * 100:.2f}")

    if n_bad:
        print(f"\nExamples of failing rows:")
        print(book.loc[bad].head(8).to_string())

    book = book.loc[~bad].reset_index(drop=True)
    print(f"\ndropped {n_bad:,} unusable rows ({n_bad / n_before:.5%}); "
          f"{len(book):,} remain")

    assert (book["spread"] > 0).all(), "spread not strictly positive after cleaning"
    assert (book["bid_size"] > 0).all(), "bid_size not positive after cleaning"
    assert (book["ask_size"] > 0).all(), "ask_size not positive after cleaning"
    print("[PASS] every remaining row: spread > 0, bid_size > 0, ask_size > 0")

    book.to_pickle(TOP_OF_BOOK_CACHE)
    print(f"\ncached working frame -> {TOP_OF_BOOK_CACHE.relative_to(TOP_OF_BOOK_CACHE.parent.parent)}"
          f"  ({TOP_OF_BOOK_CACHE.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
