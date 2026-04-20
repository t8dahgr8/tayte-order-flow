"""Step 1: load both files and prove they are row aligned."""
import sys

import numpy as np
import pandas as pd

from lobster import (
    ASK_PRICE_COLS, ASK_SIZE_COLS, BID_PRICE_COLS, BID_SIZE_COLS,
    MESSAGE_FILE, ORDERBOOK_FILE, load_raw,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def size_at_price(book_row, price, side):
    """Visible size resting at price, or None if that price is outside the 10 level window.

    None and 0 are different. None means we cannot see it. Lobster never writes a
    real level with size 0, an emptied level just drops out and a deeper one moves up.
    """
    price_cols = ASK_PRICE_COLS if side == "ask" else BID_PRICE_COLS
    size_cols = ASK_SIZE_COLS if side == "ask" else BID_SIZE_COLS
    prices = book_row[price_cols].to_numpy()
    sizes = book_row[size_cols].to_numpy()
    hit = prices == price
    if not hit.any():
        return None
    return int(sizes[hit].sum())


#how each event type moves the size sitting at its own price level
SIZE_DELTA_SIGN = {1: +1, 2: -1, 3: -1, 4: -1}


def check_alignment(messages, orderbook, n_samples=2000, seed=0):
    """Predict each level from the previous book row and see if row k matches.

    Sharp test. If the indexing is off by one nearly everything fails.
    """
    rng = np.random.default_rng(seed)
    eligible = messages.index[
        messages["event_type"].isin(SIZE_DELTA_SIGN) & (messages.index > 0)
    ].to_numpy()
    picks = rng.choice(eligible, size=min(n_samples, eligible.size), replace=False)

    checked = 0
    passed = 0
    failures = []
    for k in picks:
        msg = messages.loc[k]
        #direction describes the resting order, so it names the side this event hit
        side = "bid" if msg["direction"] == 1 else "ask"
        price = int(msg["price"])

        before = size_at_price(orderbook.loc[k - 1], price, side)
        if before is None:
            continue
        expected = before + SIZE_DELTA_SIGN[int(msg["event_type"])] * int(msg["size"])
        after = size_at_price(orderbook.loc[k], price, side)
        if after is None:
            #level emptied and fell out of the window, fine only if we expected zero
            after = 0 if expected == 0 else after

        checked += 1
        if after == expected:
            passed += 1
        elif len(failures) < 10:
            failures.append(
                dict(row=int(k), type=int(msg["event_type"]), side=side, price=price,
                     before=before, size=int(msg["size"]), expected=expected, actual=after)
            )
    return checked, passed, failures


def main():
    rule("STEP 1 - LOAD")
    print(f"message   file: {MESSAGE_FILE.name}  ({MESSAGE_FILE.stat().st_size:,} bytes)")
    print(f"orderbook file: {ORDERBOOK_FILE.name}  ({ORDERBOOK_FILE.stat().st_size:,} bytes)")

    messages, orderbook = load_raw()

    rule("ASSERTIONS")
    assert len(messages) == len(orderbook), (
        f"row count mismatch: {len(messages):,} vs {len(orderbook):,}, files are not a pair"
    )
    print(f"[PASS] row counts match: {len(messages):,} == {len(orderbook):,}")

    msg_nans = int(messages.isna().sum().sum())
    book_nans = int(orderbook.isna().sum().sum())
    assert msg_nans == 0, f"{msg_nans} NaNs in message frame"
    assert book_nans == 0, f"{book_nans} NaNs in orderbook frame"
    print(f"[PASS] no NaN values: message={msg_nans}, orderbook={book_nans}")

    non_numeric = [c for c, d in {**messages.dtypes.to_dict(),
                                  **orderbook.dtypes.to_dict()}.items()
                   if not pd.api.types.is_numeric_dtype(d)]
    assert not non_numeric, f"non-numeric columns: {non_numeric}"
    print(f"[PASS] all {messages.shape[1] + orderbook.shape[1]} columns numeric "
          f"({messages.shape[1]} message + {orderbook.shape[1]} orderbook)")

    rule("ROW COUNT")
    print(f"events loaded: {len(messages):,}")
    print("\nevent_type counts (raw, before any filtering):")
    labels = {1: "new visible limit order", 2: "partial cancellation",
              3: "total deletion", 4: "visible execution",
              5: "hidden execution", 6: "cross/auction trade", 7: "trading halt"}
    for etype, n in messages["event_type"].value_counts().sort_index().items():
        print(f"  {etype}  {labels.get(etype, '?'):<28} {n:>9,}  ({n / len(messages):6.2%})")

    rule("FIRST 20 ROWS - MESSAGE (left) vs ORDERBOOK LEVEL 1 (right)")
    print("Prices are still raw integers (dollars x 10000), Step 2 converts them.\n")
    print(pd.concat(
        [messages.head(20),
         orderbook.head(20)[["ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"]]],
        axis=1,
    ).to_string())

    rule("FULL 40-COLUMN ORDERBOOK STRUCTURE - FIRST 3 ROWS")
    print(orderbook.head(3).to_string())

    rule("ALIGNMENT PROOF - is orderbook row k BEFORE or AFTER message k?")
    checked, passed, failures = check_alignment(messages, orderbook)
    print(f"checked {checked:,} sampled events   passed {passed:,}   "
          f"failed {checked - passed:,}   ({passed / checked:.4%} agreement)")
    if failures:
        print("\nfirst failures:")
        print(pd.DataFrame(failures).to_string(index=False))

    if passed / checked > 0.99:
        print("\n[PASS] orderbook row k is the state AFTER message k.")
        return 0
    print("\n[FAIL] alignment is not as assumed, stop.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
