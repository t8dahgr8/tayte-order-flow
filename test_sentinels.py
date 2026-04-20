"""The real INTC file has zero padded levels so the sentinel code never fires.
Feed it synthetic padding to prove it actually works."""
import numpy as np
import pandas as pd

from lobster import ASK_SENTINEL, BID_SENTINEL, ORDERBOOK_COLS
from pipeline import convert_units, sentinel_masks

#row 0 fully occupied, row 1 has level 10 padded on both sides
row_full = {c: (275000 if "price" in c else 500) for c in ORDERBOOK_COLS}
row_pad = dict(row_full)
row_pad.update(ask_price_10=ASK_SENTINEL, ask_size_10=0,
               bid_price_10=BID_SENTINEL, bid_size_10=0)
book = pd.DataFrame([row_full, row_pad], columns=ORDERBOOK_COLS).astype("int64")
msgs = pd.DataFrame({"time": [34200.0, 34200.5], "event_type": [1, 1],
                     "order_id": [1, 2], "size": [10, 10],
                     "price": [275000, 275000], "direction": [1, -1]})

by_volume, by_price = sentinel_masks(book)
flagged_vol = sorted(by_volume.columns[by_volume.iloc[1]].tolist())
flagged_price = sorted(by_price.columns[by_price.iloc[1]].tolist())
expected = ["ask_price_10", "bid_price_10"]

assert flagged_vol == expected, f"volume rule missed padding: {flagged_vol}"
assert flagged_price == expected, f"price rule missed padding: {flagged_price}"
assert not by_volume.iloc[0].any(), "volume rule false-positived on a full row"
assert not by_price.iloc[0].any(), "price rule false-positived on a full row"
print(f"[PASS] both rules flag exactly {expected} on the padded row, nothing on the full row")

_, converted = convert_units(msgs, book, by_volume)
assert converted.loc[1, ["ask_price_10", "ask_size_10", "bid_price_10", "bid_size_10"]].isna().all(), \
    "padded level was not blanked to NaN"
assert converted.loc[0].notna().all(), "convert_units blanked a real level"
assert converted.loc[0, "ask_price_1"] == 27.50, "price scaling wrong"
print("[PASS] convert_units blanks the padded level to NaN, leaves real levels intact, scales prices")
print("\nrow 1 after conversion, level 9 and 10:")
print(converted.loc[[1], ["ask_price_9", "ask_size_9", "ask_price_10", "ask_size_10",
                          "bid_price_10", "bid_size_10"]].to_string())
