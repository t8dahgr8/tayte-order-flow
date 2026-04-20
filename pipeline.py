"""Steps 2-4: units, session filter, top of book."""
import numpy as np
import pandas as pd

from lobster import (
    ASK_PRICE_COLS, ASK_SIZE_COLS, BID_PRICE_COLS, BID_SIZE_COLS,
    ASK_SENTINEL, BID_SENTINEL, DATE, OUTPUT_DIR,
)

PRICE_SCALE = 10_000
SESSION_OPEN = 34_200.0   #09:30:00
SESSION_CLOSE = 57_600.0  #16:00:00
NON_CONTINUOUS_EVENTS = (6, 7)

TOP_OF_BOOK_CACHE = OUTPUT_DIR / "top_of_book.pkl"


def sentinel_masks(orderbook):
    """Find unoccupied level slots two ways so we can check they agree."""
    by_volume = pd.concat(
        [orderbook[ASK_SIZE_COLS].eq(0).rename(columns=dict(zip(ASK_SIZE_COLS, ASK_PRICE_COLS))),
         orderbook[BID_SIZE_COLS].eq(0).rename(columns=dict(zip(BID_SIZE_COLS, BID_PRICE_COLS)))],
        axis=1,
    )
    by_price = pd.concat(
        [orderbook[ASK_PRICE_COLS].eq(ASK_SENTINEL),
         orderbook[BID_PRICE_COLS].eq(BID_SENTINEL)],
        axis=1,
    )
    cols = ASK_PRICE_COLS + BID_PRICE_COLS
    return by_volume[cols], by_price[cols]


def convert_units(messages, orderbook, empty_mask):
    """Prices to dollars, add a plotting clock, blank padded levels."""
    messages = messages.copy()
    orderbook = orderbook.astype("float64")

    messages["price"] = messages["price"] / PRICE_SCALE
    #time stays the analysis clock, clock is for axis labels only
    messages["clock"] = pd.Timestamp(DATE) + pd.to_timedelta(messages["time"], unit="s")

    for price_col in ASK_PRICE_COLS + BID_PRICE_COLS:
        orderbook[price_col] = orderbook[price_col] / PRICE_SCALE

    #sizes are share counts, never scaled
    for price_col in ASK_PRICE_COLS + BID_PRICE_COLS:
        size_col = price_col.replace("price", "size")
        blank = empty_mask[price_col].to_numpy()
        orderbook.loc[blank, price_col] = np.nan
        orderbook.loc[blank, size_col] = np.nan

    return messages, orderbook


def session_masks(messages):
    """One mask per reason a row is not a continuous trading event."""
    before_open = messages["time"] < SESSION_OPEN
    after_close = messages["time"] > SESSION_CLOSE
    non_continuous = messages["event_type"].isin(NON_CONTINUOUS_EVENTS)
    return {
        "time < 09:30:00": before_open,
        "time > 16:00:00": after_close,
        "event_type 6 or 7": non_continuous,
        "_keep": ~(before_open | after_close | non_continuous),
    }


def restrict_session(messages, orderbook, keep):
    """Same mask, same reset, both frames. Touching one and not the other breaks alignment."""
    messages = messages.loc[keep].reset_index(drop=True)
    orderbook = orderbook.loc[keep].reset_index(drop=True)
    return messages, orderbook


def build_top_of_book(messages, orderbook):
    """Level 1 only, plus spread and mid."""
    book = pd.DataFrame({
        "time": messages["time"].to_numpy(),
        "clock": messages["clock"].to_numpy(),
        "event_type": messages["event_type"].to_numpy(),
        "bid": orderbook["bid_price_1"].to_numpy(),
        "ask": orderbook["ask_price_1"].to_numpy(),
        "bid_size": orderbook["bid_size_1"].to_numpy(),
        "ask_size": orderbook["ask_size_1"].to_numpy(),
    })
    book["spread"] = book["ask"] - book["bid"]
    book["mid"] = (book["ask"] + book["bid"]) / 2.0
    return book


def top_of_book_flaws(book):
    """Rows that a sane two sided book should never produce."""
    return {
        "crossed book (spread < 0)": book["spread"] < 0,
        "locked book (spread == 0)": book["spread"] == 0,
        "empty bid side (bid_size <= 0 or NaN)": ~(book["bid_size"] > 0),
        "empty ask side (ask_size <= 0 or NaN)": ~(book["ask_size"] > 0),
        "undefined spread (NaN)": book["spread"].isna(),
    }
