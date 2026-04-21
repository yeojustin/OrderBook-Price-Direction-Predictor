from __future__ import annotations

from collections import deque


def make_feature_row(
    bids: list[list[str]],
    asks: list[list[str]],
    mid_history: deque[float],
    momentum_periods: int,
) -> dict[str, float] | None:
    if len(bids) < 5 or len(asks) < 5:
        return None
    b_q0 = float(bids[0][1])
    a_q0 = float(asks[0][1])
    obi_denom = b_q0 + a_q0
    if obi_denom == 0:
        return None
    bid_depth = sum(float(q) for _, q in bids[:5])
    ask_depth = sum(float(q) for _, q in asks[:5])
    depth_denom = bid_depth + ask_depth
    if depth_denom == 0:
        return None
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    mid_history.append(mid)
    if len(mid_history) <= momentum_periods:
        return None
    old_mid = mid_history[-(momentum_periods + 1)]
    momentum = (mid / old_mid) - 1.0 if old_mid != 0 else 0.0
    return {
        "mid_price": mid,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "obi": (b_q0 - a_q0) / obi_denom,
        "spread": best_ask - best_bid,
        "depth_skew": (bid_depth - ask_depth) / depth_denom,
        "momentum_10s": momentum,
    }

