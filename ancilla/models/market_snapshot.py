# ancilla/models/market_snapshot.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MarketSnapshot:
    """
    Standardized structure for market snapshots.

    Attributes:
        timestamp: Snapshot timestamp (timezone-aware)
        price: Current price
        bid: Best bid price (optional)
        ask: Best ask price (optional)
        bid_size: Size at best bid (optional)
        ask_size: Size at best ask (optional)
        volume: Trading volume (optional)
        vwap: Volume-weighted average price (optional)
    """

    timestamp: datetime
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
