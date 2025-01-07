# ancilla/models/bar_data.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class BarData:
    """
    Standardized structure for price bar data.

    Attributes:
        timestamp: Bar timestamp (timezone-aware)
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        vwap: Volume-weighted average price (optional)
        trades: Number of trades (optional)
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None
