# ancilla/models/option_data.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class OptionData:
    """
    Standardized structure for option contract data.

    Attributes:
        ticker: Ticker symbol of the underlying asset
        strike: Strike price of the option
        expiration: Expiration datetime (timezone-aware)
        contract_type: Type of option ('call' or 'put')
        implied_volatility: Option's implied volatility (as decimal)
        underlying_price: Current price of the underlying asset
        delta: Option delta (optional)
        gamma: Option gamma (optional)
        theta: Option theta (optional)
        vega: Option vega (optional)
        bid: Current bid price (optional)
        ask: Current ask price (optional)
        volume: Trading volume (optional)
        open_interest: Open interest (optional)
        last_trade: Last trade datetime (optional)
    """

    ticker: str
    strike: float
    expiration: datetime
    contract_type: str
    implied_volatility: float
    underlying_price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    last_trade: Optional[Dict] = None
