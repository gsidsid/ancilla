# ancilla/models/market_data.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict

from ancilla.formulae.indicators import (
    sma,
    ema,
    rsi,
    bollinger_bands,
    macd,
    atr,
    volume_weighted_average_price,
    historical_volatility,
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
    calculate_implied_volatility,
    calculate_option_greeks,
)


@dataclass
class MarketData:
    """Market data structure for a single instrument"""

    # OHLCV data
    # timestamp is set to the closing time of the bar
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None

    # Option-specific fields
    implied_vol: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # Technical indicators
    hist_volatility_20d: Optional[float] = None
    garman_klass_vol: Optional[float] = None
    parkinson_vol: Optional[float] = None
    yang_zhang_vol: Optional[float] = None
    sma_20: Optional[float] = None
    ema_20: Optional[float] = None
    rsi_14: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_lower: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr_14: Optional[float] = None
    vwap_intraday: Optional[float] = None

    def __getitem__(self, key):
        """Support dictionary-style access for backward compatibility with logging"""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


class MarketDataDict(dict):
    """
    A dictionary of MarketData objects indexed by ticker symbol.
    Provides helper methods for bulk processing of market data.
    """

    def __init__(
        self,
        data_dict: Dict[str, Dict],
        history_dict: Dict[str, List[Dict]],
        risk_free_rate: Optional[float] = 0.05,
    ):
        """
        Initialize MarketDataDict from raw data dictionaries

        Args:
            data_dict: Dictionary of current market data by ticker
            history_dict: Dictionary of historical market data by ticker
            risk_free_rate: Optional risk-free rate for options calculations
        """
        from ancilla.models import Option, InstrumentType

        super().__init__()

        # Process each ticker's data
        for ticker, ticker_data in data_dict.items():
            history = history_dict.get(ticker, [])

            # Determine if this is an option
            is_option = len(ticker) > 5 and "O:" in ticker
            underlying_data = None

            # If it's an option, get the underlying data
            if is_option:
                option = Option.from_option_ticker(ticker)
                underlying_symbol = option.ticker
                underlying_data = data_dict.get(underlying_symbol)

            # Create MarketData instance
            self[ticker] = self._create_market_data(
                ticker, ticker_data, history, risk_free_rate, underlying_data
            )

    def _create_market_data(
        self,
        ticker: str,
        data: Dict,
        history: List[Dict],
        risk_free_rate: Optional[float] = None,
        underlying_data: Optional[Dict] = None,
    ) -> MarketData:
        """Create a single MarketData instance with calculated indicators"""
        from ancilla.models import Option, InstrumentType
        from datetime import timezone

        # Extract basic price histories
        close_prices = [h["close"] for h in history] + [data["close"]]
        high_prices = [h["high"] for h in history] + [data["high"]]
        low_prices = [h["low"] for h in history] + [data["low"]]
        open_prices = [h["open"] for h in history] + [data["open"]]
        volume_values = [h["volume"] for h in history] + [data["volume"]]

        # Determine if this is an option
        is_option = len(ticker) > 5 and "O:" in ticker
        option_instrument = None
        if is_option:
            option_instrument = Option.from_option_ticker(ticker)

        # Calculate implied volatility for options
        implied_vol_val = None
        if option_instrument and underlying_data and risk_free_rate is not None:
            # Calculate time to expiry
            time_to_expiry = (
                option_instrument.expiration.replace(tzinfo=timezone.utc)
                - data["timestamp"]
            ).days / 365.0

            # First calculate implied volatility
            implied_vol_val = calculate_implied_volatility(
                option_price=data["close"],
                underlying_price=underlying_data["close"],
                strike_price=option_instrument.strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                is_call=option_instrument.instrument_type == InstrumentType.CALL_OPTION,
            )

            # If we have implied vol, calculate all Greeks
            if implied_vol_val is not None:
                greeks = calculate_option_greeks(
                    option_price=data["close"],
                    underlying_price=underlying_data["close"],
                    strike_price=option_instrument.strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    implied_vol=implied_vol_val,
                    is_call=option_instrument.instrument_type
                    == InstrumentType.CALL_OPTION,
                )
                data.update(greeks)  # Add calculated Greeks to data dict

        # Calculate intraday VWAP
        current_date = data["timestamp"].date()
        intraday_data = [h for h in history if h["timestamp"].date() == current_date]
        if intraday_data:
            intraday_high = [h["high"] for h in intraday_data] + [data["high"]]
            intraday_low = [h["low"] for h in intraday_data] + [data["low"]]
            intraday_close = [h["close"] for h in intraday_data] + [data["close"]]
            intraday_volume = [h["volume"] for h in intraday_data] + [data["volume"]]
            vwap_val = volume_weighted_average_price(
                intraday_high, intraday_low, intraday_close, intraday_volume
            )
        else:
            vwap_val = data.get("vwap")

        bollinger = bollinger_bands(close_prices)
        macd_vals = macd(close_prices)

        return MarketData(
            timestamp=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            vwap=data.get("vwap"),
            trades=data.get("trades"),
            implied_vol=implied_vol_val,
            delta=data.get("delta"),
            gamma=data.get("gamma"),
            theta=data.get("theta"),
            vega=data.get("vega"),
            rho=data.get("rho"),
            hist_volatility_20d=historical_volatility(close_prices),
            garman_klass_vol=garman_klass_volatility(
                high_prices,
                low_values=low_prices,
                open_values=open_prices,
                close_values=close_prices,
            ),
            parkinson_vol=parkinson_volatility(high_prices, low_prices),
            yang_zhang_vol=yang_zhang_volatility(
                open_prices, high_prices, low_prices, close_prices
            ),
            sma_20=sma(close_prices, 20),
            ema_20=ema(close_prices, 20),
            rsi_14=rsi(close_prices),
            bb_middle=bollinger[0],
            bb_upper=bollinger[1],
            bb_lower=bollinger[2],
            macd_line=macd_vals[0],
            macd_signal=macd_vals[1],
            macd_histogram=macd_vals[2],
            atr_14=atr(high_prices, low_prices, close_prices),
            vwap_intraday=vwap_val,
        )
