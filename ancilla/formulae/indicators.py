# ancilla/formulae/indicators.py

from typing import List, Optional, Union
import numpy as np
from scipy.stats import norm
import math
from dataclasses import dataclass
from datetime import datetime


def sma(values: List[float], period: int) -> Optional[float]:
    """
    Simple Moving Average

    Args:
        values: List of price values
        period: Number of periods to average

    Returns:
        The simple moving average or None if insufficient data
    """
    if len(values) < period:
        return None
    return np.mean(values[-period:])


def ema(values: List[float], period: int, smoothing: float = 2.0) -> Optional[float]:
    """
    Exponential Moving Average

    Args:
        values: List of price values
        period: Number of periods for EMA calculation
        smoothing: Smoothing factor (default=2.0)

    Returns:
        The exponential moving average or None if insufficient data
    """
    if len(values) < period:
        return None

    alpha = smoothing / (1 + period)
    ema_val = values[-period]

    for price in values[-period + 1 :]:
        ema_val = price * alpha + ema_val * (1 - alpha)

    return ema_val


def rsi(values: List[float], period: int = 14) -> Optional[float]:
    """
    Relative Strength Index

    Args:
        values: List of price values
        period: RSI period (default=14)

    Returns:
        The RSI value or None if insufficient data
    """
    if len(values) < period + 1:
        return None

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    values: List[float], period: int = 20, num_std: float = 2.0
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Bollinger Bands (Middle, Upper, Lower)

    Args:
        values: List of price values
        period: Period for moving average (default=20)
        num_std: Number of standard deviations (default=2)

    Returns:
        Tuple of (middle_band, upper_band, lower_band) or (None, None, None) if insufficient data
    """
    if len(values) < period:
        return None, None, None

    middle = sma(values, period)
    std_dev = np.std(values[-period:])

    upper = middle + (std_dev * num_std)
    lower = middle - (std_dev * num_std)

    return middle, upper, lower


def macd(
    values: List[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Moving Average Convergence Divergence

    Args:
        values: List of price values
        fast_period: Fast EMA period (default=12)
        slow_period: Slow EMA period (default=26)
        signal_period: Signal line period (default=9)

    Returns:
        Tuple of (macd_line, signal_line, histogram) or (None, None, None) if insufficient data
    """
    if len(values) < slow_period:
        return None, None, None

    fast_ema = ema(values, fast_period)
    slow_ema = ema(values, slow_period)

    if fast_ema is None or slow_ema is None:
        return None, None, None

    macd_line = fast_ema - slow_ema
    macd_values = [macd_line]  # You would normally maintain a history of MACD values
    signal_line = ema(macd_values, signal_period)

    if signal_line is None:
        return macd_line, None, None

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(
    high_values: List[float],
    low_values: List[float],
    close_values: List[float],
    period: int = 14,
) -> Optional[float]:
    """
    Average True Range

    Args:
        high_values: List of high prices
        low_values: List of low prices
        close_values: List of closing prices
        period: ATR period (default=14)

    Returns:
        The ATR value or None if insufficient data
    """
    if len(high_values) < period + 1:
        return None

    tr_values = []
    for i in range(1, len(high_values)):
        high = high_values[i]
        low = low_values[i]
        prev_close = close_values[i - 1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)

    return np.mean(tr_values[-period:])


def volume_weighted_average_price(
    high_values: List[float],
    low_values: List[float],
    close_values: List[float],
    volume_values: List[int],
    period: int = None,
) -> Optional[float]:
    """
    Volume Weighted Average Price (VWAP)

    Args:
        high_values: List of high prices
        low_values: List of low prices
        close_values: List of closing prices
        volume_values: List of volume values
        period: Optional period for calculation (default=None for entire dataset)

    Returns:
        The VWAP value or None if insufficient data
    """
    if len(high_values) < 1:
        return None

    if period:
        high_values = high_values[-period:]
        low_values = low_values[-period:]
        close_values = close_values[-period:]
        volume_values = volume_values[-period:]

    typical_prices = [
        (h + l + c) / 3 for h, l, c in zip(high_values, low_values, close_values)
    ]
    price_volume = sum(tp * v for tp, v in zip(typical_prices, volume_values))
    total_volume = sum(volume_values)

    return price_volume / total_volume if total_volume > 0 else None


def historical_volatility(
    close_prices: List[float], period: int = 20, annualization_factor: int = 252
) -> Optional[float]:
    """
    Calculate historical volatility (standard deviation of log returns, annualized)

    Args:
        close_prices: List of closing prices
        period: Period for volatility calculation (default=20)
        annualization_factor: Number of trading periods in a year (default=252 for daily data)

    Returns:
        Annualized volatility as a decimal (e.g., 0.15 for 15% volatility) or None if insufficient data
    """
    if len(close_prices) < period + 1:
        return None

    # Calculate log returns
    prices = np.array(close_prices[-period - 1 :])
    log_returns = np.log(prices[1:] / prices[:-1])

    # Calculate annualized volatility
    return np.std(log_returns) * np.sqrt(annualization_factor)


def garman_klass_volatility(
    high_values: List[float],
    low_values: List[float],
    open_values: List[float],
    close_values: List[float],
    period: int = 20,
    annualization_factor: int = 252,
) -> Optional[float]:
    """
    Calculate Garman-Klass volatility estimator using OHLC data

    Args:
        high_values: List of high prices
        low_values: List of low prices
        open_values: List of opening prices
        close_values: List of closing prices
        period: Period for volatility calculation (default=20)
        annualization_factor: Number of trading periods in a year (default=252 for daily data)

    Returns:
        Annualized Garman-Klass volatility estimate or None if insufficient data
    """
    if len(high_values) < period:
        return None

    # Truncate to period length
    high_values = high_values[-period:]
    low_values = low_values[-period:]
    open_values = open_values[-period:]
    close_values = close_values[-period:]

    # Calculate components
    log_hl = np.log(np.array(high_values) / np.array(low_values))
    log_co = np.log(np.array(close_values) / np.array(open_values))

    # Garman-Klass estimator
    gk = 0.5 * np.mean(log_hl**2) - (2 * np.log(2) - 1) * np.mean(log_co**2)

    # Annualize the volatility
    return np.sqrt(gk * annualization_factor)


def parkinson_volatility(
    high_values: List[float],
    low_values: List[float],
    period: int = 20,
    annualization_factor: int = 252,
) -> Optional[float]:
    """
    Calculate Parkinson volatility estimator using high-low range

    Args:
        high_values: List of high prices
        low_values: List of low prices
        period: Period for volatility calculation (default=20)
        annualization_factor: Number of trading periods in a year (default=252 for daily data)

    Returns:
        Annualized Parkinson volatility estimate or None if insufficient data
    """
    if len(high_values) < period:
        return None

    # Calculate log high-low ranges
    high_values = np.array(high_values[-period:])
    low_values = np.array(low_values[-period:])
    log_hl = np.log(high_values / low_values)

    # Parkinson estimator
    estimator = 1.0 / (4.0 * period * np.log(2.0)) * np.sum(log_hl**2)

    # Annualize the volatility
    return np.sqrt(estimator * annualization_factor)


def calculate_implied_volatility(
    option_price: float,
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    is_call: bool,
    precision: float = 0.00001,
    max_iterations: int = 100,
) -> Optional[float]:
    """
    Calculate implied volatility using Newton-Raphson

    Args:
        option_price: Market price of the option
        underlying_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free interest rate (as decimal)
        is_call: True for call options, False for puts
        precision: Desired precision for the calculation
        max_iterations: Maximum number of iterations

    Returns:
        Implied volatility as a decimal or None if the calculation fails to converge
    """
    # Avoid division by zero, imaginary values
    time_to_expiry = max(time_to_expiry, 1e-10)
    strike_price = max(strike_price, 0.01)

    def black_scholes(sigma):
        d1 = (
            np.log(underlying_price / strike_price)
            + (risk_free_rate + 0.5 * sigma**2) * time_to_expiry
        ) / (sigma * np.sqrt(time_to_expiry))
        d2 = d1 - sigma * np.sqrt(time_to_expiry)

        if is_call:
            price = underlying_price * norm.cdf(d1) - strike_price * np.exp(
                -risk_free_rate * time_to_expiry
            ) * norm.cdf(d2)
        else:
            price = strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(
                -d2
            ) - underlying_price * norm.cdf(-d1)

        return price

    def vega(sigma):
        d1 = (
            np.log(underlying_price / strike_price)
            + (risk_free_rate + 0.5 * sigma**2) * time_to_expiry
        ) / (sigma * np.sqrt(time_to_expiry))
        return underlying_price * np.sqrt(time_to_expiry) * norm.pdf(d1)

    # Initial guess
    sigma = 0.3  # Start with 30% volatility

    for i in range(max_iterations):
        price = black_scholes(sigma)
        diff = option_price - price

        if abs(diff) < precision:
            return sigma

        v = vega(sigma)
        if abs(v) < 1e-10:  # Avoid division by zero
            return None

        sigma = sigma + diff / v

        if sigma <= 0:  # Volatility can't be negative
            return None

    return None  # Failed to converge


def calculate_option_greeks(
    option_price: float,
    underlying_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    implied_vol: float,
    is_call: bool,
) -> dict:
    """
    Calculate all option Greeks

    Args:
        option_price: Current option price
        underlying_price: Current price of the underlying asset
        strike_price: Option strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free interest rate (as decimal)
        implied_vol: Option's implied volatility
        is_call: True for call options, False for puts

    Returns:
        Dictionary containing all Greeks (delta, gamma, theta, vega, rho)
    """
    if implied_vol <= 0 or time_to_expiry <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}

    sqrt_t = np.sqrt(time_to_expiry)

    # Calculate d1 and d2
    d1 = (
        np.log(underlying_price / strike_price)
        + (risk_free_rate + 0.5 * implied_vol**2) * time_to_expiry
    ) / (implied_vol * sqrt_t)
    d2 = d1 - implied_vol * sqrt_t

    # Standard normal PDF and CDF
    norm_pdf_d1 = norm.pdf(d1)

    if is_call:
        norm_cdf_d1 = norm.cdf(d1)
        norm_cdf_d2 = norm.cdf(d2)
    else:
        norm_cdf_d1 = norm.cdf(-d1)
        norm_cdf_d2 = norm.cdf(-d2)

    # Calculate Greeks
    # Delta
    delta = norm_cdf_d1 if is_call else -norm_cdf_d1

    # Gamma (same for calls and puts)
    gamma = norm_pdf_d1 / (underlying_price * implied_vol * sqrt_t)

    # Theta
    theta_part1 = -(underlying_price * norm_pdf_d1 * implied_vol) / (2 * sqrt_t)
    theta_part2 = (
        risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_expiry)
    )

    if is_call:
        theta = theta_part1 - theta_part2 * norm_cdf_d2
    else:
        theta = theta_part1 + theta_part2 * (1 - norm_cdf_d2)

    # Vega (same for calls and puts)
    vega = (
        underlying_price * sqrt_t * norm_pdf_d1 / 100
    )  # Divided by 100 for percentage move

    # Rho
    if is_call:
        rho = (
            strike_price
            * time_to_expiry
            * np.exp(-risk_free_rate * time_to_expiry)
            * norm_cdf_d2
            / 100
        )
    else:
        rho = (
            -strike_price
            * time_to_expiry
            * np.exp(-risk_free_rate * time_to_expiry)
            * (1 - norm_cdf_d2)
            / 100
        )

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def yang_zhang_volatility(
    open_values: List[float],
    high_values: List[float],
    low_values: List[float],
    close_values: List[float],
    period: int = 20,
    annualization_factor: int = 252,
) -> Optional[float]:
    """
    Calculate Yang-Zhang volatility estimator, which is robust to opening jumps and drift

    Args:
        open_values: List of opening prices
        high_values: List of high prices
        low_values: List of low prices
        close_values: List of closing prices
        period: Period for volatility calculation (default=20)
        annualization_factor: Number of trading periods in a year (default=252 for daily data)

    Returns:
        Annualized Yang-Zhang volatility estimate or None if insufficient data
    """
    if len(open_values) < period:
        return None

    # Convert to numpy arrays and get the last 'period' elements
    opens = np.array(open_values[-period:])
    highs = np.array(high_values[-period:])
    lows = np.array(low_values[-period:])
    closes = np.array(close_values[-period:])

    # Calculate overnight volatility (close-to-open)
    # Use [:-1] and [1:] to ensure matching array lengths
    log_co = np.log(opens[1:] / closes[:-1])
    vo = np.sum(log_co**2) / (period - 1)

    # Calculate open-to-close volatility
    log_oc = np.log(closes[:-1] / opens[1:])
    voc = np.sum(log_oc**2) / (period - 1)

    # Calculate Rogers-Satchell volatility
    # Use arrays of the same length
    log_ho = np.log(highs[:-1] / opens[1:])
    log_lo = np.log(lows[:-1] / opens[1:])
    log_hc = np.log(highs[:-1] / closes[:-1])
    log_lc = np.log(lows[:-1] / closes[:-1])

    rs = np.sum(log_ho * (log_ho - log_lc) + log_lo * (log_lo - log_hc)) / (period - 1)

    # Calculate k (weight factor)
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    # Yang-Zhang volatility
    sigma2 = vo + k * voc + (1 - k) * rs

    # Annualize the volatility
    return np.sqrt(sigma2 * annualization_factor)
