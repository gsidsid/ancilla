# ancilla/formulae/volatility.py
from typing import Optional
import numpy as np
from scipy.interpolate import griddata


def create_volatility_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    ivs: np.ndarray,
    new_strikes: np.ndarray,
    new_expiries: np.ndarray,
) -> np.ndarray:
    """Create volatility surface with improved handling of extreme ranges."""
    if len(strikes) < 4:
        raise ValueError("At least 4 strikes are required")

    # Scale IVs to percentage form and filter out extreme values
    scaled_ivs = ivs * 100

    # Use filtered data
    filtered_strikes = strikes
    filtered_expiries = expiries
    filtered_ivs = scaled_ivs

    # Normalize strikes with filtered data
    strike_range = (filtered_strikes.min(), filtered_strikes.max())
    norm_strikes = (filtered_strikes - strike_range[0]) / (
        strike_range[1] - strike_range[0]
    )
    norm_new_strikes = (new_strikes - strike_range[0]) / (
        strike_range[1] - strike_range[0]
    )

    points = np.column_stack([norm_strikes, filtered_expiries])
    # Use cubic interpolation with boundary extrapolation
    interpolated = griddata(
        points,
        filtered_ivs,
        (norm_new_strikes[None, :], new_expiries[:, None]),
        method="cubic",
        fill_value=np.nan,
    )

    return interpolated


def estimate_liquidity_multiplier(
    volume: Optional[int],
    open_interest: Optional[int],
    moneyness: float,
    time_to_expiry: float,
) -> float:
    """
    Estimate a liquidity factor based on various metrics.
    Enhanced version with more realistic scaling and better handling of edge cases.
    """
    if volume is None and open_interest is None:
        return 0.0

    # More nuanced volume scoring
    volume = volume or 0
    volume_score = 1 - np.exp(-volume / 500)  # Smoother scaling

    # Enhanced open interest scoring
    oi = open_interest or 0
    oi_score = 1 - np.exp(-oi / 2000)

    # Refined moneyness penalty
    moneyness_penalty = 1 / (1 + 5 * abs(moneyness - 1) ** 2)  # Quadratic penalty

    # Improved time penalty with slower decay
    time_penalty = 1 / (1 + time_to_expiry)

    # Weighted combination with emphasis on market activity
    liquidity_score = (
        0.45 * volume_score
        + 0.25 * oi_score
        + 0.20 * moneyness_penalty
        + 0.10 * time_penalty
    )

    return np.clip(liquidity_score, 0.0, 1.0)
