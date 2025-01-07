# ancilla/formulae/options.py
from typing import Dict
import numpy as np
from scipy.stats import norm


def black_scholes(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> Dict[str, float]:
    """
    Calculate Black-Scholes option price and Greeks.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'

    Returns:
        Dictionary with price and Greeks
    """
    if T <= 0:
        return {
            "price": max(0, S - K) if option_type == "call" else max(0, K - S),
            "delta": 1.0 if option_type == "call" else -1.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    if option_type == "call":
        price = S * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = N_d1 - 1

    gamma = n_d1 / (S * sigma * np.sqrt(T))
    theta = (
        -S * sigma * n_d1 / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2
        if option_type == "call"
        else -S * sigma * n_d1 / (2 * np.sqrt(T))
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    ) / 365
    vega = S * np.sqrt(T) * n_d1 / 100  # Divided by 100 to match market convention

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
    }
