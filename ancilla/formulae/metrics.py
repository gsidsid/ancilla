# ancilla/formulae/metrics.py
from typing import Dict, Any, Sequence, Optional
from ancilla.models import InstrumentType
import numpy as np
import pandas as pd


def calculate_return_metrics(
    equity_series: pd.Series, trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculate basic return-based metrics from an equity curve.

    Args:
        equity_series: Series of equity values indexed by timestamp
        trading_days: Number of trading days per year (default: 252)

    Returns:
        Dictionary containing return metrics
    """
    returns = equity_series.pct_change()
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    period_length = len(equity_series)
    daily_std = returns.std()  # relatable
    annualized_vol = daily_std * np.sqrt(trading_days) if daily_std != 0 else 0
    annualized_return = ((1 + total_return) ** (trading_days / period_length)) - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": (
            (annualized_return / annualized_vol) if annualized_vol != 0 else 0
        ),
    }


def calculate_drawdown_metrics(equity_series: pd.Series) -> Dict[str, Any]:
    """
    Calculate drawdown-related metrics from an equity curve.

    Args:
        equity_series: Series of equity values indexed by timestamp

    Returns:
        Dictionary containing drawdown metrics and series
    """
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    return {"max_drawdown": drawdown.min(), "drawdown_series": drawdown}


def calculate_risk_metrics(
    returns: pd.Series, trading_days: int = 252
) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics from a return series.

    Args:
        returns: Series of period returns
        trading_days: Number of trading days per year (default: 252)

    Returns:
        Dictionary containing risk metrics
    """
    if returns.empty:
        return {
            "sortino_ratio": 0,
            "var_95": 0,
            "var_99": 0,
            "max_consecutive_losses": 0,
        }
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days)
    avg_return = returns.mean() * trading_days
    metrics = {
        "sortino_ratio": avg_return / downside_std if downside_std != 0 else 0,
        "var_95": returns.quantile(0.05),
        "var_99": returns.quantile(0.01),
        "max_consecutive_losses": (
            (returns < 0)
            .astype(int)
            .groupby((returns < 0).ne(returns.shift()).cumsum())
            .sum()
            .max()
        ),
    }

    return metrics


def calculate_trade_metrics(
    trades: Sequence, total_commission: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate trade-specific metrics from a sequence of trade objects.

    Args:
        trades: Sequence of trade objects with pnl, entry_time, and exit_time attributes
        total_commission: Optional total commission costs

    Returns:
        Dictionary containing trade metrics including win rate, profit factor,
        holding periods and P&L statistics
    """
    if not trades:
        return {
            "count": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
            "total_commission": total_commission or 0,
            "avg_holding_period": 0,
            "profit_factor": 0,
        }

    # For covered calls, P&L should be:
    # - For expired worthless: premium received - transaction costs
    # - For assigned: premium received - assignment cost - transaction costs
    trade_pnls = []
    for trade in trades:
        """
        if trade.instrument.instrument_type == InstrumentType.CALL_OPTION:
            # Calculate covered call P&L
            premium = abs(trade.quantity) * trade.entry_price * trade.instrument.get_multiplier()
            assignment_cost = (
                trade.exit_price * abs(trade.quantity) * trade.instrument.get_multiplier()
                if trade.assignment else 0
            )
            pnl = premium - assignment_cost - trade.transaction_costs
        else:
            # For other instruments (shouldn't occur in covered call strategy)
            pnl = (
                (trade.exit_price - trade.entry_price) *
                trade.quantity *
                trade.instrument.get_multiplier()
            ) - trade.transaction_costs
        """
        trade_pnls.append(trade.pnl)

    total_invested = sum(
        [
            abs(t.quantity) * t.entry_price * t.instrument.get_multiplier()
            for t in trades
        ]
    )
    return_on_invested_capital = (
        sum([t.pnl for t in trades]) / total_invested if total_invested != 0 else 0
    )
    wins = [pnl for pnl in trade_pnls if pnl > 0]
    losses = [pnl for pnl in trade_pnls if pnl <= 0]
    holding_periods = [(t.exit_time - t.entry_time).days for t in trades]
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

    return {
        "count": float(len(trades)),
        "win_rate": float(len(wins) / len(trades)),
        "avg_pnl": float(np.mean(trade_pnls)),
        "total_pnl": float(sum(trade_pnls)),
        "total_commission": float(total_commission or 0),
        "avg_holding_period": float(np.mean(holding_periods)),
        "profit_factor": float(profit_factor),
        # Additional metrics for covered call analysis
        "avg_premium": float(
            np.mean(
                [
                    abs(t.quantity) * t.entry_price * t.instrument.get_multiplier()
                    for t in trades
                    if t.instrument.instrument_type == InstrumentType.CALL_OPTION
                ]
            )
            if trades
            else 0
        ),
        "assignment_rate": float(
            len(
                [
                    t
                    for t in trades
                    if t.instrument.instrument_type == InstrumentType.CALL_OPTION
                    and t.assignment
                ]
            )
            / len(trades)
            if trades
            else 0
        ),
        "total_invested": float(total_invested),
        "return_on_invested_capital": float(return_on_invested_capital),
    }
