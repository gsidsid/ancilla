# ancilla/backtesting/results.py
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from ancilla.backtesting.instruments import InstrumentType

@dataclass
class BacktestResults:
    """Structured container for backtest results with analysis methods."""

    # Core metrics
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

    # Trade analysis
    options_metrics: Dict[str, Any]
    stock_metrics: Dict[str, Any]

    # Cost analysis
    transaction_costs: Dict[str, float]
    execution_metrics: Dict[str, Any]

    # Time series
    equity_curve: pd.DataFrame
    drawdown_series: pd.Series
    daily_returns: pd.Series

    # Raw data
    trades: List[Any]  # List of Trade objects

    @property
    def total_trades(self) -> int:
        """Get total number of trades."""
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    def plot_equity_curve(self, include_drawdown: bool = True) -> go.Figure:
        """Plot equity curve with optional drawdown overlay."""
        fig = go.Figure()

        # Add equity curve
        fig.add_trace(go.Scatter(
            x=self.equity_curve.index,
            y=self.equity_curve['equity'],
            name='Portfolio Value',
            line=dict(color='blue')
        ))

        if include_drawdown:
            # Add drawdown on secondary y-axis
            fig.add_trace(go.Scatter(
                x=self.drawdown_series.index,
                y=self.drawdown_series * 100,  # Convert to percentage
                name='Drawdown %',
                yaxis='y2',
                line=dict(color='red')
            ))

            fig.update_layout(
                yaxis2=dict(
                    title='Drawdown %',
                    overlaying='y',
                    side='right',
                    range=[0, max(abs(self.drawdown_series * 100))]
                )
            )

        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified'
        )

        return fig

    def analyze_options_performance(self) -> pd.DataFrame:
        """
        Analyze options trading performance.
        Returns a DataFrame with detailed trade metrics.
        """
        if not self.trades:
            return pd.DataFrame()

        options_trades = [t for t in self.trades if t.instrument.is_option]
        if not options_trades:
            return pd.DataFrame()

        trades_data = []
        for t in options_trades:
            # Use realized P&L instead of recalculating
            multiplier = t.instrument.get_multiplier()
            position_value = t.entry_price * abs(t.quantity) * multiplier
            return_pct = (t.realized_pnl / position_value) if position_value != 0 else 0

            trades_data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'duration': t.duration_hours,
                'option_type': t.instrument.instrument_type.value,
                'strike': t.instrument.strike,
                'expiration': t.instrument.expiration,
                'quantity': t.quantity,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'was_assigned': t.assignment if hasattr(t, 'assignment') else False,
                'pnl': t.realized_pnl,
                'return_pct': return_pct,
                'transaction_costs': t.transaction_costs
            })

        return pd.DataFrame(trades_data)

    def _extract_dte(self, entry_date: datetime, ticker: str) -> int:
        """Extract days to expiration at entry from option ticker."""
        try:
            # Parse expiration from O:TSLA230113C00015000 format
            date_part = ticker.split(':')[1][4:10]  # Extract YYMMDD
            expiry = datetime.strptime(f"20{date_part}", "%Y%m%d")
            return (expiry - entry_date).days
        except:
            return 0

    def risk_metrics(self) -> Dict[str, float]:
        """Calculate additional risk metrics."""
        metrics = {}

        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(self.daily_returns, 5)
        metrics['var_99'] = np.percentile(self.daily_returns, 1)

        # Conditional VaR (CVaR/Expected Shortfall)
        metrics['cvar_95'] = self.daily_returns[
            self.daily_returns <= metrics['var_95']
        ].mean()

        # Calmar Ratio (annual return / max drawdown)
        metrics['calmar_ratio'] = (
            self.annualized_return / abs(self.max_drawdown)
            if self.max_drawdown != 0 else 0
        )

        # Information Ratio (assuming risk-free rate of 0 for simplicity)
        excess_returns = self.daily_returns
        metrics['information_ratio'] = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() != 0 else 0
        )

        return metrics

    def summarize(self) -> str:
        """Generate a comprehensive performance summary."""
        risk_metrics = self.risk_metrics()

        # Separate trades by type
        options_trades = [t for t in self.trades if t.instrument.is_option]
        stock_trades = [t for t in self.trades if t.instrument.is_option is False]

        # Calculate options performance metrics
        if options_trades:
            calls = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.CALL_OPTION]
            puts = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.PUT_OPTION]
            long_options = [t for t in options_trades if t.quantity > 0]
            short_options = [t for t in options_trades if t.quantity < 0]

            # Use realized P&L from trades
            option_pnls = [t.realized_pnl for t in options_trades if hasattr(t, 'realized_pnl') and t.realized_pnl is not None]

        summary = [
            "Backtest Performance Summary",
            "=" * 50,
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital: ${self.final_capital:,.2f}",
            f"Total Return: {self.total_return:.2%}",
            f"Annualized Return: {self.annualized_return:.2%}",
            "",
            "Risk Metrics:",
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Sortino Ratio: {self.sortino_ratio:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"VaR (95%): {risk_metrics['var_95']:.2%}",
            f"CVaR (95%): {risk_metrics['cvar_95']:.2%}",
            "",
            "Trading Statistics:",
            f"Total Trades: {len(self.trades)}",
            f"Option Trades: {len(options_trades)}",
            f"Stock Trades: {len(stock_trades)}",
            f"Win Rate: {self.win_rate:.2%}",
            f"Average Trade Duration (Hours): {np.mean([t.duration_hours for t in self.trades]):.1f}",
        ]

        if options_trades:
            summary.extend([
                "",
                "Options Performance:",
                f"Total Option Trades: {len(options_trades)}",
                f"  - Calls: {len(calls)}",
                f"  - Puts: {len(puts)}",
                f"  - Long: {len(long_options)}",
                f"  - Short: {len(short_options)}",
                f"Average Option P&L: ${np.mean(option_pnls):.2f}",
                f"Total Option P&L: ${sum(option_pnls):.2f}",
                f"Option Win Rate: {len([p for p in option_pnls if p > 0])/len(option_pnls):.2%}",
                f"Assignment Rate: {len([t for t in options_trades if getattr(t, 'assignment', False)])/len(options_trades):.2%}"
            ])

        summary.extend([
            "",
            "Transaction Costs:",
            f"Total Costs: ${sum(t.transaction_costs for t in self.trades):,.2f}",
            f"Cost % of AUM: {sum(t.transaction_costs for t in self.trades)/self.initial_capital:.2%}"
        ])

        return "\n".join(summary)
