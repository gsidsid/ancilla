# ancilla/backtesting/results.py
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        """
        Plot equity curve with optional drawdown overlay, trade annotations,
        and current holdings in the hover tooltip.
        """
        # Create a figure with secondary y-axis for drawdown
        fig = make_subplots(specs=[[{"secondary_y": include_drawdown}]])

        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve['equity'],
                name='Portfolio Value',
                line=dict(color='blue', width=2),
                hoverinfo='text',
                hovertext=self._generate_hover_text()
            ),
            secondary_y=False,
        )

        # Add drawdown if requested
        if include_drawdown:
            fig.add_trace(
                go.Scatter(
                    x=self.drawdown_series.index,
                    y=self.drawdown_series * 100,  # Convert to percentage
                    name='Drawdown %',
                    line=dict(color='red', width=2, dash='dash'),
                    hoverinfo='text',
                    hovertext=self._generate_drawdown_hover_text()
                ),
                secondary_y=True,
            )
            fig.update_yaxes(
                title_text="Drawdown (%)",
                secondary_y=True,
                showgrid=False,
                range=[min(self.drawdown_series * 100) * 1.1, 0]
            )

        # Annotate trades
        trade_traces = self._create_trade_traces()
        for trade_trace in trade_traces:
            fig.add_trace(trade_trace, secondary_y=False)

        # Update layout with enhanced styling
        fig.update_layout(
            title={
                'text': "Portfolio Equity Curve",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            template='plotly_dark',  # Choose a professional template
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Update y-axis for equity curve
        fig.update_yaxes(
            showgrid=True,
            gridcolor='gray',
            gridwidth=0.5,
            zeroline=False,
            secondary_y=False
        )

        return fig

    def _create_trade_traces(self) -> List[go.Scatter]:
        """
        Create Plotly scatter traces for trades, differentiating between
        stock and option trades and buy/sell actions.
        """
        trade_traces = []
        for trade in self.trades:
            trade_time = trade.entry_time
            trade_type = 'Option' if trade.instrument.is_option else 'Stock'
            action = 'Buy' if trade.quantity > 0 else 'Sell'
            color = 'green' if action == 'Buy' else 'red'
            symbol = 'triangle-up' if action == 'Buy' else 'triangle-down'
            size = 10

            # Adjust symbol based on instrument type
            if trade.instrument.is_option:
                symbol = 'diamond' if action == 'Buy' else 'cross'

            trade_traces.append(
                go.Scatter(
                    x=[trade_time],
                    y=[self.equity_curve.loc[trade_time, 'equity']],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    name=f"{trade_type} {action}",
                    hoverinfo='text',
                    hovertext=self._generate_trade_hover_text(trade)
                )
            )
        return trade_traces

    def _generate_hover_text(self) -> List[str]:
        """
        Generate hover text for equity curve points, including current holdings.
        """
        hover_texts = []
        holdings = self._compute_holdings_over_time()
        for date, equity in self.equity_curve['equity'].items():
            holding_info = holdings.get(date, {})
            holdings_str = self._format_holdings(holding_info)
            hover_text = f"Date: {date.strftime('%Y-%m-%d')}<br>" \
                            f"Equity: ${equity:,.2f}<br>" \
                            f"Holdings:<br>{holdings_str}"
            hover_texts.append(hover_text)
        return hover_texts

    def _generate_drawdown_hover_text(self) -> List[str]:
        """
        Generate hover text for drawdown points.
        """
        hover_texts = []
        for date, drawdown in self.drawdown_series.items():
            hover_text = f"Date: {date.strftime('%Y-%m-%d')}<br>" \
                            f"Drawdown: {drawdown:.2f}%"
            hover_texts.append(hover_text)
        return hover_texts

    def _generate_trade_hover_text(self, trade: Any) -> str:
        """
        Generate hover text for individual trades.
        """
        trade_info = (
            f"Trade Type: {'Option' if trade.instrument.is_option else 'Stock'}<br>"
            f"Action: {'Buy' if trade.quantity > 0 else 'Sell'}<br>"
            f"Ticker: {trade.instrument.ticker}<br>"
            f"Quantity: {trade.quantity}<br>"
            f"Price: ${trade.entry_price:,.2f}<br>"
            f"P&L: ${trade.pnl:,.2f}"
        )
        return trade_info

    def _compute_holdings_over_time(self) -> Dict[pd.Timestamp, Dict[str, Any]]:
        """
        Compute current holdings at each date in the equity curve.
        Returns a dictionary mapping dates to holdings.
        """
        holdings = {}
        current_holdings = {}
        sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
        equity_dates = self.equity_curve.index

        trade_idx = 0
        num_trades = len(sorted_trades)

        for date in equity_dates:
            # Process all trades up to the current date
            while trade_idx < num_trades and sorted_trades[trade_idx].entry_time <= date:
                trade = sorted_trades[trade_idx]
                ticker = trade.instrument.ticker
                if ticker not in current_holdings:
                    current_holdings[ticker] = {'quantity': 0, 'instrument': trade.instrument}
                current_holdings[ticker]['quantity'] += trade.quantity
                # Remove the holding if quantity is zero
                if current_holdings[ticker]['quantity'] == 0:
                    del current_holdings[ticker]
                trade_idx += 1
            # Record current holdings
            holdings[date] = current_holdings.copy()
        return holdings

    def _format_holdings(self, holdings: Dict[str, Any]) -> str:
        """
        Format holdings dictionary into a readable string for hover text.
        """
        if not holdings:
            return "None"
        holdings_str = ""
        for ticker, info in holdings.items():
            instrument = info['instrument']
            quantity = info['quantity']
            if instrument.is_option:
                option_type = instrument.instrument_type.value
                strike = instrument.strike
                expiration = instrument.expiration.strftime('%Y-%m-%d')
                holdings_str += f"{ticker}: {quantity} {option_type} @ ${strike} Exp: {expiration}<br>"
            else:
                holdings_str += f"{ticker}: {quantity} shares<br>"
        return holdings_str


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
