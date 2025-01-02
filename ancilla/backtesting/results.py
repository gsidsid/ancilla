# ancilla/backtesting/results.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ancilla.backtesting.instruments import InstrumentType
from ancilla.formulae.metrics import (
    calculate_return_metrics, calculate_drawdown_metrics, calculate_risk_metrics,
    calculate_trade_metrics
)

@dataclass
class BacktestResults:
    """Structured container for backtest results with analysis methods."""
    strategy_name: str

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

    # Portfolio data
    net_pnl: float

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


    def prepare_sequential_data(self):
        """
        Prepare the equity data by resetting the index and creating a sequential index.
        Ensures that there is a 'datetime' column regardless of the original index name.
        Converts datetime from UTC to US/Eastern timezone.
        """
        # Ensure equity_curve is a DataFrame with an 'equity' column
        if isinstance(self.equity_curve, pd.Series):
            equity_df = self.equity_curve.to_frame(name='equity')
        elif isinstance(self.equity_curve, pd.DataFrame):
            if 'equity' not in self.equity_curve.columns:
                raise ValueError("The equity_curve DataFrame must contain an 'equity' column.")
            equity_df = self.equity_curve.copy()
        else:
            raise TypeError("equity_curve must be a pandas DataFrame or Series.")

        # Reset index to turn the datetime index into a column
        equity_df = equity_df.reset_index(drop=False)

        # Determine the name of the datetime column after reset
        datetime_col = equity_df.columns[0]  # Assumes the first column is datetime after reset

        # Rename the datetime column to 'datetime' for consistency
        if datetime_col != 'datetime':
            equity_df = equity_df.rename(columns={datetime_col: 'datetime'})

        # Convert 'datetime' to datetime type if not already
        if not pd.api.types.is_datetime64_any_dtype(equity_df['datetime']):
            equity_df['datetime'] = pd.to_datetime(equity_df['datetime'])

        # Localize to UTC if not timezone-aware, then convert to US/Eastern
        if equity_df['datetime'].dt.tz is None:
            equity_df['datetime'] = equity_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            equity_df['datetime'] = equity_df['datetime'].dt.tz_convert('US/Eastern')

        # Create a sequential integer index
        equity_df['sequential_index'] = equity_df.index

        return equity_df


    def plot_equity_curve(self, include_drawdown: bool = False) -> go.Figure:
            """
            Plot equity curve with optional drawdown overlay, trade annotations,
            and performance summary panel.
            """
            # Create figure with two subplots side by side
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.8, 0.13],
                specs=[[{"secondary_y": include_drawdown}, {"type": "table"}]],
                horizontal_spacing=0.06,
            )

            # Prepare equity curve data
            equity_df = self.prepare_sequential_data()

            # Prepare drawdown data
            drawdown_df = self._prepare_drawdown_data() if include_drawdown else None

            # Add equity curve
            fig.add_trace(
                go.Scatter(
                    x=equity_df['sequential_index'],
                    y=equity_df['equity'],
                    name='Portfolio Value',
                    line=dict(color='#FF9900', width=2),
                    mode='lines',
                    hoverinfo='text',
                    hovertext=self._generate_hover_text(equity_df['datetime'])
                ),
                row=1, col=1,
                secondary_y=False,
            )

            # Add drawdown if requested
            if include_drawdown and drawdown_df is not None and not drawdown_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_df['sequential_index'],
                        y=drawdown_df['drawdown'] * 100,
                        name='Drawdown %',
                        line=dict(color='#FF4444', width=1, dash='dot'),
                        mode='lines',
                        hoverinfo='text',
                        hovertext=[
                            f"Time: {dt.strftime('%Y-%m-%d %H:%M')}<br>Drawdown: {dd * 100:.2f}%"
                            for dt, dd in zip(drawdown_df['datetime'], drawdown_df['drawdown'])
                        ]
                    ),
                    row=1, col=1,
                    secondary_y=True,
                )

            # Add trade traces
            trade_traces = self._create_trade_traces()
            legend_entries = set()
            for trade_trace in trade_traces:
                # Check if this type of trade is already in legend
                trade_name = trade_trace.name
                if trade_name in legend_entries:
                    trade_trace.showlegend = False
                else:
                    legend_entries.add(trade_name)
                fig.add_trace(trade_trace, row=1, col=1, secondary_y=False)

            # Add performance summary table
            summary_data = self._prepare_summary_data()
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Metric', 'Value'],
                        fill_color='rgb(30, 30, 30)',
                        align='left',
                        font=dict(family="Arial", color='white', size=11)
                    ),
                    cells=dict(
                        values=list(zip(*summary_data)),
                        fill_color='rgb(10, 10, 10)',
                        align=['left', 'right'],
                        line_color='rgb(30, 30, 30)',
                        font=dict(family="Arial", color='white', size=10),
                        height=25
                    ),
                ),
                row=1, col=2,

            )

            # Update layout
            fig.update_layout(
                plot_bgcolor='black',
                paper_bgcolor='black',
                title={
                    'text': self.strategy_name,
                    'y': 0.95,
                    'x': 0.4,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(family="Arial", size=16, color='white')
                },
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.012,
                    xanchor="right",
                    x=0.75,
                    font=dict(family="Arial", size=10, color='white'),
                    bgcolor='rgba(0,0,0,0.5)'
                ),
                hovermode='x unified',
                margin=dict(l=100, r=0, t=80, b=100)
            )

            # Update axes for equity curve
            fig.update_xaxes(
                row=1, col=1,
                showgrid=True,
                gridcolor='#333333',
                gridwidth=1,
                griddash='dot',
                dtick=len(equity_df) // 20,  # Increased grid density
                tickfont=dict(size=10, color='gray'),
                tickangle=45,
                title_font=dict(size=11, color='gray'),
                title_text="Date"
            )

            fig.update_yaxes(
                row=1, col=1,
                secondary_y=False,
                showgrid=True,
                gridcolor='#333333',
                gridwidth=1,
                griddash='dot',
                dtick=self.final_capital / 20,  # Increased grid density
                tickfont=dict(family="Arial", size=10, color='gray'),
                title_font=dict(family="Arial", size=11, color='gray'),
                tickformat="$,.0f"
            )

            if include_drawdown:
                fig.update_yaxes(
                    row=1, col=1,
                    secondary_y=True,
                    showgrid=False,
                    gridcolor='#333333',
                    range=[drawdown_df['drawdown'].min() * 100 * 1.1, 0],
                    tickfont=dict(size=10, color='gray'),
                    title_font=dict(size=11, color='gray'),
                    tickformat=".1%"
                )

            return fig

    def _prepare_drawdown_data(self) -> Optional[pd.DataFrame]:
        """Prepare drawdown data for plotting."""
        if not hasattr(self, 'drawdown_series') or self.drawdown_series.empty:
            return None

        if isinstance(self.drawdown_series, pd.Series):
            drawdown_df = self.drawdown_series.to_frame(name='drawdown')
        else:
            if 'drawdown' not in self.drawdown_series.columns:
                return None
            drawdown_df = self.drawdown_series.copy()

        drawdown_df = drawdown_df.reset_index(drop=False)
        datetime_col = drawdown_df.columns[0]
        if datetime_col != 'datetime':
            drawdown_df = drawdown_df.rename(columns={datetime_col: 'datetime'})

        # Handle timezone conversion
        if not pd.api.types.is_datetime64_any_dtype(drawdown_df['datetime']):
            drawdown_df['datetime'] = pd.to_datetime(drawdown_df['datetime'])
        if drawdown_df['datetime'].dt.tz is None:
            drawdown_df['datetime'] = drawdown_df['datetime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
            drawdown_df['datetime'] = drawdown_df['datetime'].dt.tz_convert('US/Eastern')

        drawdown_df['sequential_index'] = drawdown_df.index
        return drawdown_df

    def _create_trade_traces(self) -> List[go.Scatter]:
        """
        Create Plotly scatter traces for trades with enhanced styling.
        """
        trade_traces = []
        equity_df = self.prepare_sequential_data()
        datetime_to_seq = dict(zip(equity_df['datetime'], equity_df['sequential_index']))

        for trade in self.trades:
            trade_time = trade.entry_time
            if trade_time not in datetime_to_seq:
                continue  # Skip trades outside trading hours

            seq_index = datetime_to_seq[trade_time]
            trade_type = 'Option' if trade.instrument.is_option else 'Stock'
            action = 'Buy' if trade.quantity > 0 else 'Sell'

            # markers
            if trade.instrument.is_option:
                color = '#00FF00' if action == 'Buy' else '#FF4444'  # Bright green/red for options
                symbol = 'diamond' if action == 'Buy' else 'diamond-cross'
                size = 12
            else:
                color = '#90EE90' if action == 'Buy' else '#FF6B6B'  # Softer green/red for stocks
                symbol = 'triangle-up' if action == 'Buy' else 'triangle-down'
                size = 10

            trade_traces.append(
                go.Scatter(
                    x=[seq_index],
                    y=[self.equity_curve.loc[trade_time, 'equity']],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    name=f"{trade_type} {action}",
                    hoverinfo='text',
                    hovertext=self._generate_trade_hover_text(trade)
                )
            )
        return trade_traces

    def _compute_holdings_over_time(self) -> Dict[pd.Timestamp, Dict[str, Any]]:
        """
        Compute current holdings at each date in the equity curve, excluding expired options.
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

                # Handle options
                if trade.instrument.is_option:
                    # Skip if option is already expired
                    if trade.instrument.expiration <= date:
                        trade_idx += 1
                        continue

                    ticker = trade.instrument.format_option_ticker()
                    if ticker not in current_holdings:
                        current_holdings[ticker] = {
                            'quantity': 0,
                            'instrument': trade.instrument,
                            'avg_price': 0,
                            'cost_basis': 0
                        }

                    position = current_holdings[ticker]
                    old_quantity = position['quantity']
                    new_quantity = old_quantity + trade.quantity

                    if abs(trade.quantity) > 0:
                        old_cost = position['avg_price'] * abs(old_quantity)
                        new_cost = trade.entry_price * abs(trade.quantity)
                        position['cost_basis'] = old_cost + new_cost
                        if new_quantity != 0:
                            position['avg_price'] = position['cost_basis'] / abs(new_quantity)

                    position['quantity'] = new_quantity

                    if new_quantity == 0:
                        del current_holdings[ticker]

                else:
                    # Handle stocks
                    if ticker not in current_holdings:
                        current_holdings[ticker] = {
                            'quantity': 0,
                            'instrument': trade.instrument,
                            'avg_price': 0,
                            'cost_basis': 0
                        }

                    position = current_holdings[ticker]
                    old_quantity = position['quantity']
                    new_quantity = old_quantity + trade.quantity

                    if trade.quantity > 0:
                        old_cost = position['avg_price'] * abs(old_quantity)
                        new_cost = trade.entry_price * trade.quantity
                        position['cost_basis'] = old_cost + new_cost
                        if new_quantity != 0:
                            position['avg_price'] = position['cost_basis'] / abs(new_quantity)

                    position['quantity'] = new_quantity

                    if new_quantity == 0:
                        del current_holdings[ticker]

                trade_idx += 1

            # Clean up expired options before recording holdings
            current_holdings = {
                ticker: info for ticker, info in current_holdings.items()
                if not info['instrument'].is_option or info['instrument'].expiration > date
            }

            # Record current holdings
            holdings[date] = current_holdings.copy()

        return holdings

    def _generate_trade_hover_text(self, trade: Any) -> str:
        """
        Generate hover text for individual trades.
        """
        trade_info = (
            f"Trade Type: {'Option' if trade.instrument.is_option else 'Stock'}<br>"
            f"Action: {'Buy' if trade.quantity > 0 else 'Sell'}<br>"
            f"Ticker: {trade.instrument.ticker}<br>"
            f"Quantity: {abs(trade.quantity)}<br>"
            f"Price: ${trade.entry_price:,.2f}<br>"
            f"P&L: ${trade.pnl:,.2f}"
        )
        if trade.instrument.is_option:
            trade_info += f"<br>Strike: ${trade.instrument.strike:,.2f}<br>"
            trade_info += f"Expiration: {trade.instrument.expiration.strftime('%Y-%m-%d')}"
        return trade_info

    def _generate_hover_text(self, dates: pd.DatetimeIndex) -> List[str]:
        """
        Generate hover text for equity curve points.
        """
        hover_texts = []
        holdings = self._compute_holdings_over_time()

        for date in dates:
            equity = self.equity_curve.loc[date, 'equity']
            holding_info = holdings.get(date, {})
            holdings_str = self._format_holdings(holding_info)
            hover_text = (
                f"Time: {date.strftime('%Y-%m-%d %H:%M')}<br>"
                f"Equity: ${equity:,.2f}<br>"
                f"Holdings:<br>{holdings_str}"
            )
            hover_texts.append(hover_text)
        return hover_texts

    def _generate_drawdown_hover_text(self, drawdown_series: pd.Series) -> List[str]:
        """
        Generate hover text for drawdown points.
        """
        hover_texts = []
        for date, drawdown in drawdown_series.items():
            hover_text = (
                f"Time: {date.strftime('%Y-%m-%d %H:%M')}<br>"
                f"Drawdown: {(drawdown * 100):.2f}%"
            )
            hover_texts.append(hover_text)
        return hover_texts

    def _format_holdings(self, holdings: Dict[str, Any]) -> str:
        """
        Format holdings dictionary into a readable string for hover text,
        including position details.
        """
        if not holdings:
            return "None"

        holdings_str = ""
        for ticker, info in holdings.items():
            instrument = info['instrument']
            quantity = info['quantity']
            avg_price = info['avg_price']

            if instrument.is_option:
                option_type = instrument.instrument_type.value
                strike = instrument.strike
                expiration = instrument.expiration.strftime('%Y-%m-%d')
                holdings_str += (
                    f"{instrument.ticker}: {quantity} {option_type} @ ${strike} "
                    f"(Avg: ${avg_price:.2f}) Exp: {expiration}<br>"
                )
            else:
                holdings_str += (
                    f"{ticker}: {quantity} shares @ ${avg_price:.2f}<br>"
                )
        return holdings_str

    def _is_market_hours(self, timestamp) -> bool:
        """
        Check if the given timestamp is during market hours (9:30 AM - 4:00 PM ET, weekdays).
        Works with both datetime and pandas Timestamp objects.
        """
        # Convert datetime to pandas Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        if timestamp.weekday() >= 5:  # Weekend
            return False

        minutes_since_midnight = timestamp.hour * 60 + timestamp.minute
        market_open = 9 * 60 + 30   # 9:30 AM
        market_close = 16 * 60      # 4:00 PM

        return market_open <= minutes_since_midnight <= market_close

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
        stock_trades = [t for t in self.trades if not t.instrument.is_option]

        # Calculate options performance metrics
        if options_trades:
            calls = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.CALL_OPTION]
            puts = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.PUT_OPTION]
            long_options = [t for t in options_trades if t.quantity > 0]
            short_options = [t for t in options_trades if t.quantity < 0]

            # Use realized P&L from closing trades
            option_pnls = [t.pnl for t in options_trades]

        summary = [
            "",
            self.strategy_name + " – performance",
            "=" * 50,
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital: ${self.final_capital:,.2f}",
            f"Net P&L: ${self.final_capital - self.initial_capital:,.2f}",
            f"Total Return: {(self.final_capital - self.initial_capital) / self.initial_capital:.2%}",
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
                "",
                f"Average Option P&L: ${np.mean(option_pnls):.2f}",
                f"Total Option P&L: ${sum(option_pnls):.2f}",
                f"Option Win Rate: {len([p for p in option_pnls if p > 0])/len(option_pnls):.2%}",
                f"Assignment Rate: {len([t for t in options_trades if getattr(t, 'assignment', False)])/len(options_trades):.2%}"
            ])
        return "\n".join(summary)

    def _prepare_summary_data(self) -> List[List[str]]:
        """Prepare summary data for the performance panel."""
        risk_metrics = self.risk_metrics()
        options_trades = [t for t in self.trades if t.instrument.is_option]
        stock_trades = [t for t in self.trades if not t.instrument.is_option]

        # Calculate options metrics if applicable
        options_metrics = {}
        if options_trades:
            calls = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.CALL_OPTION]
            puts = [t for t in options_trades if t.instrument.instrument_type == InstrumentType.PUT_OPTION]
            option_pnls = [t.pnl for t in options_trades]
            options_metrics.update({
                'Calls/Puts': f"{len(calls)}/{len(puts)}",
                'Option P&L': f"${sum(option_pnls):,.2f}",
                'Option Win Rate': f"{len([p for p in option_pnls if p > 0])/len(option_pnls):.1%}"
            })

        # Prepare summary data
        metrics = [
            ['Performance', ''],  # Section header
            ['Final Capital', f"${self.final_capital:,.2f}"],
            ['Net P&L', f"${self.net_pnl:,.2f}"],
            ['Total Return', f"{self.net_pnl / self.initial_capital:.1%}"],
            ['Ann. Return', f"{self.annualized_return:.1%}"],
            ['', ''],  # Spacing
            ['Risk Metrics', ''],  # Section header
            ['Sharpe Ratio', f"{self.sharpe_ratio:.2f}"],
            ['Sortino Ratio', f"{self.sortino_ratio:.2f}"],
            ['Max Drawdown', f"{self.max_drawdown:.1%}"],
            ['VaR (95%)', f"{risk_metrics['var_95']:.1%}"],
            ['', ''],  # Spacing
            ['Trading Stats', ''],  # Section header
            ['Total Trades', str(len(self.trades))],
            ['Win Rate', f"{self.win_rate:.1%}"],
            ['Avg Duration', f"{np.mean([t.duration_hours for t in self.trades]):.1f}h"],
        ]

        # Add options metrics if present
        if options_metrics:
            metrics.extend([
                ['', ''],  # Spacing
                ['Options', ''],  # Section header
            ])
            metrics.extend([[k, v] for k, v in options_metrics.items()])

        return metrics

    @staticmethod
    def calculate(engine) -> "BacktestResults":
        """Calculate comprehensive backtest results."""
        # Convert equity curve to dataframe
        equity_curve_dict = {'timestamp': [], 'equity': []}
        for timestamp, equity in engine.portfolio.equity_curve:
            equity_curve_dict['timestamp'].append(timestamp)
            equity_curve_dict['equity'].append(equity)

        equity_df = pd.DataFrame.from_dict(equity_curve_dict).set_index('timestamp')
        equity_df['returns'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['returns'].dropna()

        # Calculate metrics using the new formulae module
        return_metrics = calculate_return_metrics(pd.Series(equity_df['equity']))
        drawdown_metrics = calculate_drawdown_metrics(pd.Series(equity_df['equity']))
        risk_metrics = calculate_risk_metrics(pd.Series(daily_returns))

        # Separate trades by type
        options_trades = [t for t in engine.portfolio.trades if t.instrument.is_option]
        stock_trades = [t for t in engine.portfolio.trades if not t.instrument.is_option]

        # Calculate trade metrics
        options_metrics = calculate_trade_metrics(options_trades)
        stock_metrics = calculate_trade_metrics(stock_trades)

        # Calculate total transaction costs
        # Since transaction costs are already subtracted in trade.pnl, avoid double-counting
        total_commission = sum(engine.commission_costs)
        total_slippage = sum(engine.slippage_costs)
        total_transaction_costs = sum(engine.total_transaction_costs)

        # Calculate net opening cash flows
        net_opening_cash_flows = sum(engine.portfolio.opening_cash_flows)

        # Calculate realized P&L
        realized_pnl = sum(t.pnl for t in engine.portfolio.trades)  # Already includes commissions

        # Realized P&L already includes transaction costs
        expected_final_capital = engine.initial_capital + realized_pnl
        actual_final_capital = engine.portfolio.cash + engine.portfolio.get_position_value()

        # Compare with actual final capital
        if not np.isclose(expected_final_capital, actual_final_capital, atol=1e-2):
            engine.logger.warning(
                f"Discrepancy detected:\n"
                f"Expected Final Capital: {expected_final_capital}\n"
                f"Actual Final Capital: {actual_final_capital}\n"
                f"Debug Info:\n "
                f"Realized PnL: {realized_pnl}\n"
                f"Initial Capital: {engine.initial_capital}\n"
                f"Total Transaction Costs: {total_transaction_costs}"
                f"Net Opening Cash Flows: {net_opening_cash_flows}"
                f"Total Commission: {total_commission}\n"
                f"Total Slippage: {total_slippage}"
            )

        # Compile results
        results = {
            'initial_capital': engine.initial_capital,
            'final_capital': actual_final_capital,
            'net_pnl': actual_final_capital - engine.initial_capital,
            **return_metrics,
            **drawdown_metrics,
            'options_metrics': options_metrics,
            'stock_metrics': stock_metrics,
            'transaction_costs': {
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'avg_commission_per_trade': (
                    total_commission / len(engine.portfolio.trades)
                    if engine.portfolio.trades else 0
                ),
                'avg_slippage_per_trade': (
                    total_slippage / len(engine.portfolio.trades)
                    if engine.portfolio.trades else 0
                ),
                'cost_as_pct_aum': (
                    (total_commission + total_slippage) / engine.initial_capital
                    if engine.initial_capital > 0 else 0
                )
            },
            'execution_metrics': {
                'fill_ratio': np.mean(engine.fill_ratios) if engine.fill_ratios else 0,
                'daily_metrics': {
                    'avg_slippage': np.mean(engine.daily_metrics['slippage']),
                    'avg_commission': np.mean(engine.daily_metrics['commissions']),
                    'avg_fill_ratio': np.mean(engine.daily_metrics['fills']),
                    'avg_volume_participation': np.mean(engine.daily_metrics['volume_participation'])
                }
            },
            'equity_curve': equity_df,
            'daily_returns': daily_returns,
            'trade_count': len(engine.portfolio.trades)
        }

        results.update(risk_metrics)

        return BacktestResults(
            strategy_name=engine.strategy.name,
            initial_capital=engine.initial_capital,
            final_capital=results['final_capital'],
            total_return=results['total_return'],
            annualized_return=results['annualized_return'],
            annualized_volatility=results['annualized_volatility'],
            sharpe_ratio=results['sharpe_ratio'],
            sortino_ratio=results['sortino_ratio'],
            max_drawdown=results['max_drawdown'],
            options_metrics=results['options_metrics'],
            stock_metrics=results['stock_metrics'],
            transaction_costs=results['transaction_costs'],
            execution_metrics=results['execution_metrics'],
            equity_curve=results['equity_curve'],
            drawdown_series=results['drawdown_series'],
            daily_returns=results['daily_returns'],
            net_pnl=results['net_pnl'],
            trades=engine.portfolio.trades
        )