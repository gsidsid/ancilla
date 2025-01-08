# ancilla/backtesting/results.py
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ancilla.models import InstrumentType
from ancilla.formulae.metrics import (
    calculate_return_metrics,
    calculate_drawdown_metrics,
    calculate_risk_metrics,
    calculate_trade_metrics,
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

    # Payouts
    total_interest: float
    total_dividends: float

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

    def prepare_timeseries_data(
        self, data: Union[pd.DataFrame, pd.Series], value_column: str
    ) -> pd.DataFrame:
        """
        Prepare timeseries data for plotting by standardizing format and timezone.

        Args:
            data: Input DataFrame or Series containing the timeseries data
            value_column: Name of the column to use for values if data is a Series

        Returns:
            Processed DataFrame with standardized datetime index and sequential index,
            or None if data is empty or invalid
        """
        # Handle empty or invalid input
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise TypeError("Data must exist")

        # Convert Series to DataFrame if necessary
        if isinstance(data, pd.Series):
            df = data.to_frame(name=value_column)
        elif isinstance(data, pd.DataFrame):
            if value_column not in data.columns:
                raise TypeError(f"{value_column} not found in data")
            df = data.copy()
        else:
            raise TypeError("data must be a pandas DataFrame or Series")

        # Reset index to turn the datetime index into a column
        df = df.reset_index(drop=False)

        # Standardize datetime column name
        datetime_col = df.columns[0]  # Assumes first column is datetime after reset
        if datetime_col != "datetime":
            df = df.rename(columns={datetime_col: "datetime"})

        # Convert to datetime type if needed
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])

        # Handle timezone conversion
        if df["datetime"].dt.tz is None:
            df["datetime"] = (
                df["datetime"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
            )
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("US/Eastern")

        # Add sequential index
        df["sequential_index"] = df.index

        return df

    def plot(self, include_drawdown: bool = False) -> go.Figure:
        """
        Plot equity curve with optional drawdown overlay, trade annotations,
        and performance summary panel.
        """
        # Create figure with two subplots side by side
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.8, 0.13],
            specs=[[{"secondary_y": include_drawdown}, {"type": "table"}]],
            horizontal_spacing=0.06,
        )

        # For equity data
        equity_df = self.prepare_timeseries_data(self.equity_curve, "equity")

        # For drawdown data
        drawdown_df = (
            self.prepare_timeseries_data(self.drawdown_series, "drawdown")
            if include_drawdown
            else None
        )

        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df["sequential_index"],
                y=equity_df["equity"],
                name="Portfolio Value",
                line=dict(color="#FF9900", width=2),
                mode="lines",
                hoverinfo="text",
                hovertext=self._generate_hover_text(equity_df["datetime"]),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

        # Add drawdown if requested
        if include_drawdown and drawdown_df is not None and not drawdown_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=drawdown_df["sequential_index"],
                    y=drawdown_df["drawdown"] * 100,
                    name="Drawdown %",
                    line=dict(color="#FF4444", width=1, dash="dot"),
                    mode="lines",
                    hoverinfo="text",
                    hovertext=[
                        f"Time: {dt.strftime('%Y-%m-%d %H:%M')}<br>Drawdown: {dd * 100:.2f}%"
                        for dt, dd in zip(
                            drawdown_df["datetime"], drawdown_df["drawdown"]
                        )
                    ],
                ),
                row=1,
                col=1,
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
                    values=["Metric", "Value"],
                    fill_color="rgb(30, 30, 30)",
                    align="left",
                    font=dict(family="Arial", color="white", size=11),
                ),
                cells=dict(
                    values=list(zip(*summary_data)),
                    fill_color="rgb(10, 10, 10)",
                    align=["left", "right"],
                    line_color="rgb(30, 30, 30)",
                    font=dict(family="Arial", color="white", size=10),
                    height=25,
                ),
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            plot_bgcolor="black",
            paper_bgcolor="black",
            title={
                "text": self.strategy_name,
                "y": 0.95,
                "x": 0.4,
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(family="Arial", size=16, color="white"),
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.010,
                xanchor="right",
                x=0.75,
                font=dict(family="Arial", size=10, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
            ),
            hovermode="x unified",
            margin=dict(l=100, r=0, t=80, b=100),
        )

        # Update axes for equity curve
        fig.update_xaxes(
            row=1,
            col=1,
            showgrid=True,
            gridcolor="#333333",
            gridwidth=1,
            griddash="dot",
            dtick=len(equity_df) // 20,  # Increased grid density
            tickfont=dict(size=10, color="gray"),
            tickangle=45,
            title_font=dict(size=11, color="gray"),
            title_text="Date",
        )

        fig.update_yaxes(
            row=1,
            col=1,
            secondary_y=False,
            showgrid=True,
            gridcolor="#333333",
            gridwidth=1,
            griddash="dot",
            dtick=self.final_capital / 20,  # Increased grid density
            tickfont=dict(family="Arial", size=10, color="gray"),
            title_font=dict(family="Arial", size=11, color="gray"),
            tickformat="$,.0f",
        )

        if include_drawdown:
            fig.update_yaxes(
                row=1,
                col=1,
                secondary_y=True,
                showgrid=False,
                gridcolor="#333333",
                range=[drawdown_df["drawdown"].min() * 100 * 1.1, 0],
                tickfont=dict(size=10, color="gray"),
                title_font=dict(size=11, color="gray"),
                tickformat=".1%",
            )

        fig.show()

        return fig

    def _create_trade_traces(self) -> List[go.Scatter]:
        """
        Create Plotly scatter traces for trades with enhanced styling.
        """
        trade_traces = []
        equity_df = self.prepare_timeseries_data(self.equity_curve, "equity")
        datetime_to_seq = dict(
            zip(equity_df["datetime"], equity_df["sequential_index"])
        )

        for trade in self.trades:
            trade_time = trade.entry_time
            if trade_time not in datetime_to_seq:
                continue  # Skip trades outside trading hours

            seq_index = datetime_to_seq[trade_time]
            trade_type = "Option" if trade.instrument.is_option else "Stock"
            action = "Buy" if trade.quantity > 0 else "Sell"

            # markers
            if trade.instrument.is_option:
                color = (
                    "#00FF00" if action == "Buy" else "#FF4444"
                )  # Bright green/red for options
                symbol = "diamond" if action == "Buy" else "diamond-cross"
                size = 10
            else:
                color = (
                    "#90EE90" if action == "Buy" else "#FF6B6B"
                )  # Softer green/red for stocks
                symbol = "triangle-up" if action == "Buy" else "triangle-down"
                size = 11

            trade_traces.append(
                go.Scatter(
                    x=[seq_index],
                    y=[self.equity_curve.loc[trade_time, "equity"]],
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                        line=dict(width=1, color="white"),
                    ),
                    name=f"{trade_type} {action}",
                    hoverinfo="text",
                    hovertext=self._generate_trade_hover_text(trade),
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
            while (
                trade_idx < num_trades and sorted_trades[trade_idx].entry_time <= date
            ):
                trade = sorted_trades[trade_idx]
                ticker = trade.instrument.ticker

                # Handle options
                if trade.instrument.is_option:
                    # Skip if option is already expired
                    if trade.instrument.expiration <= date or trade.exit_time <= date:
                        trade_idx += 1
                        continue

                    ticker = trade.instrument.format_option_ticker()
                    if ticker not in current_holdings:
                        current_holdings[ticker] = {
                            "quantity": 0,
                            "instrument": trade.instrument,
                            "avg_price": 0,
                            "cost_basis": 0,
                        }

                    position = current_holdings[ticker]
                    old_quantity = position["quantity"]
                    new_quantity = old_quantity + trade.quantity

                    if abs(trade.quantity) > 0:
                        old_cost = position["avg_price"] * abs(old_quantity)
                        new_cost = trade.entry_price * abs(trade.quantity)
                        position["cost_basis"] = old_cost + new_cost
                        if new_quantity != 0:
                            position["avg_price"] = position["cost_basis"] / abs(
                                new_quantity
                            )

                    position["quantity"] = new_quantity

                    if new_quantity == 0:
                        del current_holdings[ticker]

                else:
                    # Handle stocks
                    if ticker not in current_holdings:
                        current_holdings[ticker] = {
                            "quantity": 0,
                            "instrument": trade.instrument,
                            "avg_price": 0,
                            "cost_basis": 0,
                        }

                    position = current_holdings[ticker]
                    old_quantity = position["quantity"]
                    new_quantity = old_quantity + trade.quantity

                    if trade.quantity > 0:
                        old_cost = position["avg_price"] * abs(old_quantity)
                        new_cost = trade.entry_price * trade.quantity
                        position["cost_basis"] = old_cost + new_cost
                        if new_quantity != 0:
                            position["avg_price"] = position["cost_basis"] / abs(
                                new_quantity
                            )

                    position["quantity"] = new_quantity

                    if new_quantity == 0 or trade.assignment:
                        del current_holdings[ticker]

                trade_idx += 1

            # Clean up expired options before recording holdings
            current_holdings = {
                ticker: info
                for ticker, info in current_holdings.items()
                if not info["instrument"].is_option
                or info["instrument"].expiration > date
            }

            # Record current holdings
            holdings[date] = current_holdings.copy()

        return holdings

    def _generate_trade_hover_text(self, trade: Any) -> str:
        """
        Generate hover text for individual trades.
        """
        trade_info = (
            f"{abs(trade.quantity)} {trade.instrument.ticker} {str(trade.instrument.instrument_type).split('_')[0] + '(s)' if trade.instrument.is_option else 'share(s)'} @ ${trade.entry_price:,.2f}<br>"
            f"P&L: ${trade.pnl:,.2f}"
        )
        if trade.instrument.is_option:
            trade_info += f"<br>Strike: ${trade.instrument.strike:,.2f}<br>"
            trade_info += f"Expires {trade.instrument.expiration.strftime('%Y-%m-%d')}"
        return trade_info

    def _generate_hover_text(self, dates: pd.DatetimeIndex) -> List[str]:
        """
        Generate hover text for equity curve points.
        """
        hover_texts = []
        holdings = self._compute_holdings_over_time()

        for date in dates:
            equity = self.equity_curve.loc[date, "equity"]
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
            instrument = info["instrument"]
            quantity = info["quantity"]
            avg_price = info["avg_price"]

            if instrument.is_option:
                option_type = instrument.instrument_type.value
                strike = instrument.strike
                expiration = instrument.expiration.strftime("%Y-%m-%d")
                holdings_str += (
                    f"{instrument.ticker}: {quantity} {option_type} @ ${strike} "
                    f"(Avg: ${avg_price:.2f}) Exp: {expiration}<br>"
                )
            else:
                holdings_str += f"{ticker}: {quantity} shares @ ${avg_price:.2f}<br>"
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
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60  # 4:00 PM

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

            trades_data.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "duration": t.duration_hours,
                    "option_type": t.instrument.instrument_type.value,
                    "strike": t.instrument.strike,
                    "expiration": t.instrument.expiration,
                    "quantity": t.quantity,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "was_assigned": t.assignment if hasattr(t, "assignment") else False,
                    "pnl": t.realized_pnl,
                    "return_pct": return_pct,
                    "transaction_costs": t.transaction_costs,
                }
            )

        return pd.DataFrame(trades_data)

    def _extract_dte(self, entry_date: datetime, ticker: str) -> int:
        """Extract days to expiration at entry from option ticker.

        Args:
            entry_date: Date of entry
            ticker: Option ticker in format O:AAPL240126C00190000

        Returns:
            Number of days between entry_date and option expiration
        """
        try:
            # Parse expiration from an option ticker
            parts = ticker.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid ticker format: {ticker}")
            symbol_part = parts[1]
            date_part = symbol_part[4:10]
            expiry = datetime.strptime(f"20{date_part}", "%Y%m%d")
            if entry_date.tzinfo is not None:
                expiry = expiry.replace(tzinfo=entry_date.tzinfo)
            elif expiry.tzinfo is not None:
                expiry = expiry.replace(tzinfo=None)
            days = (expiry - entry_date).days
            return days

        except Exception as e:
            print(f"Error processing ticker {ticker}: {str(e)}")
            return 0

    def risk_metrics(self) -> Dict[str, float]:
        """Calculate additional risk metrics."""
        metrics = {}

        # Value at Risk (VaR)
        metrics["var_95"] = np.percentile(self.daily_returns, 5)
        metrics["var_99"] = np.percentile(self.daily_returns, 1)

        # Conditional VaR (CVaR/Expected Shortfall)
        metrics["cvar_95"] = self.daily_returns[
            self.daily_returns <= metrics["var_95"]
        ].mean()

        # Calmar Ratio (annual return / max drawdown)
        metrics["calmar_ratio"] = (
            self.annualized_return / abs(self.max_drawdown)
            if self.max_drawdown != 0
            else 0
        )

        # Information Ratio (assuming risk-free rate of 0 for simplicity)
        excess_returns = self.daily_returns
        metrics["information_ratio"] = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            if excess_returns.std() != 0
            else 0
        )

        return metrics

    def summary(self, summarize_trades: bool = True) -> str:
        """Generate a comprehensive performance summary."""
        risk_metrics = self.risk_metrics()

        # Separate trades by type
        options_trades = [t for t in self.trades if t.instrument.is_option]
        stock_trades = [t for t in self.trades if not t.instrument.is_option]

        total_invested = sum(
            [
                t.quantity * t.entry_price * t.instrument.get_multiplier()
                for t in self.trades
                if t.quantity > 0
            ]
        )
        return_on_invested_capital = (
            sum([t.pnl for t in self.trades]) / total_invested
            if total_invested != 0
            else 0
        )

        # Calculate options performance metrics
        if options_trades:
            calls = [
                t
                for t in options_trades
                if t.instrument.instrument_type == InstrumentType.CALL_OPTION
            ]
            puts = [
                t
                for t in options_trades
                if t.instrument.instrument_type == InstrumentType.PUT_OPTION
            ]
            long_options = [t for t in options_trades if t.quantity > 0]
            short_options = [t for t in options_trades if t.quantity < 0]

            # Use realized P&L from closing trades
            option_pnls = [t.pnl for t in options_trades]

        summary = [
            "\n",
            self.strategy_name + " – performance",
            "=" * 50,
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Capital: ${self.final_capital:,.2f}",
            f"Net P&L: ${self.final_capital - self.initial_capital:,.2f}",
            f"Total Return: {(self.final_capital - self.initial_capital) / self.initial_capital:.2%}",
            f"Annualized Return: {self.annualized_return:.2%}",
            f"ROI: {return_on_invested_capital:.2%}",
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
            summary.extend(
                [
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
                    f"Assignment Rate: {len([t for t in options_trades if getattr(t, 'assignment', False)])/len(options_trades):.2%}",
                ]
            )

        if summarize_trades:
            trade_summary = "\n=====================TRADES========================\n"
            # sort trades by entry time
            self.trades.sort(key=lambda trade: trade.entry_time)

            for trade in self.trades:
                # Extract trade metrics
                metrics = trade.get_metrics()
                quantity = metrics.get("quantity", 0)
                trade_type = metrics.get("type", "unknown").lower()
                ticker = metrics.get("ticker", "UNKNOWN")
                entry_time = metrics.get("entry_time")
                exit_time = metrics.get("exit_time")
                entry_price = metrics.get("entry_price", 0.0)
                exit_price = metrics.get("exit_price", 0.0)
                pnl = metrics.get("pnl", 0.0)
                assignment = metrics.get("assignment", False)
                exercised = metrics.get("exercised", False)
                expiration = metrics.get("expiration", None)
                option_type = str(metrics.get("option_type", None))
                strike = metrics.get("strike", 0.0)
                option_ticker = metrics.get("option_ticker", "")
                duration_hours = metrics.get("duration_hours", 0)

                # Determine if the trade is an option or shares
                is_option = trade_type == "option"

                # Format the instrument description
                if is_option:
                    option_side = "long" if quantity > 0 else "short"
                    option_kind = (
                        option_type.replace("_option", "") if option_type else "unknown"
                    )
                    instrument = f"{ticker} {option_side} {option_kind} option"
                    dte = self._extract_dte(entry_time, option_ticker)
                    instrument += f" (Strike: ${strike:.2f}, Expiry: {dte} days)"
                else:
                    instrument = f"{abs(quantity)} shares of {ticker}"

                # Format timestamps
                def format_time(timestamp):
                    if not timestamp:
                        return "N/A"
                    ts = pd.Timestamp(timestamp)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC").tz_convert("US/Eastern")
                    else:
                        ts = ts.tz_convert("US/Eastern")
                    return ts.strftime("%Y-%m-%d %H:%M")

                formatted_entry_time = format_time(entry_time)
                formatted_exit_time = format_time(exit_time) if exit_time else "N/A"

                # Determine entry action
                entry_action = "Bought" if quantity > 0 else "Sold"

                # Format P&L
                pnl_status = "profit" if pnl > 0 else "loss"
                pnl_str = f"${abs(pnl):,.2f}"

                # Start building trade description
                trade_desc = f"{entry_action} {instrument} at ${entry_price:.2f} on {formatted_entry_time},\n\t"

                # Handle exit description
                if exit_time:
                    if is_option:
                        if quantity < 0:  # Short option
                            if assignment:
                                exit_action = (
                                    f"expired ITM and was assigned at ${strike:.2f}"
                                )
                            else:
                                exit_action = "expired OTM"
                            trade_desc += f"{exit_action} on {formatted_exit_time} with a {pnl_status} of {pnl_str}"
                        else:  # Long option
                            if exercised:
                                exit_action = f"exercised at ${strike:.2f}"
                            else:
                                if exit_time and exit_time < expiration:  # Early close
                                    exit_action = f"closed at ${exit_price:.2f}"
                                else:  # True expiration
                                    exit_action = "expired worthless"
                            trade_desc += f"{exit_action} on {formatted_exit_time} with a {pnl_status} of {pnl_str}"
                    else:  # Stock trades
                        if assignment:
                            exit_action = "called away" if quantity > 0 else "assigned"
                            trade_desc += f"{exit_action} at ${exit_price:.2f}"
                        else:
                            exit_action = "closed"
                            trade_desc += f"{exit_action} at ${exit_price:.2f}"

                        trade_desc += f" on {formatted_exit_time} with a {pnl_status} of {pnl_str}"

                    # Add duration for closed trades
                    if exit_time:
                        duration_days = duration_hours / 24
                        trade_desc += f" (held for {duration_days:.1f} days)"
                else:
                    trade_desc += "position still open"

                # Append the trade description
                trade_summary += trade_desc + "\n"

            summary.extend([trade_summary])

        return "\n".join(summary)

    def _prepare_summary_data(self) -> List[List[str]]:
        """Prepare summary data for the performance panel."""
        risk_metrics = self.risk_metrics()
        options_trades = [t for t in self.trades if t.instrument.is_option]
        option_pnls = [t.pnl for t in options_trades]

        total_invested = sum(
            [
                t.quantity * t.entry_price * t.instrument.get_multiplier()
                for t in self.trades
                if t.quantity > 0
            ]
        )
        return_on_invested_capital = (
            sum([t.pnl for t in self.trades]) / total_invested
            if total_invested != 0
            else 0
        )

        # Calculate options metrics if applicable
        options_metrics = {}
        if options_trades:
            calls = [
                t
                for t in options_trades
                if t.instrument.instrument_type == InstrumentType.CALL_OPTION
            ]
            puts = [
                t
                for t in options_trades
                if t.instrument.instrument_type == InstrumentType.PUT_OPTION
            ]
            option_pnls = [t.pnl for t in options_trades]
            options_metrics.update(
                {
                    "Calls/Puts": f"{len(calls)}/{len(puts)}",
                    "Option P&L": f"${sum(option_pnls):,.2f}",
                    "Option Win Rate": f"{len([p for p in option_pnls if p > 0])/len(option_pnls):.1%}",
                    "Assignment Rate": f"{len([t for t in options_trades if getattr(t, 'assignment', False)])/len(options_trades):.1%}",
                    "Avg Option Duration": f"{np.mean([t.duration_hours for t in options_trades]):.1f}h",
                }
            )

        # Prepare summary data
        metrics = [
            ["Final Capital", f"${self.final_capital:,.2f}"],
            ["Net P&L", f"${self.net_pnl:,.2f}"],
            ["Total Return", f"{self.net_pnl / self.initial_capital:.1%}"],
            ["Ann. Return", f"{self.annualized_return:.1%}"],
            ["ROI", f"{return_on_invested_capital:.1%}"],
            ["", ""],  # Section header
            ["Sharpe Ratio", f"{self.sharpe_ratio:.2f}"],
            ["Sortino Ratio", f"{self.sortino_ratio:.2f}"],
            ["Max Drawdown", f"{self.max_drawdown:.1%}"],
            ["VaR (95%)", f"{risk_metrics['var_95']:.1%}"],
            ["", ""],
            ["Total Trades", str(len(self.trades))],
            ["Win Rate", f"{self.win_rate:.1%}"],
            [
                "Avg Duration",
                f"{np.mean([t.duration_hours for t in self.trades]):.1f}h",
            ],
            ["", ""],
            ["Interest", f"+ ${self.total_interest:,.2f}"],
            ["Dividends", f"+ ${self.total_dividends:,.2f}"],
        ]

        # Add options metrics if present
        if options_metrics:
            metrics.extend(
                [
                    ["", ""],  # Spacing
                    ["Options", ""],  # Section header
                ]
            )
            metrics.extend([[k, v] for k, v in options_metrics.items()])

        return metrics

    @staticmethod
    def calculate(engine) -> "BacktestResults":
        """Calculate comprehensive backtest results."""
        # Convert equity curve to dataframe
        equity_curve_dict = {"timestamp": [], "equity": []}
        for timestamp, equity in engine.portfolio.equity_curve:
            equity_curve_dict["timestamp"].append(timestamp)
            equity_curve_dict["equity"].append(equity)

        equity_df = pd.DataFrame.from_dict(equity_curve_dict).set_index("timestamp")
        equity_df["returns"] = equity_df["equity"].pct_change()
        daily_returns = equity_df["returns"].dropna()

        # Calculate metrics using the new formulae module
        return_metrics = calculate_return_metrics(pd.Series(equity_df["equity"]))
        drawdown_metrics = calculate_drawdown_metrics(pd.Series(equity_df["equity"]))
        risk_metrics = calculate_risk_metrics(pd.Series(daily_returns))

        # Separate trades by type
        options_trades = [t for t in engine.portfolio.trades if t.instrument.is_option]
        stock_trades = [
            t for t in engine.portfolio.trades if not t.instrument.is_option
        ]

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

        # Calculate total dividend and interest payouts
        total_dividend_payouts = sum(engine.dividend_payouts)
        total_interest_payouts = sum(engine.interest_payouts)

        # Calculate realized P&L
        realized_pnl = sum(
            t.pnl for t in engine.portfolio.trades
        )  # Already includes commissions

        # Realized P&L already includes transaction costs
        expected_final_capital = (
            engine.initial_capital
            + realized_pnl
            + total_dividend_payouts
            + total_interest_payouts
        )
        actual_final_capital = (
            engine.portfolio.cash + engine.portfolio.get_position_value()
        )

        # Compare with actual final capital
        if not np.isclose(expected_final_capital, actual_final_capital, atol=1e-2):
            engine.logger.warning(
                f"Discrepancy detected during trade reconciliation:\n"
                f"\tExpected Final Capital: {expected_final_capital}\n"
                f"\tActual Final Capital: {actual_final_capital}\n"
                f"\nDebug Info:\n "
                f"\tRealized PnL: {realized_pnl}\n"
                f"\tInitial Capital: {engine.initial_capital}\n"
                f"\tTotal Transaction Costs: {total_transaction_costs}\n"
                f"\tNet Opening Cash Flows: {net_opening_cash_flows}\n"
                f"\tTotal Commission: {total_commission}\n"
                f"\tTotal Slippage: {total_slippage}\n"
            )

        # Compile results
        results = {
            "initial_capital": engine.initial_capital,
            "final_capital": actual_final_capital,
            "net_pnl": actual_final_capital - engine.initial_capital,
            **return_metrics,
            **drawdown_metrics,
            "options_metrics": options_metrics,
            "stock_metrics": stock_metrics,
            "transaction_costs": {
                "total_commission": total_commission,
                "total_slippage": total_slippage,
                "avg_commission_per_trade": (
                    total_commission / len(engine.portfolio.trades)
                    if engine.portfolio.trades
                    else 0
                ),
                "avg_slippage_per_trade": (
                    total_slippage / len(engine.portfolio.trades)
                    if engine.portfolio.trades
                    else 0
                ),
                "cost_as_pct_aum": (
                    (total_commission + total_slippage) / engine.initial_capital
                    if engine.initial_capital > 0
                    else 0
                ),
            },
            "execution_metrics": {
                "fill_ratio": np.mean(engine.fill_ratios) if engine.fill_ratios else 0,
                "daily_metrics": {
                    "avg_slippage": np.mean(engine.daily_metrics["slippage"]),
                    "avg_commission": np.mean(engine.daily_metrics["commissions"]),
                    "avg_fill_ratio": np.mean(engine.daily_metrics["fills"]),
                    "avg_volume_participation": np.mean(
                        engine.daily_metrics["volume_participation"]
                    ),
                },
            },
            "equity_curve": equity_df,
            "daily_returns": daily_returns,
            "trade_count": len(engine.portfolio.trades),
            "total_interest": total_interest_payouts,
            "total_dividends": total_dividend_payouts,
        }

        results.update(risk_metrics)

        return BacktestResults(
            strategy_name=engine.strategy.name,
            initial_capital=engine.initial_capital,
            final_capital=results["final_capital"],
            total_return=results["total_return"],
            annualized_return=results["annualized_return"],
            annualized_volatility=results["annualized_volatility"],
            sharpe_ratio=results["sharpe_ratio"],
            sortino_ratio=results["sortino_ratio"],
            max_drawdown=results["max_drawdown"],
            options_metrics=results["options_metrics"],
            stock_metrics=results["stock_metrics"],
            transaction_costs=results["transaction_costs"],
            execution_metrics=results["execution_metrics"],
            equity_curve=results["equity_curve"],
            drawdown_series=results["drawdown_series"],
            daily_returns=results["daily_returns"],
            net_pnl=results["net_pnl"],
            trades=engine.portfolio.trades,
            total_interest=results["total_interest"],
            total_dividends=results["total_dividends"],
        )
