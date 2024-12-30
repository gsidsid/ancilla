# ancilla/backtesting/engine.py
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

from ancilla.providers.polygon_data_provider import PolygonDataProvider
from ancilla.utils.logging import BacktesterLogger
from ancilla.backtesting.strategy import Strategy
from ancilla.backtesting.portfolio import Portfolio
from ancilla.models import OptionData
from ancilla.backtesting.simulation import (
    MarketSimulator, CommissionConfig, SlippageConfig
)

def format_option_ticker(ticker: str, expiry: datetime, strike: float, opt_type: str) -> str:
    """Format option ticker for Polygon API."""
    exp_str = expiry.strftime('%y%m%d')
    strike_str = f"{strike:08.0f}"
    return f"O:{ticker}{exp_str}{opt_type}{strike_str}"

class BacktestEngine:
    """Main backtesting engine with realistic market simulation."""

    def __init__(
        self,
        data_provider: PolygonDataProvider,
        strategy: Strategy,
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        tickers: List[str],
        commission_config: Optional[CommissionConfig] = None,
        slippage_config: Optional[SlippageConfig] = None,
        name: str = "backtesting"
    ):
        self.data_provider = data_provider
        self.strategy = strategy
        # Generate portfolio name from strategy name and timestamp
        name = f"{strategy.name}_orders"
        self.portfolio = Portfolio(name, initial_capital)
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers

        # Initialize market simulator
        self.market_simulator = MarketSimulator(commission_config, slippage_config)

        # Cache for market data and analytics
        self._market_data_cache = {}
        self._volume_profile_cache = {}  # For intraday volume patterns
        self._atr_cache = {}  # For volatility-based adjustments

        # Trade analytics
        self.fill_ratios = []  # Track fill rates
        self.slippage_costs = []  # Track slippage
        self.commission_costs = []  # Track commissions

        # Initialize strategy
        self.strategy.initialize(self.portfolio)
        self.logger = BacktesterLogger().get_logger()

        # Cache for daily metrics
        self.daily_metrics = {
            'slippage': [],
            'commissions': [],
            'fills': [],
            'volume_participation': []
        }

    def _get_market_data(self, ticker: str, date: datetime) -> Optional[Dict[str, Any]]:
        """Get market data with caching and enhanced analytics."""
        cache_key = (ticker, date.date())
        if cache_key not in self._market_data_cache:
            bars = self.data_provider.get_daily_bars(ticker, date, date)
            if bars is not None and not bars.empty:
                data = bars.iloc[0].to_dict()

                # Add enhanced analytics
                data['atr'] = self._calculate_atr(ticker, date)
                data['avg_spread'] = self._estimate_spread(data)
                data['liquidity_score'] = self._calculate_liquidity_score(data)

                self._market_data_cache[cache_key] = data
            else:
                self._market_data_cache[cache_key] = None
        return self._market_data_cache[cache_key]

    def _calculate_atr(self, ticker: str, date: datetime, window: int = 14) -> float:
        """Calculate Average True Range for volatility estimation."""
        cache_key = (ticker, date.date())
        if cache_key not in self._atr_cache:
            end_date = date
            start_date = end_date - timedelta(days=window * 2)  # Extra days for calculation

            bars = self.data_provider.get_daily_bars(ticker, start_date, end_date)
            if bars is not None and not bars.empty:
                high = bars['high']
                low = bars['low']
                close = bars['close']

                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))

                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=window).mean().iloc[-1]
                self._atr_cache[cache_key] = atr
            else:
                self._atr_cache[cache_key] = None

        return self._atr_cache[cache_key]

    def _estimate_spread(self, market_data: Dict[str, Any]) -> float:
        """Estimate average spread from OHLC data."""
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        volume = market_data.get('volume', 0)
        price = market_data.get('close', 0)

        if price == 0 or volume == 0:
            return 0.0

        # Base spread on price level and volume
        base_spread = (high - low) / (2 * price)  # Half the day's range
        volume_factor = np.log10(max(volume, 1))
        return base_spread / volume_factor

    def _calculate_liquidity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate a liquidity score (0-1) based on volume and price."""
        volume = market_data.get('volume', 0)
        price = market_data.get('close', 0)
        dollar_volume = volume * price

        # Score based on dollar volume (adjust thresholds as needed)
        if dollar_volume == 0:
            return 0.0
        return min(1.0, np.log10(dollar_volume) / 7.0)  # 7.0 ~ $10M daily volume

    def execute_order(
        self,
        ticker: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        market_data: Dict[str, Any],
        position_type: str = 'stock',
        option_data: Optional[OptionData] = None
    ) -> bool:
        """Execute order with full market simulation."""
        # Check if we have sufficient market data
        if not market_data:
            self.logger.warning(f"Insufficient market data for {ticker}")
            return False

        # Adjust quantity for liquidity
        adjusted_quantity = self.market_simulator.adjust_for_liquidity(
            quantity, market_data, position_type
        )

        if adjusted_quantity != quantity:
            self.logger.warning(
                f"Order size adjusted for liquidity: {quantity} -> {adjusted_quantity} "
                f"({ticker})"
            )
            if adjusted_quantity == 0:
                return False

        # Calculate execution price with slippage
        direction = 1 if quantity > 0 else -1
        execution_price = self.market_simulator.calculate_execution_price(
            ticker, price, adjusted_quantity, market_data, direction, position_type
        )

        # Calculate commission
        commission = self.market_simulator.calculate_commission(
            execution_price, adjusted_quantity, position_type
        )

        # Check fill probability
        fill_probability = self.market_simulator.estimate_market_hours_fill_probability(
            execution_price, market_data, position_type
        )

        # Adjust fill probability based on liquidity score
        liquidity_score = market_data.get('liquidity_score', 1.0)
        fill_probability *= liquidity_score

        # Simulate fill
        if np.random.random() > fill_probability:
            self.logger.warning(
                f"Order failed to fill: {ticker} {adjusted_quantity} @ {price:.2f} "
                f"(fill probability: {fill_probability:.2%})"
            )
            self.fill_ratios.append(0.0)
            return False

        # Track costs
        slippage = abs(execution_price - price) / price
        self.slippage_costs.append(slippage * abs(adjusted_quantity * execution_price))
        self.commission_costs.append(commission)
        self.fill_ratios.append(1.0)

        # Execute the trade
        success = self.portfolio.open_position(
            ticker=ticker,
            quantity=adjusted_quantity,
            price=execution_price,
            timestamp=timestamp,
            position_type=position_type,
            option_data=option_data,
            commission=commission
        )

        if success:
            price_impact = (execution_price - price) / price
            self.logger.info(
                f"Order executed: {ticker} {adjusted_quantity} @ {execution_price:.2f} "
                f"(slippage: {price_impact:.2%}, commission: ${commission:.2f})"
            )

            # Update daily metrics
            self.daily_metrics['slippage'].append(slippage)
            self.daily_metrics['commissions'].append(commission)
            self.daily_metrics['fills'].append(1.0)
            self.daily_metrics['volume_participation'].append(
                abs(adjusted_quantity) / market_data.get('volume', 1)
            )

        return success

    def run(self) -> Dict[str, Any]:
        """Run the backtest with market simulation."""
        current_date = self.start_date
        consecutive_no_data = 0
        max_no_data_days = 5

        while current_date <= self.end_date:
            # Skip non-trading days
            if not self.data_provider.get_market_hours(current_date):
                current_date += timedelta(days=1)
                continue

            # Get market data
            market_data = {}
            has_data = False

            for ticker in self.tickers:
                data = self._get_market_data(ticker, current_date)
                if data:
                    market_data[ticker] = data
                    has_data = True

            if not has_data:
                consecutive_no_data += 1
                if consecutive_no_data >= max_no_data_days:
                    self.logger.warning(
                        f"No market data for {max_no_data_days} consecutive days "
                        f"as of {current_date.date()}"
                    )
                    consecutive_no_data = 0
            else:
                consecutive_no_data = 0

            # Process market data in strategy
            if market_data:
                self.strategy.on_data(current_date, market_data)

                # Update portfolio equity curve with current market prices
                current_prices = {ticker: data['close']
                                for ticker, data in market_data.items()}
                self.portfolio.update_equity(current_date, current_prices)

            current_date += timedelta(days=1)

        # Calculate and return performance metrics
        metrics = self._calculate_results()
        self._log_summary(metrics)
        return metrics

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        equity_df = pd.DataFrame(
            self.portfolio.equity_curve,
            columns=['timestamp', 'equity']
        ).set_index('timestamp')

        equity_df['returns'] = equity_df['equity'].pct_change()

        # Basic metrics
        total_return = (equity_df['equity'].iloc[-1] / equity_df['equity'].iloc[0]) - 1
        trading_days = len(equity_df)

        # Risk metrics
        daily_returns = equity_df['returns'].dropna()
        daily_std = daily_returns.std()
        annualized_vol = daily_std * np.sqrt(252) if daily_std != 0 else 0
        annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1

        # Drawdown analysis
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        # Transaction cost analysis
        total_commission = sum(self.commission_costs)
        total_slippage = sum(self.slippage_costs)
        avg_fill_ratio = np.mean(self.fill_ratios) if self.fill_ratios else 0

        # Separate options and stock trades
        options_trades = [t for t in self.portfolio.trades if t.option_data]
        stock_trades = [t for t in self.portfolio.trades if not t.option_data]

        # Calculate trade metrics
        def calculate_trade_metrics(trades):
            if not trades:
                return {
                    'count': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'total_commission': 0,
                    'avg_holding_period': 0,
                    'profit_factor': 0
                }

            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]

            holding_periods = [
                (t.exit_date - t.entry_date).days
                for t in trades
            ]

            return {
                'count': len(trades),
                'win_rate': len(wins) / len(trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
                'total_pnl': sum(t.pnl for t in trades),
                'total_commission': sum(t.commission for t in trades),
                'avg_holding_period': np.mean(holding_periods),
                'profit_factor': (
                    abs(sum(t.pnl for t in wins)) /
                    abs(sum(t.pnl for t in losses))
                    if losses else float('inf')
                )
            }

        # Daily metrics averages
        avg_daily_metrics = {
            'avg_slippage': np.mean(self.daily_metrics['slippage']),
            'avg_commission': np.mean(self.daily_metrics['commissions']),
            'avg_fill_ratio': np.mean(self.daily_metrics['fills']),
            'avg_volume_participation': np.mean(self.daily_metrics['volume_participation'])
        }

        # Compile results
        results = {
            'initial_capital': equity_df['equity'].iloc[0],
            'final_capital': equity_df['equity'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': (annualized_return / annualized_vol) if annualized_vol != 0 else 0,
            'max_drawdown': max_drawdown,
            'options_metrics': calculate_trade_metrics(options_trades),
            'stock_metrics': calculate_trade_metrics(stock_trades),
            'transaction_costs': {
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'avg_commission_per_trade': (
                    total_commission / len(self.portfolio.trades)
                    if self.portfolio.trades else 0
                ),
                'avg_slippage_per_trade': (
                    total_slippage / len(self.portfolio.trades)
                    if self.portfolio.trades else 0
                ),
                'cost_as_pct_aum': (
                    (total_commission + total_slippage) / equity_df['equity'].iloc[0]
                    if equity_df['equity'].iloc[0] > 0 else 0
                )
            },
            'execution_metrics': {
                'fill_ratio': avg_fill_ratio,
                'daily_metrics': avg_daily_metrics
            },
            'equity_curve': equity_df,
            'drawdown_series': equity_df['drawdown'],
            'daily_returns': daily_returns,
            'trade_count': len(self.portfolio.trades),
            'win_rate': (
                sum(1 for t in self.portfolio.trades if t.pnl > 0) /
                len(self.portfolio.trades) if self.portfolio.trades else 0
            )
        }

        # Add risk analysis metrics
        results.update(self._calculate_risk_metrics(daily_returns))

        return results

    def _calculate_risk_metrics(self, daily_returns: pd.Series) -> Dict[str, Any]:
        """Calculate additional risk metrics."""
        if daily_returns.empty:
            return {
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'var_95': 0,
                'var_99': 0,
                'max_consecutive_losses': 0
            }

        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        avg_return = daily_returns.mean() * 252
        sortino_ratio = avg_return / downside_std if downside_std != 0 else 0

        # Value at Risk
        var_95 = daily_returns.quantile(0.05)
        var_99 = daily_returns.quantile(0.01)

        # Maximum consecutive losses
        losses = (daily_returns < 0).astype(int)
        max_consecutive_losses = (
            losses.groupby((losses != losses.shift()).cumsum())
            .count().max()
        )

        return {
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'var_99': var_99,
            'max_consecutive_losses': max_consecutive_losses
        }

    def _log_summary(self, metrics: Dict[str, Any]) -> None:
        """Log detailed backtest summary."""
        summary_lines = [
            f"\n{self.strategy.name} Backtest Results",
            "=" * 50,
            "\nPerformance Metrics:",
            f"  Total Return: {metrics['total_return']:.2%}",
            f"  Annualized Return: {metrics['annualized_return']:.2%}",
            f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}",

            "\nRisk Metrics:",
            f"  Max Drawdown: {metrics['max_drawdown']:.2%}",
            f"  Annualized Volatility: {metrics['annualized_volatility']:.2%}",
            f"  95% VaR: {metrics['var_95']:.2%}",
            f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}",

            "\nTransaction Costs:",
            f"  Total Commission: ${metrics['transaction_costs']['total_commission']:,.2f}",
            f"  Total Slippage: ${metrics['transaction_costs']['total_slippage']:,.2f}",
            f"  Cost % of AUM: {metrics['transaction_costs']['cost_as_pct_aum']:.2%}",

            "\nExecution Quality:",
            f"  Fill Ratio: {metrics['execution_metrics']['fill_ratio']:.2%}",
            f"  Avg Daily Volume Participation: "
            f"{metrics['execution_metrics']['daily_metrics']['avg_volume_participation']:.2%}",

            "\nTrade Analysis:",
            "  Options Trades:",
            f"    Count: {metrics['options_metrics']['count']}",
            f"    Win Rate: {metrics['options_metrics']['win_rate']:.2%}",
            f"    Avg P&L: ${metrics['options_metrics']['avg_pnl']:,.2f}",
            f"    Avg Holding Period: {metrics['options_metrics']['avg_holding_period']:.1f} days",
            "  Stock Trades:",
            f"    Count: {metrics['stock_metrics']['count']}",
            f"    Win Rate: {metrics['stock_metrics']['win_rate']:.2%}",
            f"    Avg P&L: ${metrics['stock_metrics']['avg_pnl']:,.2f}",
            f"    Avg Holding Period: {metrics['stock_metrics']['avg_holding_period']:.1f} days",

            "\nCapital Summary:",
            f"  Initial: ${metrics['initial_capital']:,.2f}",
            f"  Final: ${metrics['final_capital']:,.2f}",
            "=" * 50,
        ]

        self.logger.info("\n".join(summary_lines))
