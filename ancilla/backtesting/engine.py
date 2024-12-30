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
        self.initial_capital = initial_capital
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
        self.strategy.initialize(self.portfolio, self)
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
                atr = pd.Series(tr).rolling(window=window).mean()
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
        market_data = market_data[ticker]

        if not market_data:
            self.logger.warning(f"Insufficient market data for {ticker}")
            return False

        liquidity_score = self._calculate_liquidity_score(market_data)

        if liquidity_score < 0.1:
            self.logger.warning(f"Insufficient liquidity for {ticker}")
            return False

        daily_volume = market_data.get('volume', 0)
        participation_rate = abs(quantity) / daily_volume if daily_volume > 0 else 1
        if participation_rate > 0.1:
            adjusted_quantity = int(0.1 * daily_volume) * (1 if quantity > 0 else -1)
            self.logger.warning(
                f"Order size adjusted for liquidity: {quantity} -> {adjusted_quantity}"
            )
            quantity = adjusted_quantity

        if quantity == 0:
            return False

        # spread = self._estimate_spread(market_data)
        base_price = market_data['close']
        price_impact = self.market_simulator.calculate_price_impact(
            base_price,
            quantity,
            daily_volume,
            liquidity_score
        )
        execution_price = round(base_price * (1 + price_impact), 2)
        commission = self.market_simulator.calculate_commission(
            execution_price,
            quantity,
            position_type
        )
        fill_probability = self.market_simulator.estimate_market_hours_fill_probability(
            execution_price,
            quantity,
            market_data,
            position_type
        )

        if np.random.random() > fill_probability:
            self.logger.warning(
                f"Order failed to fill: {ticker} {quantity} @ {execution_price:.2f} "
                f"(fill probability: {fill_probability:.2%})"
            )
            return False

        success = self.portfolio.open_position(
            ticker=ticker,
            quantity=quantity,
            price=execution_price,
            timestamp=timestamp,
            position_type=position_type,
            commission=commission,
            slippage=abs(execution_price - base_price) * abs(quantity)
        )

        if success:
            # Track daily metrics (existing logic)
            self.daily_metrics['volume_participation'].append(participation_rate)
            self.daily_metrics['fills'].append(1.0)
            self.daily_metrics['slippage'].append(price_impact)
            self.daily_metrics['commissions'].append(commission)

            # Track total metrics for final reporting
            self.fill_ratios.append(fill_probability)
            self.slippage_costs.append(abs(execution_price - base_price) * abs(quantity))
            self.commission_costs.append(commission)

            print(
                f"Order executed: {ticker} {quantity} @ {execution_price:.2f}\n"
                f"  Base price: ${base_price:.2f}\n"
                f"  Impact: {price_impact:.2%}\n"
                f"  Commission: ${commission:.2f}\n"
                f"  Volume participation: {participation_rate:.2%}"
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


        # Close any remaining positions at the end of the backtest
        positions_to_close = list(self.portfolio.positions.items())
        for ticker, position in positions_to_close:
            market_data = self._get_market_data(ticker, current_date)
            if market_data:
                success = self.portfolio.close_position(
                    ticker=ticker,
                    price=market_data['close'],
                    timestamp=current_date
                )
                if success:
                    self.logger.info(
                        f"Closed remaining position in {ticker} @ {market_data['close']:.2f}"
                    )

        # Calculate and return performance metrics
        metrics = self._calculate_results()
        self._log_summary(metrics)
        return metrics

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        # Convert equity curve to dataframe with proper column names
        equity_curve_dict = {'timestamp': [], 'equity': []}
        for timestamp, equity in self.portfolio.equity_curve:
            equity_curve_dict['timestamp'].append(timestamp)
            equity_curve_dict['equity'].append(equity)

        equity_df = pd.DataFrame.from_dict(equity_curve_dict).set_index('timestamp')

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
        options_trades = [t for t in self.portfolio.trades if t.position_type == 'option']
        stock_trades = [t for t in self.portfolio.trades if not t.position_type == 'option']

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
                'total_commission': total_commission,
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
            'initial_capital': self.initial_capital,
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
                    (total_commission + total_slippage) / self.initial_capital
                    if self.initial_capital > 0 else 0
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
        # ensure daily_returns is not empty
        results.update({k: v for k, v in self._calculate_risk_metrics(daily_returns).items()})

        return results

    def _calculate_risk_metrics(self, daily_returns) -> Dict[str, Any]:
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
