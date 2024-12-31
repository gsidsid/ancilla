from audioop import add
# ancilla/backtesting/engine.py
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz
import logging

from ancilla.utils.logging import BacktesterLogger
from ancilla.models import Trade
from ancilla.providers.polygon_data_provider import PolygonDataProvider
from ancilla.backtesting.instruments import Instrument, Option, Stock, InstrumentType
from ancilla.backtesting.results import BacktestResults
from ancilla.backtesting.strategy import Strategy
from ancilla.backtesting.portfolio import Portfolio
from ancilla.backtesting.simulation import (
    MarketSimulator, CommissionConfig, SlippageConfig
)
from ancilla.formulae.metrics import (
    calculate_return_metrics, calculate_drawdown_metrics, calculate_risk_metrics,
    calculate_trade_metrics
)

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

    def _execute_instrument_order(
        self,
        instrument: Instrument,
        quantity: int,
        timestamp: datetime,
        market_data: Dict[str, Any],
        is_assignment: bool = False
    ) -> bool:
        """
        Internal method to execute orders for any instrument type.

        Args:
            instrument: Instrument to trade
            quantity: Order quantity (positive for buy, negative for sell)
            timestamp: Current timestamp
            market_data: Market data dictionary
        """
        if not market_data or instrument.underlying_ticker not in market_data:
            self.logger.warning(f"Insufficient market data for underlying {instrument.underlying_ticker}")
            return False

        market_data_ticker = market_data[instrument.underlying_ticker]

        # Get current price/target price to execute with
        price = market_data_ticker.get('close', 0)
        if price is None:
            self.logger.warning(f"No price data for {instrument.ticker}")
            return False

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(market_data_ticker)
        if liquidity_score < 0.1:
            self.logger.warning(f"Insufficient liquidity for {instrument.ticker}")
            return False

        # Adjust quantity for volume
        daily_volume = market_data_ticker.get('volume', 0)
        participation_rate = abs(quantity) / daily_volume if daily_volume > 0 else 1
        if participation_rate > 0.1:  # Limit to 10% of daily volume
            adjusted_quantity = int(0.1 * daily_volume) * (1 if quantity > 0 else -1)
            self.logger.warning(
                f"Order size adjusted for liquidity: {quantity} -> {adjusted_quantity}"
            )
            quantity = adjusted_quantity

        if quantity == 0:
            return False

        # Calculate execution details
        price_impact = self.market_simulator.calculate_price_impact(
            price,
            quantity,
            daily_volume,
            liquidity_score
        )

        execution_price = round(price * (1 + price_impact), 2)
        commission = self.market_simulator.calculate_commission(
            execution_price,
            quantity,
            'option' if instrument.is_option else 'stock'
        )

        fill_probability = self.market_simulator.estimate_market_hours_fill_probability(
            execution_price,
            quantity,
            market_data_ticker,
            'option' if instrument.is_option else 'stock'
        )
        if is_assignment:
            fill_probability = 1.0 # Assume assignment always fills

        # Check if order fills
        if np.random.random() > fill_probability:
            self.logger.warning(
                f"Order failed to fill: {instrument.ticker} {quantity} @ {execution_price:.2f} "
                f"(fill probability: {fill_probability:.2%})"
            )
            return False

        # Calculate total transaction costs
        slippage = abs(execution_price - price) * abs(quantity)
        total_transaction_costs = commission + slippage

        # Execute the order based on whether it's opening or closing a position
        success = False
        if quantity > 0:  # Buying
            if instrument.is_option:
                option_ticker = instrument.format_option_ticker()
                if option_ticker in self.portfolio.positions:
                    # Buying to close a short option position
                    position = self.portfolio.positions[option_ticker]
                    initial_premium = abs(position.quantity) * position.entry_price * instrument.get_multiplier()
                    buyback_cost = abs(quantity) * execution_price * instrument.get_multiplier()
                    realized_pnl = initial_premium - buyback_cost - total_transaction_costs

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs,
                        realized_pnl=realized_pnl
                    )
                else:
                    # Opening a new long option position
                    success = self.portfolio.open_position(
                        instrument=instrument,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs
                    )
            else:
                # Opening a new stock position
                success = self.portfolio.open_position(
                    instrument=instrument,
                    quantity=quantity,
                    price=execution_price,
                    timestamp=timestamp,
                    transaction_costs=total_transaction_costs
                )
        else:  # Selling
            if instrument.is_option:
                option_ticker = instrument.format_option_ticker()
                if option_ticker in self.portfolio.positions:
                    # Selling to close a long option position
                    position = self.portfolio.positions[option_ticker]
                    realized_pnl = (execution_price - position.entry_price) * position.quantity * instrument.get_multiplier() - total_transaction_costs

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs,
                        realized_pnl=realized_pnl
                    )
                else:
                    # Opening a new short option position
                    success = self.portfolio.open_position(
                        instrument=instrument,
                        quantity=quantity,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs
                    )
            else:
                # Closing a stock position
                if instrument.ticker in self.portfolio.positions:
                    success = self.portfolio.close_position(
                        instrument=self.portfolio.positions[instrument.ticker].instrument,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs
                    )

        if success:
            # Track metrics
            self.daily_metrics['volume_participation'].append(participation_rate)
            self.daily_metrics['fills'].append(1.0)
            self.daily_metrics['slippage'].append(price_impact)
            self.daily_metrics['commissions'].append(commission)

            self.fill_ratios.append(fill_probability)
            self.slippage_costs.append(slippage)
            self.commission_costs.append(commission)

            # Log execution
            self.logger.info(
                f"Order executed: {instrument.ticker} {quantity} @ {execution_price:.2f}\n"
                f"  Base price: ${price:.2f}\n"
                f"  Impact: {price_impact:.2%}\n"
                f"  Commission: ${commission:.2f}\n"
                f"  Volume participation: {participation_rate:.2%}"
            )

        return success

    def buy_stock(
        self,
        ticker: str,
        quantity: int,
        timestamp: datetime,
        market_data: Dict[str, Any],
        is_assignment: bool = False
    ) -> bool:
        """Wrapper for buying stocks."""
        instrument = Stock(ticker)
        return self._execute_instrument_order(
            instrument=instrument,
            quantity=abs(quantity),  # Ensure positive
            timestamp=timestamp,
            market_data=market_data,
            is_assignment=is_assignment
        )

    def sell_stock(
        self,
        ticker: str,
        quantity: int,
        timestamp: datetime,
        market_data: Dict[str, Any],
        is_assignment: bool = False
    ) -> bool:
        """Wrapper for selling stocks."""
        instrument = Stock(ticker)
        return self._execute_instrument_order(
            instrument=instrument,
            quantity=-abs(quantity),  # Ensure negative
            timestamp=timestamp,
            market_data=market_data,
            is_assignment=is_assignment
        )

    def buy_option(
        self,
        option: Option,
        quantity: int,
        timestamp: datetime,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Wrapper for buying options with validation.

        Args:
            option: Option instrument to trade
            quantity: Number of contracts
            timestamp: Current timestamp
            market_data: Market data dictionary
        """
        # Validate option exists and is tradeable
        if not self._validate_option_order(option, timestamp):
            return False

        return self._execute_instrument_order(
            instrument=option,
            quantity=abs(quantity),
            timestamp=timestamp,
            market_data=market_data,
        )

    def sell_option(
        self,
        option: Option,
        quantity: int,
        timestamp: datetime,
        market_data: Dict[str, Any]
    ) -> bool:
        """Wrapper for selling options with validation."""
        if not self._validate_option_order(option, timestamp):
            return False

        return self._execute_instrument_order(
            instrument=option,
            quantity=-abs(quantity),
            timestamp=timestamp,
            market_data=market_data
        )

    def _validate_option_order(self, option: Option, timestamp: datetime) -> bool:
        """Validate that an option exists and is tradeable."""
        try:
            # Get option data using the formatted option ticker
            option_ticker = option.format_option_ticker()

            # Ensure all datetimes are timezone aware
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            if option.expiration.tzinfo is None:
                option.expiration = pytz.UTC.localize(option.expiration)

            # Check expiration
            if option.expiration <= timestamp:
                self.logger.warning(f"Option {option_ticker} is expired")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating option order {option.ticker}: {str(e)}")
            return False

    def run(self) -> BacktestResults:
        """Run the backtest with hourly resolution and return structured results."""
        current_date = self.start_date
        consecutive_no_data = 0
        max_no_data_days = 5

        while current_date <= self.end_date:
            # Check if it's a trading day
            market_hours = self.data_provider.get_market_hours(current_date)
            if not market_hours:
                current_date += timedelta(days=1)
                continue

            # Get market open and close times for the day
            market_open = market_hours['market_open']
            market_close = market_hours['market_close']

            # Iterate through each hour of the trading day
            current_time = market_open
            while current_time <= market_close:
                # Get intraday market data for all tickers
                market_data = {}
                has_data = False

                # Expand self.tickers to include option tickers for any open positions
                open_positions = self.portfolio.positions.values()
                for position in open_positions:
                    if position.instrument.is_option and position.instrument.format_option_ticker() not in self.tickers:
                        self.tickers.append(position.instrument.format_option_ticker())

                for ticker in self.tickers:
                    # Get 1-hour bars for the current hour
                    bars = self.data_provider.get_intraday_bars(
                        ticker=ticker,
                        start_date=current_time,
                        end_date=current_time + timedelta(hours=1),
                        interval='1hour'
                    )

                    if bars is not None and not bars.empty:
                        # Use the last bar's data for the current hour
                        data = bars.iloc[-1].to_dict()
                        market_data[ticker] = data
                        has_data = True

                if not has_data:
                    consecutive_no_data += 1
                    if consecutive_no_data >= max_no_data_days * 6.5:  # Adjust for ~6.5 trading hours per day
                        self.logger.warning(
                            f"No market data for {max_no_data_days} consecutive days "
                            f"as of {current_time}"
                        )
                        consecutive_no_data = 0
                else:
                    consecutive_no_data = 0

                    # Process market data in strategy
                    # Override the strategy logger to include the timestamp
                    strategyCurrentTimeFormatter = logging.Formatter(
                        fmt=f"ancilla.{self.strategy.name} - [{
                            current_time.strftime('%Y-%m-%d %H:%M:%S')
                        }] - %(levelname)s - %(message)s"
                    )
                    self.strategy.logger.handlers[0].setFormatter(strategyCurrentTimeFormatter)
                    self.strategy.logger.handlers[1].setFormatter(strategyCurrentTimeFormatter)

                    # self.logger.info(f"{current_time}")
                    self.strategy.on_data(current_time, market_data)

                    # Update portfolio equity curve
                    current_prices = {
                        ticker: data['close'] for ticker, data in market_data.items()
                    }
                    self.portfolio.update_equity(current_time, current_prices)

                current_time += timedelta(hours=1)

            self._process_option_expiration(current_date, market_close)

            current_date += timedelta(days=1)

        # Close remaining positions (rest of the method remains the same)
        positions_to_close = list(self.portfolio.positions.items())
        for ticker, position in positions_to_close:
            # Use the final hour's data for closing positions
            bars = self.data_provider.get_intraday_bars(
                ticker=ticker,
                start_date=current_date - timedelta(hours=1),
                end_date=current_date,
                interval='1hour'
            )

            if bars is not None and not bars.empty:
                market_data = bars.iloc[-1].to_dict()
                success = self.portfolio.close_position(
                    instrument=position.instrument,
                    price=market_data['close'],
                    timestamp=current_date
                )
                if success:
                    self.logger.info(
                        f"Closed remaining position in {ticker} @ {market_data['close']:.2f}"
                    )

        # Calculate performance metrics
        metrics = self._calculate_results()

        # Create and return structured results
        results = BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=metrics['final_capital'],
            total_return=metrics['total_return'],
            annualized_return=metrics['annualized_return'],
            annualized_volatility=metrics['annualized_volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            options_metrics=metrics['options_metrics'],
            stock_metrics=metrics['stock_metrics'],
            transaction_costs=metrics['transaction_costs'],
            execution_metrics=metrics['execution_metrics'],
            equity_curve=metrics['equity_curve'],
            drawdown_series=metrics['drawdown_series'],
            daily_returns=metrics['daily_returns'],
            trades=self.portfolio.trades
        )

        print(self.portfolio.trades)
        print(self)

        # Log summary
        self.logger.info("\n" + results.summarize())

        return results

    def _process_option_expiration(self, current_date: datetime, expiration_time: datetime):
        """Handle option expiration and assignments."""
        option_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.instrument.is_option and pos.instrument.expiration.date() == current_date.date()
        ]

        for position in option_positions:
            option = position.instrument
            ticker = option.format_option_ticker()

            # Get underlying price
            underlying_ticker = option.underlying_ticker
            underlying_data = self.data_provider.get_intraday_bars(
                ticker=underlying_ticker,
                start_date=expiration_time - timedelta(hours=1),
                end_date=expiration_time,
                interval='1hour'
            )

            if underlying_data is None or underlying_data.empty:
                self.logger.warning(f"No underlying data found for {underlying_ticker}")
                continue

            underlying_close = underlying_data.iloc[-1]['close']

            # Calculate intrinsic value
            if option.instrument_type == InstrumentType.CALL_OPTION:
                intrinsic_value = max(underlying_close - option.strike, 0)
            else:  # PUT_OPTION
                intrinsic_value = max(option.strike - underlying_close, 0)

            # Log expiration status
            if intrinsic_value > 0:
                self.logger.info(
                    f"Option {ticker} is in-the-money at expiration with intrinsic value ${intrinsic_value:.2f}."
                )
            else:
                self.logger.info(f"Option {ticker} expires worthless (out-of-the-money).")

            # Calculate transaction costs for potential assignment
            transaction_costs = 0
            if intrinsic_value > 0:
                assignment_quantity = abs(position.quantity) * 100
                transaction_costs = self.market_simulator.calculate_commission(
                    option.strike * assignment_quantity,
                    assignment_quantity,
                    'stock'
                )

                # Handle assignment
                if option.instrument_type == InstrumentType.CALL_OPTION and position.quantity < 0:
                    # For covered calls, execute stock sale at strike
                    self.sell_stock(
                        ticker=underlying_ticker,
                        quantity=assignment_quantity,
                        timestamp=expiration_time,
                        market_data={underlying_ticker: {'close': underlying_close}},
                        is_assignment=True
                    )

            # Calculate P&L
            multiplier = option.get_multiplier()
            initial_premium = abs(position.quantity) * position.entry_price * multiplier
            assignment_cost = intrinsic_value * abs(position.quantity) * multiplier
            realized_pnl = initial_premium - assignment_cost - transaction_costs

            # Log P&L breakdown
            self.logger.info(
                f"Option {ticker} expiration P&L breakdown:"
                f"\n  Initial premium received: ${initial_premium:.2f}"
                f"\n  Assignment cost: ${assignment_cost:.2f}"
                f"\n  Transaction costs: ${transaction_costs:.2f}"
                f"\n  Total P&L: ${realized_pnl:.2f}"
            )

            # Close the position (this will handle cash updates)
            trade = Trade(
                instrument=option,
                entry_time=position.entry_date,
                exit_time=expiration_time,
                entry_price=position.entry_price,
                exit_price=intrinsic_value,
                quantity=position.quantity,
                transaction_costs=transaction_costs,
                assignment=(intrinsic_value > 0),
                realized_pnl=realized_pnl
            )

            # Add trade to portfolio and remove position
            self.portfolio.trades.append(trade)
            del self.portfolio.positions[ticker]

    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        # Convert equity curve to dataframe
        equity_curve_dict = {'timestamp': [], 'equity': []}
        for timestamp, equity in self.portfolio.equity_curve:
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
        options_trades = [t for t in self.portfolio.trades if t.instrument.is_option]
        stock_trades = [t for t in self.portfolio.trades if not t.instrument.is_option]

        # Calculate trade metrics
        options_metrics = calculate_trade_metrics(options_trades)
        stock_metrics = calculate_trade_metrics(stock_trades)

        # Calculate total transaction costs
        total_commission = sum(t.transaction_costs for t in self.portfolio.trades if t.instrument.is_option)
        total_slippage = sum(t.transaction_costs for t in self.portfolio.trades if t.instrument.is_option)
        # Alternatively, if slippage and commissions are tracked separately
        total_commission = sum(self.commission_costs)
        total_slippage = sum(self.slippage_costs)

        # Compile results
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': equity_df['equity'].iloc[-1],
            **return_metrics,
            **drawdown_metrics,
            'options_metrics': options_metrics,
            'stock_metrics': stock_metrics,
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
                'fill_ratio': np.mean(self.fill_ratios) if self.fill_ratios else 0,
                'daily_metrics': {
                    'avg_slippage': np.mean(self.daily_metrics['slippage']),
                    'avg_commission': np.mean(self.daily_metrics['commissions']),
                    'avg_fill_ratio': np.mean(self.daily_metrics['fills']),
                    'avg_volume_participation': np.mean(self.daily_metrics['volume_participation'])
                }
            },
            'equity_curve': equity_df,
            'daily_returns': daily_returns,
            'trade_count': len(self.portfolio.trades)
        }

        results.update(risk_metrics)
        return results
