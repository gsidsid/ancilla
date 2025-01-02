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
        name: str = "backtesting",
        market_data = {}
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

        # Initialize market data
        self.market_data = market_data
        self.last_timestamp = start_date

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
        self.total_transaction_costs = [] # Track total transaction costs for reconciliation

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
        market_data = self.market_data

        market_data_ticker = market_data[instrument.underlying_ticker]

        if instrument.is_option:
            # Get market data for the option ticker
            option_ticker = instrument.format_option_ticker()
            bars = self.data_provider.get_intraday_bars(
                ticker=option_ticker,
                start_date=timestamp - timedelta(hours=1),
                end_date=timestamp,
                interval='1hour'
            )
            data = bars.iloc[-1].to_dict() if bars is not None and not bars.empty else {}
            market_data[option_ticker] = data
            market_data_ticker = data
        else:
            # Get market data for the stock ticker
            bars = self.data_provider.get_intraday_bars(
                ticker=instrument.underlying_ticker,
                start_date=timestamp - timedelta(hours=1),
                end_date=timestamp,
                interval='1hour'
            )
            data = bars.iloc[-1].to_dict() if bars is not None and not bars.empty else {}
            market_data[instrument.underlying_ticker] = data
            market_data_ticker = data

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
        volume = market_data_ticker.get('volume', 0)
        participation_rate = abs(quantity) / volume if volume > 0 else 1
        if participation_rate > 0.1:  # Limit to 10% of daily volume
            adjusted_quantity = int(0.1 * volume) * (1 if quantity > 0 else -1)
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
            volume,
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
            int(volume),
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

        slippage = abs(price * price_impact) * abs(quantity)  # Calculate slippage based on percentage impact
        if instrument.is_option:
            slippage *= instrument.get_multiplier()
        total_transaction_costs = commission + slippage

        print()

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

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs,
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

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_price,
                        timestamp=timestamp,
                        transaction_costs=total_transaction_costs,
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
            self.total_transaction_costs.append(total_transaction_costs)

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
        _is_assignment: bool = False
    ) -> bool:
        """
        Buy stocks.

        Args:
            ticker: Stock ticker
            quantity: Number of shares
        """
        instrument = Stock(ticker)
        return self._execute_instrument_order(
            instrument=instrument,
            quantity=abs(quantity),  # Ensure positive
            timestamp=self.last_timestamp,
            is_assignment=_is_assignment
        )

    def sell_stock(
        self,
        ticker: str,
        quantity: int,
        _is_assignment: bool = False
    ) -> bool:
        """
        Sell stocks.

        Args:
            ticker: Stock ticker
            quantity: Number of shares
        """
        instrument = Stock(ticker)
        return self._execute_instrument_order(
            instrument=instrument,
            quantity=-abs(quantity),  # Ensure negative
            timestamp=self.last_timestamp,
            is_assignment=_is_assignment
        )

    def buy_option(
        self,
        option: Option,
        quantity: int
    ) -> bool:
        """
        Buy options.

        Args:
            option: Option instrument to trade
            quantity: Number of contracts
        """
        # Validate option exists and is tradeable
        if not self._validate_option_order(option, self.last_timestamp):
            return False

        # When we handle an option position, we should add the option ticker to the list of tickers
        if option.format_option_ticker() not in self.tickers:
            self.tickers.append(option.format_option_ticker())

        return self._execute_instrument_order(
            instrument=option,
            quantity=abs(quantity),
            timestamp=self.last_timestamp
        )

    def sell_option(
        self,
        option: Option,
        quantity: int
    ) -> bool:
        """
        Sell options.

        Args:
            option: Option instrument to trade
            quantity: Number of contracts
        """
        if not self._validate_option_order(option, self.last_timestamp):
            return False

        # When we handle an option position, we should add the option ticker to the list of tickers
        if option.format_option_ticker() not in self.tickers:
            self.tickers.append(option.format_option_ticker())

        return self._execute_instrument_order(
            instrument=option,
            quantity=-abs(quantity),
            timestamp=self.last_timestamp
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
                        # Instead of looking for exact hour match, find the closest bar
                        # that falls within this hour window
                        window_start = current_time
                        window_end = current_time + timedelta(hours=1)

                        # Filter bars that fall within our window
                        window_bars = bars[
                            (bars.index >= window_start) &
                            (bars.index < window_end)
                        ]

                        if not window_bars.empty:
                            # Take the first bar in the window
                            market_data[ticker] = window_bars.iloc[0].to_dict()
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
                    strategyCurrentTimeFormatter = logging.Formatter(
                        fmt=f"ancilla.{self.strategy.name} - [{
                            current_time.strftime('%Y-%m-%d %H:%M:%S')
                        }] - %(levelname)s - %(message)s"
                    )
                    self.strategy.logger.handlers[0].setFormatter(strategyCurrentTimeFormatter)
                    self.strategy.logger.handlers[1].setFormatter(strategyCurrentTimeFormatter)

                    current_line = "\r" + "ancilla." + self.strategy.name + " – [" + current_time.strftime('%Y-%m-%d %H:%M:%S') + "]" + "\033[K"
                    print(current_line, end='\r')
                    self.market_data = market_data
                    self.last_timestamp = current_time
                    self.strategy.on_data(current_time, market_data)

                    # Update portfolio equity curve
                    current_prices = {
                        ticker: data['close'] for ticker, data in market_data.items()
                    }

                    self.portfolio.update_equity(current_time, current_prices)

                current_time += timedelta(hours=1)

            self._process_option_expiration(current_date, market_close)
            current_date += timedelta(days=1)

        # Close remaining positions
        self._close_all_positions(current_date)

        # Interpret engine data
        results = BacktestResults.calculate(self)

        # Log summary
        self.logger.info("\n" + results.summarize() + "\n")
        self.logger.info("\n=====================TRADES========================\n")
        trade_summary = ""
        # sort trades by entry time
        self.portfolio.trades.sort(key=lambda x: x.entry_time)
        for trade in self.portfolio.trades:
            # Create a human-readable summary of all trades
            trade_metrics = trade.get_metrics()

            # Determine if it was a buy or sell
            action = "Bought" if trade_metrics['quantity'] > 0 else "Sold"
            abs_quantity = abs(trade_metrics['quantity'])

            # Format the instrument type
            if trade_metrics['type'] == 'option':
                instrument = f"{trade_metrics['ticker']} {trade_metrics['type']}"
            else:
                instrument = f"shares of {trade_metrics['ticker']}"

            # Format prices and P&L
            entry_price = "${:,.2f}".format(trade_metrics['entry_price'])
            exit_price = "${:,.2f}".format(trade_metrics['exit_price'])
            pnl = trade_metrics['pnl']
            pnl_str = "${:,.2f}".format(abs(pnl))

            # Create the trade description
            trade_desc = f"{action} {abs_quantity} {instrument} at {entry_price}, "
            if trade_metrics['exit_time']:
                trade_desc += f"closed at {exit_price} for a "
                trade_desc += f"{'profit' if pnl > 0 else 'loss'} of {pnl_str}"
            else:
                trade_desc += "position still open"

            # Add duration if closed
            if trade_metrics['exit_time']:
                duration_days = trade_metrics['duration_hours'] / 24
                trade_desc += f" (held for {duration_days:.1f} days)"

            trade_summary += trade_desc + "\n"

        self.logger.info("\n" + trade_summary)

        return results

    def _process_option_expiration(self, current_date: datetime, expiration_time: datetime):
        """Handle option expiration and assignments."""
        # Check if we're at market close (after 4 PM ET)
        is_market_close = (expiration_time.hour >= 16)
        if not is_market_close:
            return

        # Get expiring positions
        option_positions = [
            pos for pos in self.portfolio.positions.values()
            if pos.instrument.is_option and pos.instrument.expiration.date() == current_date.date()
        ]

        for position in option_positions:
            option = position.instrument
            ticker = option.format_option_ticker()

            self.logger.info(
                f"Processing expiration for {ticker}:\n"
                f"  Position direction: {'Long' if position.quantity > 0 else 'Short'}\n"
                f"  Option type: {option.instrument_type}\n"
                f"  Strike: ${option.strike:.2f}\n"
                f"  Underlying price: ${underlying_close:.2f}\n"
                f"  Intrinsic value: ${intrinsic_value:.2f}"
            )

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

            # Handle assignment
            self.logger.info(
                f"Attempting assignment for {ticker}:\n"
                f"  Assignment quantity: {assignment_quantity}\n"
                f"  Direction: {'Buy' if is_buying else 'Sell'} stock\n"
                f"  Strike price: ${option.strike:.2f}"
            )
            if intrinsic_value > 0:
                assignment_quantity = abs(position.quantity) * 100

                # For calls: the holder buys stock at strike, writer sells stock at strike
                if option.instrument_type == InstrumentType.CALL_OPTION:
                    self.sell_stock(
                        ticker=underlying_ticker,
                        quantity=assignment_quantity,
                        _is_assignment=True
                    )

                # For puts: the holder sells stock at strike, writer buys stock at strike
                else:  # PUT_OPTION
                    self.buy_stock(
                        ticker=underlying_ticker,
                        quantity=assignment_quantity,
                        _is_assignment=True
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


    def _close_all_positions(self, current_date: datetime):
        self.logger.info(f"Starting to close all positions at {current_date}")
        positions_to_close = list(self.portfolio.positions.items())
        self.logger.info(f"Found {len(positions_to_close)} positions to close:")
        for t, p in positions_to_close:
            self.logger.info(f"  {t}: {p.quantity} units @ {p.entry_price}")

        for ticker, position in positions_to_close:
            # Use the actual position ticker for lookup, not the underlying
            lookup_ticker = ticker  # This is the key change

            self.logger.info(f"Attempting to close {ticker} using lookup ticker {lookup_ticker}")
            self.logger.info(f"Fetching market data from {current_date - timedelta(hours=1)} to {current_date}")

            bars = self.data_provider.get_intraday_bars(
                ticker=lookup_ticker,
                start_date=current_date - timedelta(days=1),
                end_date=current_date,
                interval='1hour'
            )

            if bars is None or bars.empty:
                self.logger.warning(f"No market data found for {lookup_ticker} - searching for recent data")
                # Try to get data from the last week
                historical_bars = self.data_provider.get_intraday_bars(
                    ticker=lookup_ticker,
                    start_date=current_date - timedelta(days=7),
                    end_date=current_date,
                    interval='1hour'
                )

                if historical_bars is not None and not historical_bars.empty:
                    closing_price = historical_bars.iloc[-1]['close']
                    self.logger.info(f"Found recent price for {lookup_ticker}: {closing_price}")
                else:
                    self.logger.warning(f"No recent data found - falling back to entry price")
                    closing_price = position.entry_price
            else:
                self.logger.info(f"Found market data for {lookup_ticker}: {bars.tail(1)}")
                market_data = bars.iloc[-1].to_dict()
                closing_price = market_data['close']

            # Calculate value (this part remains the same)
            if position.instrument.is_option:
                multiplier = position.instrument.get_multiplier()
                value = closing_price * multiplier
            else:
                value = closing_price

            self.logger.info(f"Attempting to close {ticker} @ {closing_price:.2f}")

            success = self.portfolio.close_position(
                instrument=position.instrument,
                price=closing_price,
                timestamp=current_date,
                transaction_costs=self.market_simulator.calculate_commission(
                    closing_price,
                    position.quantity,
                    'option' if position.instrument.is_option else 'stock'
                )
            )

            if success:
                self.logger.info(f"Successfully closed position in {ticker} @ {closing_price:.2f}")
            else:
                self.logger.error(f"Failed to close position in {ticker}")

        remaining = list(self.portfolio.positions.items())
        if remaining:
            self.logger.error("Failed to close all positions. Remaining positions:")
            for t, p in remaining:
                self.logger.error(f"  {t}: {p.quantity} units @ {p.entry_price}")
        else:
            self.logger.info("Successfully closed all positions")
