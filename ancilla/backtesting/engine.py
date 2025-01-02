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
    Broker, CommissionConfig, SlippageConfig
)

class BacktestEngine:
    """Main backtesting engine with realistic broker simulation."""

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

        # Set timezone to Eastern Time
        start_date = start_date.astimezone(pytz.timezone("US/Eastern"))
        end_date = end_date.astimezone(pytz.timezone("US/Eastern"))
        start_date = start_date.replace(hour=9, minute=30, second=0, microsecond=0)
        end_date = end_date.replace(hour=16, minute=0, second=0, microsecond=0)

        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers

        # Initialize market data
        self.market_data = market_data
        self.last_timestamp = start_date

        # Initialize market simulator
        self.broker = Broker(commission_config, slippage_config)

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

    def _execute_instrument_order(
        self,
        instrument: Instrument,
        quantity: int,
        timestamp: datetime,
        is_assignment: bool = False
    ) -> bool:
        """
        Internal method to execute orders with broker simulation.
        """
        market_data = self.market_data

        # Get market data for either option or stock ticker
        ticker = instrument.format_option_ticker() if instrument.is_option else instrument.underlying_ticker

        bars = self.data_provider.get_intraday_bars(
            ticker=ticker,
            start_date=timestamp - timedelta(hours=1),
            end_date=timestamp,
            interval='1hour'
        )

        data = bars.iloc[-1].to_dict() if bars is not None and not bars.empty else {}
        market_data[ticker] = data
        market_data_ticker = data

        # Get target price
        price = market_data_ticker.get('close', 0)
        if price is None:
            self.logger.warning(f"No price data for {instrument.ticker}")
            return False

        # Calculate execution details using broker simulation
        execution_details = self.broker.calculate_execution_details(
            ticker=instrument.ticker if not instrument.is_option else instrument.format_option_ticker(),
            base_price=price,
            quantity=quantity,
            market_data=market_data_ticker,
            asset_type='option' if instrument.is_option else 'stock'
        )
        if not execution_details or execution_details.adjusted_quantity == 0:
            return False

        # Use fill probability from execution details
        fill_probability = 1.0 if is_assignment else execution_details.fill_probability
        if not is_assignment and np.random.random() > fill_probability:
            self.logger.warning(
                f"Order failed to fill: {instrument.ticker} {execution_details.adjusted_quantity} @ {execution_details.execution_price:.2f} "
                f"(fill probability: {fill_probability:.2%})"
            )
            return False

        # Execute the order based on whether it's opening or closing a position
        success = False
        quantity = execution_details.adjusted_quantity  # Use adjusted quantity for execution
        if quantity > 0:  # Buying
            if instrument.is_option:
                option_ticker = instrument.format_option_ticker()
                if option_ticker in self.portfolio.positions:
                    # Buying to close a short option position
                    position = self.portfolio.positions[option_ticker]
                    # initial_premium = abs(position.quantity) * position.entry_price * instrument.get_multiplier()
                    # buyback_cost = abs(quantity) * execution_details.execution_price * instrument.get_multiplier()

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs,
                    )
                else:
                    # Opening a new long option position
                    success = self.portfolio.open_position(
                        instrument=instrument,
                        quantity=quantity,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs
                    )
            else:
                # Opening a new stock position
                success = self.portfolio.open_position(
                    instrument=instrument,
                    quantity=quantity,
                    price=execution_details.execution_price,
                    timestamp=timestamp,
                    transaction_costs=execution_details.total_transaction_costs
                )
        else:  # Selling
            if instrument.is_option:
                option_ticker = instrument.format_option_ticker()
                if option_ticker in self.portfolio.positions:
                    # Selling to close a long option position
                    position = self.portfolio.positions[option_ticker]

                    success = self.portfolio.close_position(
                        instrument=position.instrument,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs,
                    )
                else:
                    # Opening a new short option position
                    success = self.portfolio.open_position(
                        instrument=instrument,
                        quantity=quantity,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs
                    )
            else:
                # Closing a long stock position
                if instrument.ticker in self.portfolio.positions:
                    success = self.portfolio.close_position(
                        instrument=self.portfolio.positions[instrument.ticker].instrument,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs
                    )
                # Going short on a stock
                else:
                    success = self.portfolio.open_position(
                        instrument=instrument,
                        quantity=quantity,
                        price=execution_details.execution_price,
                        timestamp=timestamp,
                        transaction_costs=execution_details.total_transaction_costs
                    )


        if success:
            # Track metrics using consolidated execution details
            self.daily_metrics['volume_participation'].append(execution_details.participation_rate)
            self.daily_metrics['fills'].append(1.0)
            self.daily_metrics['slippage'].append(execution_details.price_impact)
            self.daily_metrics['commissions'].append(execution_details.commission)

            self.fill_ratios.append(fill_probability)
            self.slippage_costs.append(execution_details.slippage)
            self.commission_costs.append(execution_details.commission)
            self.total_transaction_costs.append(execution_details.total_transaction_costs)

            # Log execution
            self.logger.info(
                f"Order executed: {instrument.ticker} {quantity} @ {execution_details.execution_price:.2f}\n"
                f"  Base price: ${price:.2f}\n"
                f"  Impact: {execution_details.price_impact:.2%}\n"
                f"  Commission: ${execution_details.commission:.2f}\n"
                f"  Volume participation: {execution_details.participation_rate:.2%}"
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
                market_data = self.market_data
                has_data = False

                # Expand self.tickers to include option tickers for any open positions
                open_positions = self.portfolio.positions.values()
                for position in open_positions:
                    if position.instrument.is_option and position.instrument.format_option_ticker() not in self.tickers:
                        self.tickers.append(position.instrument.format_option_ticker())

                # Remove any tickers for expired options
                for ticker in self.tickers:
                    if len(ticker) > 6:
                        option = Option.from_option_ticker(ticker)
                        if option.expiration.date() < current_date.date():
                            self.tickers.remove(ticker)
                            market_data = {k: v for k, v in market_data.items() if k != ticker}

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
                    # Log with eastern time
                    eastern_time_12_hour_format = current_time.astimezone(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %I:%M %p')
                    strategyCurrentTimeFormatter = logging.Formatter(
                        fmt=f"ancilla.{self.strategy.name} - [{eastern_time_12_hour_format}] - %(levelname)s - %(message)s"
                    )
                    self.strategy.logger.handlers[0].setFormatter(strategyCurrentTimeFormatter)
                    self.strategy.logger.handlers[1].setFormatter(strategyCurrentTimeFormatter)

                    current_line = "\r" + "ancilla." + self.strategy.name + " – [" + eastern_time_12_hour_format + "]" + "\033[K"
                    print(current_line, end='\r')
                    self.market_data = market_data
                    self.last_timestamp = current_time
                    self.strategy.on_data(current_time, market_data)

                    # Update portfolio equity curve
                    current_prices = {
                        ticker: data['close'] for ticker, data in self.market_data.items()
                    }
                    self.portfolio.update_equity(current_time, current_prices)

                current_time += timedelta(hours=1)

            self._process_option_expiration(current_date, market_close)
            current_date += timedelta(days=1)

        # Close remaining positions
        self._close_all_positions(current_date - timedelta(days=1))

        # Interpret engine data
        results = BacktestResults.calculate(self)

        # Log summary
        self.logger.info(results.summarize())

        return results

    def _process_option_expiration(self, current_date: datetime, expiration_time: datetime):
        """
        Handle option expiration and assignments with careful cash flow tracking.
        Process:
        1. Identify expiring positions
        2. Calculate all values first
        3. Process assignments and handle position changes
        4. Record trades by directly interacting with the portfolio's handle_assignment and handle_exercise methods
        """
        if expiration_time.hour < 16:  # Only process at market close
            return

        # Get expiring options
        option_positions = [
            pos for pos in self.portfolio.positions.values()
            if (pos.instrument.is_option and
                pos.instrument.expiration.date() == current_date.date())
        ]

        for position in option_positions:
            option = position.instrument
            ticker = option.format_option_ticker()
            underlying_ticker = option.underlying_ticker

            # Get final underlying price
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
            self.logger.info(f"Processing expiration for {ticker} with underlying at ${underlying_close:.2f}")

            # Calculate intrinsic value
            if option.instrument_type == InstrumentType.CALL_OPTION:
                intrinsic_value = max(underlying_close - option.strike, 0)
            else:
                intrinsic_value = max(option.strike - underlying_close, 0)

            # Determine if option is in-the-money (ITM)
            MIN_ITM_THRESHOLD = 0.01
            is_itm = intrinsic_value > MIN_ITM_THRESHOLD

            # Determine the action based on position type and option type
            if is_itm:
                if position.quantity < 0:
                    # Short Option Assignment
                    if option.instrument_type == InstrumentType.CALL_OPTION:
                        # Short Call Assignment: Sell underlying stock at strike price
                        self.portfolio.handle_assignment(
                            option=option,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=True
                        )
                    elif option.instrument_type == InstrumentType.PUT_OPTION:
                        # Short Put Assignment: Buy underlying stock at strike price
                        self.portfolio.handle_assignment(
                            option=option,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=False
                        )
                else:
                    # Long Option Exercise
                    if option.instrument_type == InstrumentType.CALL_OPTION:
                        # Long Call Exercise: Buy underlying stock at strike price
                        self.portfolio.handle_exercise(
                            option=option,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=True
                        )
                    elif option.instrument_type == InstrumentType.PUT_OPTION:
                        # Long Put Exercise: Sell underlying stock at strike price
                        self.portfolio.handle_exercise(
                            option=option,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=False
                        )
            else:
                # Option expires worthless
                if position.quantity < 0:
                    # Short option: Keep premium
                    self.logger.info(f"Option {ticker} expired worthless. Premium kept.")
                    # Close the option position without any cash impact
                    self.portfolio.close_position(
                        instrument=option,
                        price=0.0,  # Option expired worthless
                        timestamp=expiration_time,
                        quantity=abs(position.quantity),
                        transaction_costs=0.0  # Adjust as needed
                    )
                else:
                    # Long option: Lose premium
                    self.logger.info(f"Option {ticker} expired worthless. Premium lost.")
                    # Close the option position without any additional action
                    self.portfolio.close_position(
                        instrument=option,
                        price=0.0,  # Option expired worthless
                        timestamp=expiration_time,
                        quantity=position.quantity,
                        transaction_costs=0.0  # Adjust as needed
                    )

            # Remove the option ticker from the list of tickers if fully closed
            if ticker not in self.portfolio.positions:
                if ticker in self.tickers:
                    self.tickers.remove(ticker)

            # Log final state
            self.logger.info(
                f"Option Expiration Summary for {ticker}:\n"
                f"  ITM: {is_itm}\n"
                f"  Intrinsic Value: ${intrinsic_value:.2f}"
            )

    def _close_all_positions(self, current_date: datetime):
        self.logger.info("End of test– automatically closing all positions")
        positions_to_close = list(self.portfolio.positions.items())
        self.logger.info(f"Found {len(positions_to_close)} open positions")
        for t, p in positions_to_close:
            self.logger.info(f"  {t}: {p.quantity} units @ {p.entry_price}")

        for ticker, position in positions_to_close:
            # Use the actual position ticker for lookup, not the underlying
            lookup_ticker = ticker  # This is the key change
            closing_time = current_date.replace(hour=16, minute=0, second=0)

            bars = self.data_provider.get_intraday_bars(
                ticker=lookup_ticker,
                start_date=current_date,
                end_date=current_date,
                interval='1hour'
            )

            if bars is None or bars.empty:
                # Try to get data from the last week
                historical_bars = self.data_provider.get_intraday_bars(
                    ticker=lookup_ticker,
                    start_date=current_date,
                    end_date=current_date,
                    interval='1hour'
                )

                if historical_bars is not None and not historical_bars.empty:
                    closing_price = historical_bars.iloc[0]['close']
                    closing_time = historical_bars.index[0]
                else:
                    self.logger.warning("No recent data found - falling back to entry price")
                    closing_price = position.entry_price
            else:
                market_data = bars.iloc[0].to_dict()
                closing_price = market_data['close']
                closing_time = bars.index[0]

            # Calculate value (this part remains the same)
            if position.instrument.is_option:
                multiplier = position.instrument.get_multiplier()
                value = closing_price * multiplier
            else:
                value = closing_price

            # Close the position using last trading hour timestamp
            success = self.portfolio.close_position(
                instrument=position.instrument,
                price=closing_price,
                timestamp=closing_time, # type: ignore
                transaction_costs=self.broker.calculate_commission(
                    closing_price,
                    position.quantity,
                    'option' if position.instrument.is_option else 'stock'
                )
            )

            if success:
                self.logger.info(f"Automatically closed position in {ticker} @ {closing_price:.2f}")
            else:
                self.logger.error(f"Failed to close position in {ticker}")

        remaining = list(self.portfolio.positions.items())
        if remaining:
            self.logger.error("Failed to close all positions. Remaining positions:")
            for t, p in remaining:
                self.logger.error(f"  {t}: {p.quantity} units @ {p.entry_price}")
        else:
            self.logger.info("Automatically closed all positions")
