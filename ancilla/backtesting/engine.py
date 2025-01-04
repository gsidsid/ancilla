# ancilla/backtesting/engine.py
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import pytz
import logging

from ancilla.utils.logging import BacktesterLogger
from ancilla.providers.polygon import PolygonDataProvider
from ancilla.models import Instrument, Option, Stock, InstrumentType
from ancilla.backtesting.configuration import (
    Broker, CommissionConfig, SlippageConfig
)

if TYPE_CHECKING:
    from ancilla.backtesting import Strategy, Portfolio, BacktestResults


class Backtest:
    """Main backtesting engine with realistic broker simulation."""

    def __init__(
        self,
        data_provider: PolygonDataProvider,
        strategy: "Strategy",
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        tickers: List[str],
        commission_config: Optional[CommissionConfig] = None,
        slippage_config: Optional[SlippageConfig] = None,
        name: str = "backtesting",
        market_data = {}
    ):
        from ancilla.backtesting import Portfolio

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

        if ticker not in self.tickers:
            self.tickers.append(ticker)
        if ticker not in self.market_data:
            self.market_data.update(market_data)

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

    def run(self) -> "BacktestResults":
        """Run the backtest with hourly resolution using weekly batched data requests."""
        current_date = self.start_date
        consecutive_no_data = 0
        max_no_data_days = 5
        last_market_close = self.start_date

        eastern_tz = pytz.timezone('US/Eastern')
        utc_tz = pytz.UTC

        # Ensure initial current_date is timezone aware in Eastern
        if current_date.tzinfo is None:
            current_date = eastern_tz.localize(current_date)
        elif current_date.tzinfo != eastern_tz:
            current_date = current_date.astimezone(eastern_tz)

        while current_date <= self.end_date:
            # Calculate the end of the current week (Friday or end_date, whichever comes first)
            days_until_friday = (4 - current_date.weekday()) % 7
            week_end = min(
                current_date + timedelta(days=days_until_friday),
                self.end_date
            )

            # Pre-fetch a week's worth of intraday data for all tickers
            weekly_data = {}

            # Update tickers list with options from open positions
            open_positions = self.portfolio.positions.values()
            for position in open_positions:
                if position.instrument.is_option:
                    option_ticker = position.instrument.format_option_ticker()
                    if option_ticker not in self.tickers:
                        self.tickers.append(option_ticker)

            # Remove expired options from ticker list
            self.tickers = self._filter_expired_tickers(current_date)
            active_tickers = set(self.tickers)

            # Convert dates to UTC for data fetching
            fetch_start_utc = current_date.astimezone(utc_tz)
            fetch_end_utc = (week_end + timedelta(days=1)).astimezone(utc_tz)

            # Fetch weekly data for each active ticker
            for ticker in active_tickers:
                bars = self.data_provider.get_intraday_bars(
                    ticker=ticker,
                    start_date=fetch_start_utc,
                    end_date=fetch_end_utc,
                    interval='1hour'
                )
                if bars is not None and not bars.empty:
                    weekly_data[ticker] = bars

            # Process each day in the week
            while current_date <= week_end:
                # Get market hours in UTC
                market_hours = self.data_provider.get_market_hours(current_date.astimezone(utc_tz))
                if not market_hours:
                    current_date += timedelta(days=1)
                    continue

                market_open = market_hours['market_open']
                market_close = market_hours['market_close']

                # Convert string timestamps if needed and ensure UTC
                if isinstance(market_open, str):
                    market_open = datetime.fromisoformat(market_open.replace('Z', '+00:00'))
                if isinstance(market_close, str):
                    market_close = datetime.fromisoformat(market_close.replace('Z', '+00:00'))

                if market_open.tzinfo is None:
                    market_open = utc_tz.localize(market_open)
                if market_close.tzinfo is None:
                    market_close = utc_tz.localize(market_close)

                # Process each hour of the trading day
                current_time = market_open
                while current_time <= market_close:
                    market_data = self.market_data
                    has_data = False

                    # Extract relevant data for current hour from weekly data
                    for ticker in active_tickers:
                        if ticker not in weekly_data:
                            continue

                        bars = weekly_data[ticker]
                        window_start = current_time
                        window_end = current_time + timedelta(hours=1)

                        window_bars = bars[
                            (bars.index >= window_start) &
                            (bars.index < window_end)
                        ]

                        if not window_bars.empty:
                            market_data[ticker] = window_bars.iloc[0].to_dict()
                            has_data = True

                    if not has_data:
                        consecutive_no_data += 1
                        if consecutive_no_data >= max_no_data_days * 6.5:
                            self.logger.warning(
                                f"No market data for {max_no_data_days} consecutive days "
                                f"as of {current_time.astimezone(eastern_tz)}"
                            )
                            consecutive_no_data = 0
                    else:
                        consecutive_no_data = 0

                        # Convert current time to Eastern for display
                        eastern_time = current_time.astimezone(eastern_tz)
                        eastern_time_12_hour_format = eastern_time.strftime('%Y-%m-%d %I:%M %p')

                        # Update strategy logger format
                        strategy_formatter = logging.Formatter(
                            fmt=f"ancilla.{self.strategy.name} - [{eastern_time_12_hour_format}] - %(levelname)s - %(message)s"
                        )
                        self.strategy.logger.handlers[0].setFormatter(strategy_formatter)
                        self.strategy.logger.handlers[1].setFormatter(strategy_formatter)

                        # Update progress display
                        current_line = "\r" + "ancilla." + self.strategy.name + " – [" + eastern_time_12_hour_format + "]" + "\033[K"
                        print(current_line, end='\r')

                        # Merge new market data with existing market data
                        self.market_data.update(market_data)
                        self.last_timestamp = current_time

                        # When an order is executed in strategy.on_data(), _execute_instrument_order
                        # will update self.market_data with fresh data. That fresh data should be
                        # merged back into weekly_data to maintain consistency
                        self.strategy.on_data(current_time, self.market_data)

                        # After strategy execution, update weekly_data with any new data
                        # from executed orders
                        for ticker, data in self.market_data.items():
                            if ticker in weekly_data:
                                current_index = current_time
                                # Create a new row with the updated data
                                new_row = pd.Series(data, name=current_index)
                                # Update the specific row in weekly_data
                                weekly_data[ticker].loc[current_index] = new_row

                        # Update portfolio equity curve using merged market data
                        current_prices = {}
                        for ticker, data in self.market_data.items():
                            if 'close' in data:
                                current_prices[ticker] = data['close']

                        self.portfolio.update_equity(current_time, current_prices)

                    current_time += timedelta(hours=1)

                # Process option expirations at end of day (using Eastern time)
                eastern_current_date = current_date.astimezone(eastern_tz)
                eastern_market_close = market_close.astimezone(eastern_tz)
                last_market_close = eastern_market_close
                self._process_option_expiration(eastern_current_date, eastern_market_close)

                current_date += timedelta(days=1)
                if current_date.tzinfo is None:
                    current_date = eastern_tz.localize(current_date)

        # Close remaining positions at end of backtest
        self._close_all_positions(last_market_close)

        # Calculate and return results
        from ancilla.backtesting.results import BacktestResults
        results = BacktestResults.calculate(self)
        self.logger.info(results.summary())
        return results

    def _process_option_expiration(self, current_date: datetime, expiration_time: datetime):
        """
        Handle option expiration and assignments.
        """
        option_positions = [
                pos for pos in self.portfolio.positions.values()
                if (isinstance(pos.instrument, Option) and
                    pos.instrument.expiration.date() == current_date.date())
            ]

        if not option_positions:
            return

        # Then check if we're at Friday 4 PM
        is_expiration_time = (
            expiration_time.hour == 16 and
            expiration_time.minute == 0 and
            expiration_time.weekday() == 4  # Friday
        )

        if not is_expiration_time:
            return

        for position in option_positions:
            option: Option = position.instrument # type: ignore
            ticker = option.format_option_ticker()
            underlying_ticker = option.underlying_ticker

            # Get final underlying price
            expiry_close = expiration_time.replace(hour=16, minute=0)
            underlying_data = self.data_provider.get_intraday_bars(
                ticker=underlying_ticker,
                start_date=expiry_close,
                end_date=expiry_close + timedelta(minutes=1),
                interval='1hour'
            )

            if underlying_data is None or underlying_data.empty:
                self.logger.warning(f"No underlying data found for {underlying_ticker}")
                continue

            underlying_close = underlying_data.iloc[-1]['close']
            self.logger.info(f"Processing expiration for {ticker} with underlying at ${underlying_close:.2f}")

            # Calculate intrinsic value
            intrinsic_value = 0.0
            if isinstance(option, Option):
                if underlying_close:
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
                            market_data=self.market_data,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=True,
                            broker=self.broker
                        )
                    elif option.instrument_type == InstrumentType.PUT_OPTION:
                        # Short Put Assignment: Buy underlying stock at strike price
                        self.portfolio.handle_assignment(
                            option=option,
                            market_data=self.market_data,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=False,
                            broker=self.broker
                        )
                else:
                    # Long Option Exercise
                    if option.instrument_type == InstrumentType.CALL_OPTION:
                        # Long Call Exercise: Buy underlying stock at strike price
                        self.portfolio.handle_exercise(
                            option=option,
                            market_data=self.market_data,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=True,
                            intrinsic_value=intrinsic_value,
                            broker=self.broker
                        )
                    elif option.instrument_type == InstrumentType.PUT_OPTION:
                        # Long Put Exercise: Sell underlying stock at strike price
                        self.portfolio.handle_exercise(
                            option=option,
                            market_data=self.market_data,
                            strike_price=option.strike,
                            timestamp=expiration_time,
                            is_call=False,
                            intrinsic_value=intrinsic_value,
                            broker=self.broker
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
                        quantity=position.quantity,
                        transaction_costs=0.0
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
                        transaction_costs=0.0
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
        self.logger.info("End of test – automatically closing all positions")
        positions_to_close = list(self.portfolio.positions.items())
        self.logger.info(f"Found {len(positions_to_close)} open positions")
        for t, p in positions_to_close:
            self.logger.info(f"  {t}: {p.quantity} units @ {p.entry_price}")

        for ticker, position in positions_to_close:
            closing_time = current_date.replace(hour=16, minute=0, second=0)
            closing_price = self.market_data[ticker]['close']

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

    def _filter_expired_tickers(self, current_date):
        """
        Filter out expired options from ticker list and clean up market data.
        Only removes tickers that have been expired for more than a week.

        Args:
            current_date: datetime object representing the current date

        Returns:
            list: Active tickers that haven't expired
        """
        active_tickers = []
        expiration_buffer = timedelta(days=7)

        # Ensure current_date is a datetime and timezone-aware
        if not isinstance(current_date, datetime):
            raise TypeError(f"current_date must be a datetime object, got {type(current_date)}")

        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=pytz.UTC)

        for ticker in self.tickers:
            # Standard tickers (non-options) are always active
            if len(ticker) <= 6:
                active_tickers.append(ticker)
                continue

            try:
                # Parse option ticker
                option = Option.from_option_ticker(ticker)
                expiration_date = option.expiration

                # Ensure expiration_date is timezone-aware
                if expiration_date.tzinfo is None:
                    expiration_date = expiration_date.replace(tzinfo=pytz.UTC)

                # Calculate removal date as expiration date + buffer
                removal_date = expiration_date + expiration_buffer

                # Compare current date against removal date
                if current_date > removal_date:
                    self.market_data.pop(ticker, None)
                else:
                    active_tickers.append(ticker)

            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error processing option ticker {ticker}: {str(e)}")
                continue

        return active_tickers
