# ancilla/backtesting/engine.py
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING
import numpy as np
import pandas as pd
import pytz
import dotenv
import os
import logging

from ancilla.utils.logging import BacktesterLogger
from ancilla.providers import PolygonDataProvider, FREDDataProvider
from ancilla.models import Instrument, Option, Stock, InstrumentType, MarketDataDict
from ancilla.backtesting.configuration import Broker, CommissionConfig, SlippageConfig

if TYPE_CHECKING:
    from ancilla.backtesting import Strategy, BacktestResults

dotenv.load_dotenv()


class Backtest:
    """Backtesting engine."""

    def __init__(
        self,
        strategy: "Strategy",
        initial_capital: float = 100000.0,
        start_date: datetime = datetime.now() - timedelta(days=365),
        end_date: datetime = datetime.now(),
        tickers: List[str] = ["SPY"],
        frequency: str = "30min",  # realistically either 30min or 1hour
        enable_naked_options: bool = True,
        commission_config: CommissionConfig = CommissionConfig(),
        slippage_config: SlippageConfig = SlippageConfig(),
        deterministic_fill: bool = False,
        name: str = "backtesting",
    ) -> None:
        """Initialize a new backtesting instance.

        Args:
            strategy (Strategy): Strategy instance to be backtested (must be a subclass of ancilla.backtesting.Strategy).
            initial_capital (float, optional): Starting capital in base currency (USD). Defaults to 100000.0.
            start_date (datetime, optional): Backtest period start date. Defaults to 1 year ago.
            end_date (datetime, optional): Backtest period end date. Defaults to current date.
            tickers (List[str], optional): List of asset symbols to trade. Defaults to ["SPY"].
            frequency (str, optional): Data sampling frequency, either "30min" or "1hour". Defaults to "30min".
            enable_naked_options (bool, optional): Allow trading options without holding underlying. Defaults to True.
            commission_config (ancilla.backtesting.configuration.CommissionConfig): Trading fee structure.
            slippage_config (ancilla.backtesting.configuration.SlippageConfig): Price slippage model.
            deterministic_fill (bool, optional): If True, ignores volume based fill probabilities and reliably fills orders. Defaults to False.
            name (str, optional): Identifier for backtest engine logs. Defaults to "backtesting".
        """
        from ancilla.backtesting import Portfolio

        if not isinstance(commission_config, CommissionConfig):
            raise ValueError(
                "CommissionConfig must be an instance of ancilla.backtesting.configuration.CommissionConfig"
            )
        if not isinstance(slippage_config, SlippageConfig):
            raise ValueError(
                "SlippageConfig must be an instance of ancilla.backtesting.configuration.SlippageConfig"
            )
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise ValueError("start_date and end_date must be datetime objects")
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        if frequency not in [
            "1min",
            "5min",
            "15min",
            "30min",
            "1hour",
            "2hour",
            "1day",
        ]:
            raise ValueError(
                "Invalid frequency. Must be one of: 1min, 5min, 15min, 30min, 1hour, 2hour, 1day"
            )
        if initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number")
        if os.getenv("POLYGON_API_KEY") is None:
            raise ValueError(
                "POLYGON_API_KEY environment variable not set. Get a key at https://polygon.io/"
            )
        if os.getenv("FRED_API_KEY") is None:
            raise ValueError(
                "FRED_API_KEY environment variable not set. Get a key at https://fred.stlouisfed.org/"
            )

        name = f"{strategy.name}_orders"
        self.portfolio = Portfolio(name, initial_capital, enable_naked_options)
        self.initial_capital = initial_capital
        self.data_provider = PolygonDataProvider(
            api_key=os.getenv("POLYGON_API_KEY") or "your-polygon-io-api-key"
        )
        self.strategy = strategy
        self.frequency = frequency

        fred_provider = FREDDataProvider(
            api_key=os.getenv("FRED_API_KEY") or "your-fred-api-key"
        )
        self.risk_free_rates = fred_provider.get_fed_funds_rate(
            start_date=start_date - timedelta(days=1),
            end_date=end_date + timedelta(days=1),
        )

        self.risk_free_rate = 0.0
        rate_index = pd.Timestamp(start_date.replace(day=1, tzinfo=None).date())
        if self.risk_free_rates is not None and not self.risk_free_rates.empty:
            self.risk_free_rate = self.risk_free_rates[rate_index]

        self.logger = BacktesterLogger().get_logger()

        # Set timezone to Eastern Time
        start_date = start_date.astimezone(pytz.timezone("US/Eastern")).replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        end_date = end_date.astimezone(pytz.timezone("US/Eastern")).replace(
            hour=16, minute=0, second=0, microsecond=0
        )

        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.weekly_data = {}
        self.dividend_data = self._fetch_dividend_data()

        # Initialize market data
        self.market_data = {}
        self.last_timestamp = start_date

        # Initialize market simulator
        self.broker = Broker(commission_config, slippage_config, deterministic_fill)

        # History for market data and analytics
        self.market_history = {}

        # Trade analytics
        self.fill_ratios = []  # Track fill rates
        self.slippage_costs = []  # Track slippage
        self.commission_costs = []  # Track commissions
        self.total_transaction_costs = (
            []
        )  # Track total transaction costs for reconciliation

        # Payout analytics
        self.dividend_payouts = []  # Track dividend payouts
        self.interest_payouts = []  # Track interest payouts

        # Initialize strategy
        self.strategy.initialize(self.portfolio, self)

        # Cache for daily metrics
        self.daily_metrics = {
            "slippage": [],
            "commissions": [],
            "fills": [],
            "volume_participation": [],
        }

    def _execute_instrument_order(
        self,
        instrument: Instrument,
        quantity: int,
        timestamp: datetime,
        is_assignment: bool = False,
    ) -> bool:
        """
        Internal method to execute orders with broker simulation.
        """
        # Get latest market data
        ticker = (
            instrument.format_option_ticker()
            if instrument.is_option
            else instrument.underlying_ticker
        )
        week_end = timestamp + timedelta(days=(4 - timestamp.weekday()) % 7)
        bars = self.data_provider.get_intraday_bars(
            ticker=ticker,
            start_date=timestamp.astimezone(pytz.UTC),
            end_date=week_end.astimezone(pytz.UTC),
            interval=self.frequency,
        )
        if bars is None or bars.empty:
            self.logger.warning(
                f"No bars found for {instrument.ticker} on {timestamp}. It is likely not trading currently."
            )
            return False

        # Update weekly data and tickers if ordering a new option or equity
        if ticker not in self.weekly_data:
            self.weekly_data[ticker] = bars
        if ticker not in self.tickers:
            self.tickers.append(ticker)

        # Update market data with first bar for order execution
        data = bars.iloc[0].to_dict()
        self.market_data[ticker] = data

        # Get target price
        price = data.get("open", 0)
        if price is None:
            self.logger.warning(f"No price data for {instrument.ticker}")
            return False

        # Calculate execution details using broker simulation
        execution_details = self.broker.calculate_execution_details(
            ticker=ticker,
            base_price=price,
            quantity=quantity,
            market_data=data,
            asset_type="option" if instrument.is_option else "stock",
        )
        if not execution_details or execution_details.adjusted_quantity == 0:
            self.logger.warning(
                f"Failed to calculate execution details for order ({ticker}), likely due to low volume. Attempted: {instrument.ticker if not instrument.is_option else instrument.format_option_ticker()}, {quantity} @ {price}"
                + "\n"
                + f"\t  Market data: {data}"
            )
            return False

        # Use fill probability from execution details
        fill_probability = 1.0 if is_assignment else execution_details.fill_probability
        if not is_assignment and np.random.random() > fill_probability:
            self.logger.warning(
                f"Order failed to fill: {instrument.ticker} {execution_details.adjusted_quantity} @ {execution_details.execution_price:.2f} "
                f"(fill probability @ {fill_probability:.2%})"
            )
            return False

        # Create, update, or close a position
        success = False
        quantity = execution_details.adjusted_quantity
        position_exists = ticker in self.portfolio.positions
        position_args = {
            "price": execution_details.execution_price,
            "timestamp": timestamp,
            "transaction_costs": execution_details.total_transaction_costs,
        }
        if position_exists:
            # Close existing position
            success = self.portfolio.close_position(
                instrument=self.portfolio.positions[ticker].instrument, **position_args
            )
        else:
            # Open new position
            success = self.portfolio.open_position(
                instrument=instrument, quantity=quantity, **position_args
            )

        # Track trade metrics
        if success:
            self.daily_metrics["volume_participation"].append(
                execution_details.participation_rate
            )
            self.daily_metrics["fills"].append(1.0)
            self.daily_metrics["slippage"].append(execution_details.price_impact)
            self.daily_metrics["commissions"].append(execution_details.commission)
            self.fill_ratios.append(fill_probability)
            self.slippage_costs.append(execution_details.slippage)
            self.commission_costs.append(execution_details.commission)
            self.total_transaction_costs.append(
                execution_details.total_transaction_costs
            )
            self.logger.info(
                f"Order executed: {ticker} {quantity} @ {execution_details.execution_price:.2f}\n"
                f"  Base price: ${price:.2f}\n"
                f"  Impact: {execution_details.price_impact:.2%}\n"
                f"  Commission: ${execution_details.commission:.2f}\n"
                f"  Volume participation: {execution_details.participation_rate:.2%}"
            )

        return success

    def run(self) -> "BacktestResults":
        """Run the backtest."""
        current_date = self.start_date
        last_market_close = self.start_date
        frequency_delta = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "1hour": timedelta(hours=1),
            "2hour": timedelta(hours=2),
            "1day": timedelta(days=1),
        }[self.frequency]

        while current_date <= self.end_date:
            # Find the end of the current week
            days_until_friday = (4 - current_date.weekday()) % 7
            week_end = min(
                current_date + timedelta(days=days_until_friday), self.end_date
            )

            # Update options tickers
            self.tickers = self._update_option_tickers(current_date - timedelta(days=1))

            # Batch-fetch data for each active ticker this trading week
            # Will used cached data if available at backtest frequency
            for ticker in self.tickers:
                bars = self.data_provider.get_intraday_bars(
                    ticker=ticker,
                    start_date=current_date.astimezone(pytz.UTC),
                    end_date=(week_end + timedelta(days=1)).astimezone(pytz.UTC),
                    interval=self.frequency,
                )
                if bars is not None and not bars.empty:
                    self.weekly_data[ticker] = bars

            # Process each day in the trading week
            while current_date <= week_end:
                market_hours = self.data_provider.get_market_hours(current_date)
                if not market_hours:
                    current_date += timedelta(days=1)
                    continue
                market_open = market_hours["market_open"]
                market_close = market_hours["market_close"]
                current_time = market_open

                while current_time <= market_close:
                    self.last_timestamp = current_time
                    self._update_risk_free_rate(current_time)

                    # Find current bars in weekly pull for each ticker
                    for ticker in self.tickers:
                        bars = self.weekly_data[ticker]
                        window_start = (current_time - frequency_delta).astimezone(
                            pytz.UTC
                        )
                        window_end = (current_time).astimezone(pytz.UTC)
                        window_bars = bars[
                            (bars.index > window_start) & (bars.index <= window_end)
                        ]
                        if window_bars.empty:
                            continue
                        self.market_data[ticker] = window_bars.iloc[0].to_dict()
                        self.market_data[ticker]["timestamp"] = current_time

                    # Apply dividend adjustments for the current time
                    # This needs to happen right after market data is updated but before strategy execution
                    self._adjust_for_dividends(current_time, self.market_data)

                    # Legible strategy progress display
                    eastern_time_12_hour_format = current_time.astimezone(
                        pytz.timezone("US/Eastern")
                    ).strftime("%Y-%m-%d %I:%M %p")
                    strategy_formatter = logging.Formatter(
                        fmt=f"ancilla.{self.strategy.name} - [{eastern_time_12_hour_format}] - %(levelname)s - %(message)s"
                    )
                    self.strategy.logger.handlers[0].setFormatter(strategy_formatter)
                    self.strategy.logger.handlers[1].setFormatter(strategy_formatter)
                    current_line = (
                        "\r"
                        + "ancilla."
                        + self.strategy.name
                        + " – ["
                        + eastern_time_12_hour_format
                        + "]"
                        + "\033[K"
                    )
                    print(current_line, end="\r")

                    # Execute backtest strategy with dividend-adjusted market data
                    market_data_with_indicators = MarketDataDict(
                        self.market_data,
                        self.market_history,
                        risk_free_rate=float(self.risk_free_rate / 100.0),
                    )
                    self.strategy.on_data(current_time, market_data_with_indicators)
                    for ticker in self.market_data:
                        if ticker not in self.market_history:
                            self.market_history[ticker] = []
                        self.market_data[ticker]["timestamp"] = current_time
                        self.market_history[ticker].append(self.market_data[ticker])

                    # Update portfolio equity curve prices as of strategy execution
                    # Note: These prices are already dividend-adjusted
                    current_prices = {}
                    for ticker, data in self.market_data.items():
                        if "close" in data:
                            current_prices[ticker] = data["close"]
                    self.portfolio.update_equity(current_time, current_prices)

                    current_time += frequency_delta

                # Attempt to process option expirations
                # Options only clear here on Fridays at 4:00 PM
                # So there may be hanging expired positions
                last_market_close = market_close
                self._process_option_expiration(
                    current_date, market_close.astimezone(pytz.timezone("US/Eastern"))
                )

                # If the current day is the last day of the month, payout monthly interest
                if (
                    current_date.day
                    == current_date.replace(day=1).replace(tzinfo=None).date().day
                ):
                    interest_payout = self.portfolio.payout_interest(
                        float(self.risk_free_rate / 100.0)
                    )
                    self.interest_payouts.append(interest_payout)

                current_date += timedelta(days=1)

        # Close remaining positions at end of backtest
        self._close_all_positions(last_market_close)

        # Calculate and return results
        from ancilla.backtesting.results import BacktestResults

        results = BacktestResults.calculate(self)
        self.logger.info(results.summary())
        return results

    def _process_option_expiration(
        self, current_date: datetime, expiration_time: datetime
    ):
        """
        Handle option expiration and assignments.
        """
        option_positions = [
            pos
            for pos in self.portfolio.positions.values()
            if (
                isinstance(pos.instrument, Option)
                and pos.instrument.expiration.date() == current_date.date()
                and pos.instrument.expiration.year == current_date.year
            )
        ]
        if not option_positions:
            return

        is_expiration_time = (
            expiration_time.hour == 16
            and expiration_time.minute == 0
            and expiration_time.weekday() == 4  # Friday
        )
        if not is_expiration_time:
            return

        for position in option_positions:
            option: Option = position.instrument  # type: ignore
            ticker = option.format_option_ticker()
            underlying_ticker = option.underlying_ticker

            # Get final underlying price
            expiry_close = expiration_time.replace(hour=16, minute=0)
            underlying_data = self.data_provider.get_intraday_bars(
                ticker=underlying_ticker,
                start_date=expiry_close,
                end_date=expiry_close + timedelta(minutes=1),
                interval=self.frequency,
            )

            if underlying_data is None or underlying_data.empty:
                self.logger.warning(f"No underlying data found for {underlying_ticker}")
                continue

            underlying_close = underlying_data.iloc[-1]["close"]
            self.logger.info(
                f"Processing expiration for {ticker} with underlying at ${underlying_close:.2f}"
            )

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

            # Determine the action based on position and option types
            if is_itm:
                if position.quantity < 0:
                    # Short Option Assignment
                    self.portfolio.handle_assignment(
                        option=option,
                        strike_price=option.strike,
                        timestamp=expiration_time,
                        is_call=option.instrument_type == InstrumentType.CALL_OPTION,
                        broker=self.broker,
                    )
                else:
                    # Long Option Exercise
                    # TODO: Only do this if we have the paper? Verify this logic is sound
                    self.portfolio.handle_exercise(
                        option=option,
                        strike_price=option.strike,
                        timestamp=expiration_time,
                        is_call=option.instrument_type == InstrumentType.CALL_OPTION,
                        intrinsic_value=intrinsic_value,
                        broker=self.broker,
                    )
            else:
                # Option expires worthless
                # Close the option position without any cash impact
                self.logger.info(f"Option {ticker} expired worthless. Premium kept.")
                self.portfolio.close_position(
                    instrument=option,
                    price=0.0,  # Option expired worthless
                    timestamp=expiration_time,
                    quantity=position.quantity,
                    transaction_costs=0.0,
                )

            if ticker not in self.portfolio.positions:
                if ticker in self.tickers:
                    self.tickers.remove(ticker)
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
            closing_price = self.market_data[ticker]["close"]

            # Close the position using last trading hour timestamp
            success = self.portfolio.close_position(
                instrument=position.instrument,
                price=closing_price,
                timestamp=closing_time,  # type: ignore
                transaction_costs=self.broker.calculate_commission(
                    closing_price,
                    position.quantity,
                    "option" if position.instrument.is_option else "stock",
                ),
            )
            if success:
                self.logger.info(
                    f"Automatically closed position in {ticker} @ {closing_price:.2f}"
                )
            else:
                self.logger.error(f"Failed to close position in {ticker}")

        remaining = list(self.portfolio.positions.items())
        if remaining:
            self.logger.error("Failed to close all positions. Remaining positions:")
            for t, p in remaining:
                self.logger.error(f"  {t}: {p.quantity} units @ {p.entry_price}")
        else:
            self.logger.info("Automatically closed all positions")

    def _update_option_tickers(self, current_date):
        """
        Add tickers for new option positions to the list of active tickers.
        Filter out expired options from ticker list and clean up market data.
        Only remove tickers that have been expired for more than a week.

        Args:
            current_date: datetime object representing the current date

        Returns:
            list: Active tickers that haven't expired
        """
        active_tickers = []
        expiration_buffer = timedelta(days=7)

        # Add option tickers for any new positions to self.tickers
        open_positions = self.portfolio.positions.values()
        for position in open_positions:
            if position.instrument.is_option:
                option_ticker = position.instrument.format_option_ticker()
                if option_ticker not in self.tickers:
                    self.tickers.append(option_ticker)

        # Ensure current_date is a datetime and timezone-aware
        if not isinstance(current_date, datetime):
            raise TypeError(
                f"current_date must be a datetime object, got {type(current_date)}"
            )

        if current_date.tzinfo is None:
            current_date = current_date.replace(tzinfo=pytz.UTC)

        # Remove options tickers that are expired beyond a buffer for their clearing time
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
                self.logger.warning(
                    f"Error processing option ticker {ticker}: {str(e)}"
                )
                continue

        # Ensure the tickers are de-duplicated
        return list(set(active_tickers))

    def buy_stock(
        self, ticker: str, quantity: int, _is_assignment: bool = False
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
            is_assignment=_is_assignment,
        )

    def sell_stock(
        self, ticker: str, quantity: int, _is_assignment: bool = False
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
            is_assignment=_is_assignment,
        )

    def buy_option(self, option: Option, quantity: int) -> bool:
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
            instrument=option, quantity=abs(quantity), timestamp=self.last_timestamp
        )

    def sell_option(self, option: Option, quantity: int) -> bool:
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
            instrument=option, quantity=-abs(quantity), timestamp=self.last_timestamp
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
            self.logger.error(
                f"Error validating option order {option.ticker}: {str(e)}"
            )
            return False

    def _fetch_dividend_data(self):
        """
        Fetch dividend data for all tickers in the backtest period.
        Returns a dictionary mapping tickers to their dividend DataFrames.
        """
        dividend_data = {}

        # Format dates for Polygon API (YYYY-MM-DD)
        formatted_start = self.start_date.strftime("%Y-%m-%d")
        formatted_end = self.end_date.strftime("%Y-%m-%d")

        for ticker in self.tickers:
            # Skip option tickers
            if len(ticker) > 6:  # Basic check for option tickers
                continue

            dividends_df = self.data_provider.get_dividends(
                ticker=ticker, start_date=formatted_start, end_date=formatted_end
            )

            if dividends_df is not None and not dividends_df.empty:
                dividend_data[ticker] = dividends_df
                self.logger.info(f"Found {len(dividends_df)} dividends for {ticker}")

        return dividend_data

    def _adjust_for_dividends(self, current_time: datetime, market_data: dict):
        """
        Adjust market data for dividends on ex-dividend dates.
        This modifies the market_data dictionary in place.
        """
        current_date = current_time.date()
        # Only apply adjustment at market open UTC (9:30 AM ET)
        is_market_open = current_time.strftime("%H:%M") == "14:30"
        if not is_market_open:
            return

        for ticker, data in market_data.items():
            # Skip if not a stock ticker or no dividend data
            if len(ticker) > 6 or ticker not in self.dividend_data:
                continue

            dividends = self.dividend_data[ticker]
            dividend_dates = dividends.index.date

            # Check if there's a dividend on this date
            if current_date in dividend_dates:
                dividend_date_str = current_date.isoformat()
                dividend_amount = dividends[
                    dividends.index.strftime("%Y-%m-%d") == dividend_date_str
                ]["amount"].iloc[0]

                # Adjust OHLC prices by adding the dividend amount
                # This simulates the price drop that occurs on ex-dividend date
                data["open"] += dividend_amount
                data["high"] += dividend_amount
                data["low"] += dividend_amount
                data["close"] += dividend_amount

                dividend_payout = self.portfolio.payout_dividend(
                    dividend_ticker=ticker,
                    cash_amount=dividend_amount,
                )
                self.dividend_payouts.append(dividend_payout)
                self.logger.debug(
                    f"Adjusted {ticker} prices for ${dividend_amount:.2f} dividend on {current_date}"
                )

    def _update_risk_free_rate(self, current_time: datetime):
        """
        Update the interest rate for the current date.
        """
        rate_index = pd.Timestamp(current_time.replace(day=1, tzinfo=None).date())
        try:
            if self.risk_free_rates is not None:
                self.risk_free_rate = self.risk_free_rates[rate_index]
        except KeyError:
            self.logger.warning(
                f"No rate data found for {rate_index} – using last known rate, {self.risk_free_rate}%"
            )
