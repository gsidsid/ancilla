# ancilla/providers/polygon.py
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Union, Tuple, Callable, Any

# Third party imports
import numpy as np
import pandas as pd
from polygon import RESTClient
from zoneinfo import ZoneInfo
import pytz
import time

# Local imports
from ancilla.models import OptionData, BarData, MarketSnapshot, Option
from ancilla.utils.caching import HybridCache
from ancilla.utils.logging import MarketDataLogger


class PolygonDataProvider:
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Polygon data provider with multi-level caching.

        Args:
            api_key: Polygon.io API key
            max_retries: Maximum number of API retry attempts
            retry_delay: Base delay between retries (uses exponential backoff)
            cache_dir: Directory for file cache
        """
        self.client = RESTClient(api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.eastern_tz = pytz.timezone("US/Eastern")
        self.utc_tz = pytz.UTC

        # Set up logging
        self.logger = MarketDataLogger("polygon").get_logger()
        self.logger.debug("Initializing Polygon data provider")

        # Initialize cache
        self.cache = HybridCache(
            memory_ttl=300,  # 5 minutes memory cache
            file_ttl=86400,  # 24 hours file cache
            cleanup_interval=3600,  # Cleanup every hour
        )

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.025

    def get_current_price(self, ticker: str) -> Optional[MarketSnapshot]:
        """Get current price snapshot with short-term caching."""
        try:
            # Use shorter cache duration for current price
            cache_params = {"ticker": ticker}
            cached_data = self._get_cached_data("current_price", **cache_params)
            if cached_data is not None:
                return MarketSnapshot(**cached_data)

            snapshot = self._retry_with_backoff(
                self.client.get_snapshot_ticker, "stocks", ticker
            )

            if snapshot:
                now = datetime.now(self.utc_tz)

                # Extract price data
                price = None
                session = getattr(snapshot, "session", None)
                if session:
                    price = session.close or session.last or None
                if price is None and hasattr(snapshot, "prev_day"):
                    prev_day = snapshot.prev_day
                    price = prev_day.close if hasattr(prev_day, "close") else None

                if price is None:
                    self.logger.warning(f"No valid price data available for {ticker}")
                    return None

                # Create snapshot object
                market_snapshot = MarketSnapshot(
                    timestamp=now,
                    price=price,
                    bid=(
                        float(session.bid)
                        if session and hasattr(session, "bid")
                        else None
                    ),
                    ask=(
                        float(session.ask)
                        if session and hasattr(session, "ask")
                        else None
                    ),
                    bid_size=(
                        int(session.bid_size)
                        if session and hasattr(session, "bid_size")
                        else None
                    ),
                    ask_size=(
                        int(session.ask_size)
                        if session and hasattr(session, "ask_size")
                        else None
                    ),
                    volume=(
                        int(session.volume)
                        if session and hasattr(session, "volume")
                        else None
                    ),
                    vwap=(
                        float(session.vwap)
                        if session and hasattr(session, "vwap")
                        else None
                    ),
                )

                # Cache the result
                self._cache_data(vars(market_snapshot), "current_price", **cache_params)

                return market_snapshot

            return None

        except Exception as e:
            self.logger.error(f"Error fetching current price for {ticker}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_options_expiration(self, ticker: str) -> Optional[List[datetime]]:
        """
        Get available option expiration dates for a ticker with caching.

        Args:
            ticker: Stock symbol

        Returns:
            List of expiration dates in UTC
        """
        try:
            # Check cache first
            cache_params = {"ticker": ticker}
            cached_data = self._get_cached_data("options_expiration", **cache_params)
            if cached_data is not None:
                # Convert ISO format strings back to datetime objects
                return [pd.to_datetime(exp_date) for exp_date in cached_data]

            self.logger.debug(f"Fetching option expirations for {ticker}")

            # Get reference data for options
            contracts = self._retry_with_backoff(
                self.client.list_options_contracts,
                underlying_ticker=ticker,
                limit=1000,  # Get a large number to ensure we get all expirations
            )

            if not contracts:
                self.logger.warning(f"No option contracts found for {ticker}")
                return None

            # Extract unique expiration dates
            expirations = set()
            for contract in contracts:
                try:
                    if (
                        hasattr(contract, "expiration_date")
                        and contract.expiration_date
                    ):
                        # Parse expiration date and convert to datetime
                        expiry = pd.to_datetime(contract.expiration_date)
                        # Ensure timezone awareness (assume Eastern)
                        if expiry.tzinfo is None:
                            expiry = self.eastern_tz.localize(expiry)
                        # Convert to UTC
                        expiry = expiry.astimezone(self.utc_tz)
                        expirations.add(expiry)
                except Exception as e:
                    self.logger.warning(f"Error processing expiration date: {str(e)}")
                    continue

            # Sort expiration dates
            sorted_expirations = sorted(list(expirations))

            if sorted_expirations:
                # Cache the result as ISO format strings for better serialization
                cache_data = [exp_date.isoformat() for exp_date in sorted_expirations]
                self._cache_data(cache_data, "options_expiration", **cache_params)

            self.logger.debug(
                f"Found {len(sorted_expirations)} expiration dates for {ticker}"
            )
            return sorted_expirations

        except Exception as e:
            self.logger.error(
                f"Error fetching option expirations for {ticker}: {str(e)}"
            )
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_options_chain(
        self,
        ticker: str,
        reference_date: Optional[datetime] = None,
        expiration_range_days: int = 90,
        min_days: int = 7,
        min_volume: int = 10,
        delta_range: Tuple[float, float] = (0.1, 0.9),
        interpolation_days: Optional[int] = None,
    ) -> Optional[List[OptionData]]:
        """
        Get the options chain for a ticker with caching and filtering.

        Args:
            ticker: Stock symbol.
            reference_date: The date to reference for fetching options (defaults to now).
            expiration_range_days: Number of days forward to fetch expirations.
            min_days: Minimum days until expiration.
            min_volume: Minimum option volume to include.
            delta_range: Only include options within this delta range (magnitude).
            interpolation_days: (Optional) Additional parameter for future use.

        Returns:
            List of OptionData objects or None if no data is available.
        """
        try:
            # Define cache parameters
            cache_params = {
                "ticker": ticker,
                "reference_date": (
                    reference_date.isoformat() if reference_date else None
                ),
                "expiration_range_days": expiration_range_days,
                "min_days": min_days,
                "min_volume": min_volume,
                "delta_range": delta_range,
                "interpolation_days": interpolation_days,
            }

            # Attempt to retrieve cached data
            cached_data = self._get_cached_data("options_chain", **cache_params)
            if cached_data is not None:
                self.logger.debug(
                    f"Cache hit for {ticker}. Returning cached options chain."
                )
                # Reconstruct OptionData objects from cached data
                return [OptionData(**opt_dict) for opt_dict in cached_data]

            self.logger.debug(
                f"No cache found for {ticker}. Fetching options chain from API."
            )

            # Get price as of reference date, if reference date is today use current price
            current_price = None
            if (
                reference_date is None
                or reference_date.date() == datetime.now(self.utc_tz).date()
            ):
                snapshot = self.get_current_price(ticker)
                if not snapshot:
                    self.logger.error(f"Could not get current price for {ticker}")
                    return None

                current_price = snapshot.price
            else:
                # Get the closing price for the reference date
                daily_bars = self.get_daily_bars(ticker, reference_date, reference_date)
                if daily_bars is None or daily_bars.empty:
                    self.logger.error(
                        f"No daily bars found for {ticker} on {reference_date}"
                    )
                    return None
                current_price = daily_bars["close"].iloc[0]

            # Set up date range
            now = reference_date if reference_date else datetime.now(self.utc_tz)
            min_expiry = now + timedelta(days=min_days)
            # max expiry is the reference date + min days + expiration range
            max_expiry = now + timedelta(days=expiration_range_days)

            self.logger.info(
                f"Querying {ticker} chain: T+ {min_days}d to {expiration_range_days}d, Ref={now}"
            )
            chain_generator = self._retry_with_backoff(
                self.client.list_snapshot_options_chain,
                ticker,
                params={
                    "expiration_date.gte": min_expiry.strftime("%Y-%m-%d"),
                    "expiration_date.lte": max_expiry.strftime("%Y-%m-%d"),
                    "as_of": now.strftime("%Y-%m-%d"),
                },
            )
            chain = list(chain_generator)
            if not chain:
                self.logger.info("No options returned from API")
                return None

            processed_count = 0
            skipped_count = 0
            skipped_due_to_volume = 0
            skipped_due_to_expiry = 0
            skipped_due_to_delta = 0
            skipped_due_to_iv = 0
            processed_options = []

            for option in chain:
                try:
                    details = option.details
                    contract_type = details.contract_type.lower()
                    strike = float(details.strike_price)
                    expiration = pd.to_datetime(details.expiration_date).tz_localize(
                        self.eastern_tz
                    )
                    days_to_expiry = (expiration - now).days

                    # Check expiration range
                    days_to_expiry = (expiration - now).days
                    if (
                        days_to_expiry < min_days
                        or days_to_expiry > expiration_range_days
                    ):
                        skipped_count += 1
                        skipped_due_to_expiry += 1
                        continue

                    # Greeks
                    if hasattr(option, "greeks") and option.greeks:
                        delta = (
                            float(option.greeks.delta) if option.greeks.delta else None
                        )
                        gamma = (
                            float(option.greeks.gamma) if option.greeks.gamma else None
                        )
                        theta = (
                            float(option.greeks.theta) if option.greeks.theta else None
                        )
                        vega = float(option.greeks.vega) if option.greeks.vega else None
                    else:
                        self.logger.debug(f"No Greeks for {option.details.ticker}")
                        delta = gamma = theta = vega = None

                    # Check delta range using absolute value
                    if delta is not None:
                        if not (delta_range[0] <= abs(delta) <= delta_range[1]):
                            skipped_count += 1
                            skipped_due_to_delta += 1
                            continue

                    # Volume
                    volume = 0
                    if hasattr(option, "day") and option.day:
                        volume = option.day.volume
                        if volume and volume < min_volume:
                            skipped_count += 1
                            skipped_due_to_volume += 1
                            continue

                    # Create OptionData object
                    opt_data = OptionData(
                        ticker=option.details.ticker,
                        strike=strike,
                        expiration=expiration,
                        contract_type=contract_type,
                        implied_volatility=option.implied_volatility,
                        underlying_price=current_price,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                        bid=None,
                        ask=None,
                        volume=volume,
                        open_interest=option.open_interest,
                        last_trade=None,
                    )

                    # Validate and add to processed options
                    if self._validate_option_data(opt_data, days_to_expiry):
                        processed_options.append(opt_data)
                        processed_count += 1
                    else:
                        skipped_count += 1
                        self.logger.debug(
                            f"Skipping option due to failed validation: {ticker}"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Error processing option {ticker} for {ticker}: {str(e)}"
                    )
                    skipped_count += 1
                    continue

            # Create a detailed summary log
            summary_lines = [
                f"{ticker} T+ {min_days}d to {expiration_range_days}d Ref={now}",
                "═" * 50,
                f"✓ Successfully processed: {processed_count} options",
                f"✗ Total skipped/filtered: {skipped_count} options",
                "─" * 50,
                "Skipped Options Breakdown:",
                f"• Low volume (<{min_volume}):        {skipped_due_to_volume:>6}",
                f"• Out-of-range expiry:     {skipped_due_to_expiry:>6}",
                f"• Out-of-range delta:      {skipped_due_to_delta:>6}",
                f"• Missing IV data:         {skipped_due_to_iv:>6}",
                "═" * 50,
            ]
            self.logger.info("\n" + "\n".join(summary_lines) + "\n")

            # Cache the result as dictionaries
            if processed_options:
                cache_data = [vars(opt) for opt in processed_options]
                self._cache_data(cache_data, "options_chain", **cache_params)
                self.logger.debug(f"Options chain for {ticker} cached successfully.")
                return processed_options
            return None
        except Exception as e:
            self.logger.error(f"Error fetching options chain for {ticker}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_options_contracts(
        self,
        ticker: str,
        as_of: datetime,
        strike_range: Tuple[float, float],
        min_expiration_days: int = 0,
        max_expiration_days: int = 365,
        contract_type: Optional[str] = None,
    ) -> Optional[List["Option"]]:
        """
        Get available option contracts for a ticker.

        Args:
            ticker: Stock symbol.
            as_of: Reference date for fetching options.
            strike_range: Tuple of minimum and maximum strike prices.
            contract_type: Filter by 'call' or 'put' contracts.

        Returns:
            List of Option objects or None if no data is available.
        """
        try:
            # Define cache parameters
            cache_params = {
                "ticker": ticker,
                "as_of": as_of.isoformat(),
                "strike_range": strike_range,
                "min_expiration_days": min_expiration_days,
                "max_expiration_days": max_expiration_days,
                "contract_type": contract_type,
            }

            # Attempt to retrieve cached data
            cached_data = self._get_cached_data("options_contracts", **cache_params)
            if cached_data is not None:
                self.logger.debug(
                    f"Cache hit for {ticker}. Returning cached options contracts."
                )
                return [Option(**opt_dict) for opt_dict in cached_data]

            self.logger.debug(
                f"No cache found for {ticker}. Fetching options contracts from API."
            )

            # Calculate max expiration date
            min_expiry = as_of + timedelta(days=min_expiration_days)
            max_expiry = as_of + timedelta(days=max_expiration_days)

            # Get options contracts from API
            contracts = self._retry_with_backoff(
                self.client.list_options_contracts,
                underlying_ticker=ticker,
                contract_type=contract_type,
                strike_price_gte=strike_range[0],
                strike_price_lte=strike_range[1],
                expiration_date_gte=min_expiry.strftime("%Y-%m-%d"),
                expiration_date_lte=max_expiry.strftime("%Y-%m-%d"),
                as_of=as_of.strftime("%Y-%m-%d"),
            )

            if not contracts:
                self.logger.info("No options returned from API")
                return None

            processed_count = 0
            skipped_count = 0
            processed_contracts = []

            for contract in contracts:
                try:
                    contract_type = contract.contract_type.lower()
                    strike = float(contract.strike_price)

                    # Check contract type
                    # There are rare "other" types that we skip
                    if contract_type != contract_type or contract_type not in [
                        "call",
                        "put",
                    ]:
                        skipped_count += 1
                        continue

                    # Check strike range
                    if not (strike_range[0] <= strike <= strike_range[1]):
                        skipped_count += 1
                        continue

                    # Create Option object
                    expiry = pd.to_datetime(contract.expiration_date)
                    if expiry.tzinfo is None:
                        expiry = pytz.UTC.localize(expiry)

                    option = Option(
                        ticker=contract.underlying_ticker,
                        strike=float(contract.strike_price),
                        expiration=expiry,
                        option_type=contract.contract_type.lower(),  # Ensure lowercase
                    )

                    processed_contracts.append(option)
                    processed_count += 1

                except Exception as e:
                    self.logger.warning(
                        f"Error processing option {ticker} for {ticker}: {str(e)}"
                    )
                    skipped_count += 1
                    continue

            # Create a detailed summary log
            summary_lines = [
                f"{ticker} Options Contracts",
                "═" * 50,
                f"✓ Successfully processed: {processed_count} contracts",
                f"✗ Total skipped/filtered: {skipped_count} contracts",
                "═" * 50,
            ]
            self.logger.debug("\n" + "\n".join(summary_lines) + "\n")

            # Cache the result as dictionaries
            if processed_contracts:
                cache_data = [vars(opt) for opt in processed_contracts]
                self._cache_data(cache_data, "options_contracts", **cache_params)
                self.logger.debug(
                    f"Options contracts for {ticker} cached successfully."
                )
                return processed_contracts

            return None
        except Exception as e:
            self.logger.error(
                f"Error fetching options contracts for {ticker}: {str(e)}"
            )
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_daily_bars(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get daily OHLCV bars with caching."""
        try:
            # Check cache first
            cache_params = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "adjusted": adjusted,
            }
            cached_data = self._get_cached_data("daily_bars", **cache_params)
            if cached_data is not None:
                return pd.DataFrame(cached_data)

            # If not in cache, fetch from API
            start_date, end_date = self._validate_date_range(start_date, end_date)

            aggs = self._retry_with_backoff(
                self.client.list_aggs,
                ticker,
                1,
                "day",
                start_date,
                end_date,
                adjusted=adjusted,
            )

            if not aggs:
                return None

            # Process the data
            bars = []
            for agg in aggs:
                try:
                    bar = BarData(
                        timestamp=pd.to_datetime(agg.timestamp, unit="ms", utc=True),
                        open=float(agg.open),
                        high=float(agg.high),
                        low=float(agg.low),
                        close=float(agg.close),
                        volume=int(agg.volume),
                        vwap=float(agg.vwap) if hasattr(agg, "vwap") else None,
                    )
                    if self._validate_bar_data(bar):
                        bars.append(bar)
                except Exception as e:
                    self.logger.warning(f"Error processing bar: {str(e)}")
                    continue

            if not bars:
                return None

            # Convert to DataFrame
            df = pd.DataFrame([vars(bar) for bar in bars])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Calculate returns and volatility
            df["returns"] = df["close"].pct_change()
            df["realized_vol"] = df["returns"].rolling(window=20).std() * np.sqrt(252)

            # Cache the result
            self._cache_data(df.to_dict(), "daily_bars", **cache_params)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching daily bars for {ticker}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_intraday_bars(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
        interval: str = "1min",
        adjusted: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday price bars for a ticker with caching.

        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date (defaults to today)
            interval: Time interval ('1min', '5min', '15min', '30min', '1hour')
            adjusted: Whether to return adjusted prices

        Returns:
            DataFrame with OHLCV data and regular_session indicator
        """
        interval_map = {
            "1min": "minute",
            "5min": "minute",
            "15min": "minute",
            "30min": "minute",
            "1hour": "hour",
        }

        try:
            if interval not in interval_map:
                raise ValueError(f"Invalid interval: {interval}")

            # Generate cache parameters
            cache_params = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "adjusted": adjusted,
            }

            # Check cache first
            cached_data = self._get_cached_data("intraday_bars", **cache_params)
            if cached_data is not None:
                # Reconstruct DataFrame from cached data
                df = pd.DataFrame(cached_data)
                if not df.empty:
                    # Convert index back to datetime
                    df.index = pd.to_datetime(df.index)
                    return df

            self.logger.debug(f"Fetching {interval} bars for {ticker}")
            start_date, end_date = self._validate_date_range(start_date, end_date)

            multiplier = interval.split("min")[0] if "min" in interval else "1"
            aggs = self._retry_with_backoff(
                self.client.list_aggs,
                ticker,
                multiplier,
                timespan=interval_map[interval],
                from_=start_date,
                to=end_date,
                adjusted=adjusted,
            )

            if not aggs:
                self.logger.warning(f"No intraday bars data for {ticker}")
                return None

            bars = []
            for agg in aggs:
                try:
                    bar = BarData(
                        timestamp=pd.to_datetime(agg.timestamp, unit="ms", utc=True),
                        open=float(agg.open),
                        high=float(agg.high),
                        low=float(agg.low),
                        close=float(agg.close),
                        volume=int(agg.volume),
                        vwap=float(agg.vwap) if hasattr(agg, "vwap") else None,
                        trades=(
                            int(agg.transactions)
                            if hasattr(agg, "transactions")
                            else None
                        ),
                    )
                    if self._validate_bar_data(bar):
                        bars.append(bar)
                except Exception as e:
                    self.logger.warning(f"Error processing intraday bar: {str(e)}")
                    continue

            if not bars:
                return None

            # Convert to DataFrame
            df = pd.DataFrame([vars(bar) for bar in bars])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)

            # Add trading session indicators
            df["regular_session"] = df.index.map(
                lambda x: self._is_regular_session(x.astimezone(self.eastern_tz))
            )

            # Cache the result
            # Convert index to strings for serialization
            cache_data = df.copy()
            cache_data.index = cache_data.index.astype(str)
            self._cache_data(cache_data.to_dict(), "intraday_bars", **cache_params)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching intraday bars for {ticker}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_option_chain_stats(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for the full options chain.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary containing:
            - put_call_ratio: Volume-weighted P/C ratio
            - implied_volatility_skew: 25-delta put/call IV ratio
            - term_structure: Array of ATM IVs by expiration
            - total_gamma: Total gamma exposure by strike
        """
        try:
            self.logger.debug(f"Calculating option chain statistics for {ticker}")
            options_data = self.get_options_chain(ticker)
            if not options_data:
                return None

            # Current price for moneyness calculations
            current_price = options_data[0].underlying_price

            # Initialize containers
            total_call_volume = 0
            total_put_volume = 0
            near_25_delta_calls = []
            near_25_delta_puts = []
            atm_options = []
            total_gamma = defaultdict(float)

            for opt in options_data:
                # Put/Call ratio
                if opt.volume:
                    if opt.contract_type == "call":
                        total_call_volume += opt.volume
                    else:
                        total_put_volume += opt.volume

                # Volatility skew (25-delta options)
                if opt.delta:
                    abs_delta = abs(opt.delta)
                    if 0.2 <= abs_delta <= 0.3:
                        if opt.contract_type == "call":
                            near_25_delta_calls.append(opt.implied_volatility)
                        else:
                            near_25_delta_puts.append(opt.implied_volatility)

                # ATM options for term structure
                moneyness = abs(opt.strike / current_price - 1)
                if moneyness < 0.02:  # Within 2% of ATM
                    atm_options.append(
                        {"expiry": opt.expiration, "iv": opt.implied_volatility}
                    )

                # Gamma exposure
                if opt.gamma is not None and opt.volume:
                    total_gamma[opt.strike] += (
                        opt.gamma * opt.volume * 100
                    )  # Convert to 100 shares

            # Calculate statistics
            stats = {}

            # Put/Call ratio
            if total_call_volume > 0:
                stats["put_call_ratio"] = total_put_volume / total_call_volume
            else:
                stats["put_call_ratio"] = None

            # Volatility skew
            if near_25_delta_calls and near_25_delta_puts:
                stats["implied_volatility_skew"] = np.mean(
                    near_25_delta_puts
                ) / np.mean(near_25_delta_calls)
            else:
                stats["implied_volatility_skew"] = None

            # Term structure
            term_structure = pd.DataFrame(atm_options)
            if not term_structure.empty:
                term_structure = term_structure.sort_values("expiry").set_index(
                    "expiry"
                )["iv"]
                stats["term_structure"] = term_structure
            else:
                stats["term_structure"] = None

            # Gamma exposure
            stats["total_gamma"] = pd.Series(total_gamma).sort_index()

            return stats

        except Exception as e:
            self.logger.error(
                f"Error calculating option chain statistics for {ticker}: {str(e)}"
            )
            return None

    def get_historical_volatility(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> Optional[pd.DataFrame]:
        """Get historical volatility metrics with caching."""
        try:
            # Check cache first
            cache_params = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            }
            cached_data = self._get_cached_data("historical_volatility", **cache_params)
            if cached_data is not None:
                return pd.DataFrame(cached_data)

            df = self.get_daily_bars(ticker, start_date, end_date, adjusted=True)
            if df is None or df.empty:
                return None

            # Calculate volatility metrics
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            df["realized_vol"] = df["log_returns"].rolling(20).std() * np.sqrt(252)
            df["park_r"] = np.log(df["high"] / df["low"]) ** 2
            df["parkinson_vol"] = np.sqrt(
                df["park_r"].rolling(20).mean() / (4 * np.log(2))
            ) * np.sqrt(252)
            df["gk"] = 0.5 * (np.log(df["high"] / df["low"]) ** 2) - (
                2 * np.log(2) - 1
            ) * ((np.log(df["close"] / df["open"])) ** 2)
            df["garman_klass_vol"] = np.sqrt(df["gk"].rolling(20).mean()) * np.sqrt(252)

            # Cache the result
            self._cache_data(df.to_dict(), "historical_volatility", **cache_params)

            return df
        except Exception as e:
            self.logger.error(
                f"Error calculating historical volatility for {ticker}: {str(e)}"
            )
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_market_hours(
        self, date_input: Union[str, date], include_holidays: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get market open and close times for a specific date.

        Args:
            date_input: Date to check
            include_holidays: Include holidays as market closed
        """
        if isinstance(date_input, str):
            date_input = datetime.fromisoformat(date_input).date()
        if date_input.weekday() >= 5:
            return None
        holidays_2024 = {
            # month/day
            (1, 1),  # New Year's (observed)
            (1, 15),  # MLK day
            (2, 19),  # Presidents Day
            (3, 29),  # Good Friday
            (5, 27),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),  # Independence Day
            (9, 2),  # Labor Day
            (11, 28),  # Thanksgiving
            (12, 25),  # Christmas
        }
        if (date_input.month, date_input.day) in holidays_2024:
            if include_holidays:
                return {"is_holiday": True}
            return None
        # Normal open/close times
        market_open_est = datetime(
            date_input.year, date_input.month, date_input.day, 9, 30
        )
        market_open_est = market_open_est.replace(tzinfo=ZoneInfo("America/New_York"))

        market_close_est = datetime(
            date_input.year, date_input.month, date_input.day, 16, 0
        )
        market_close_est = market_close_est.replace(tzinfo=ZoneInfo("America/New_York"))

        # Early close checks
        if date_input.month == 12 and date_input.day == 24:
            market_close_est = market_close_est.replace(hour=13, minute=0)

        # Convert to UTC
        market_open_utc = market_open_est.astimezone(ZoneInfo("UTC"))
        market_close_utc = market_close_est.astimezone(ZoneInfo("UTC"))
        return {"market_open": market_open_utc, "market_close": market_close_utc}

    def get_dividends(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> Optional[pd.DataFrame]:
        """Get dividend data with caching."""
        try:
            # Format dates to YYYY-MM-DD strings if they aren't already
            if isinstance(start_date, (datetime, date)):
                start_date = start_date.strftime("%Y-%m-%d")
            if end_date and isinstance(end_date, (datetime, date)):
                end_date = end_date.strftime("%Y-%m-%d")

            # Check cache first
            cache_params = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            }
            cached_data = self._get_cached_data("dividends", **cache_params)
            if cached_data is not None:
                return pd.DataFrame(cached_data)

            # Skip _validate_date_range since we've already formatted the dates
            # This was likely converting dates to nanosecond timestamps

            dividends = self._retry_with_backoff(
                self.client.list_dividends,
                ticker=ticker,
                ex_dividend_date_gte=start_date,
                ex_dividend_date_lte=end_date,
                sort="ex_dividend_date",
                order="desc",
                limit=1000,
            )

            if not dividends:
                return None

            # Process the data
            div_data = []
            for div in dividends:
                try:
                    div_data.append(
                        {
                            "date": pd.to_datetime(div.ex_dividend_date),
                            "amount": float(div.cash_amount),
                            "type": div.dividend_type,
                            "frequency": div.frequency,
                            "ticker": div.ticker,
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error processing dividend: {str(e)}")
                    continue

            if not div_data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(div_data)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Cache the result
            self._cache_data(df.to_dict(), "dividends", **cache_params)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching dividends for {ticker}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    ##############################
    # VALIDATION/CACHING HELPERS
    ##############################

    def clean_timeseries(
        self,
        df: pd.DataFrame,
        handle_missing: str = "ffill",
        handle_outliers: bool = True,
        outlier_std: float = 3.0,
    ) -> pd.DataFrame:
        """
        Clean and preprocess a timeseries DataFrame.

        Args:
            df: Input DataFrame
            handle_missing: Method to handle missing values ('ffill', 'bfill', 'drop')
            handle_outliers: Whether to handle outliers
            outlier_std: Number of standard deviations for outlier detection

        Returns:
            Cleaned DataFrame
        """
        if handle_missing == "ffill":
            df = df.ffill()
        elif handle_missing == "bfill":
            df = df.bfill()
        else:
            df = df.dropna()
        if "volume" in df.columns:
            df.loc[df["volume"] < 0, "volume"] = 0
            df["volume"] = np.where(np.isinf(df["volume"]), np.nan, df["volume"])
            df["volume"] = df["volume"].fillna(0)
        if handle_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                upper_bound = mean_val + outlier_std * std_val
                lower_bound = mean_val - outlier_std * std_val
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        return df

    def _generate_cache_key(self, method: str, **params) -> str:
        """Generate a standardized cache key."""
        parts = [method]
        for k, v in sorted(params.items()):
            # Handle different types of values
            if isinstance(v, (datetime, date)):
                v = v.isoformat()
            elif isinstance(v, (list, tuple)):
                v = ",".join(str(x) for x in v)
            parts.append(f"{k}={v}")
        return ":".join(parts)

    def _get_cached_data(self, method: str, **params) -> Optional[Any]:
        """Get data from cache using standardized key."""
        cache_key = self._generate_cache_key(method, **params)
        return self.cache.get(cache_key)

    def _cache_data(self, data: Any, method: str, **params) -> None:
        """Cache data using standardized key."""
        if data is not None:
            cache_key = self._generate_cache_key(method, **params)
            self.cache.set(cache_key, data)

    def _rate_limit(self) -> None:
        """Implement rate limiting between API calls"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _retry_with_backoff(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Max retries reached: {str(e)}")
                    raise
                wait_time = self.retry_delay * (2**attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

    def _validate_date_range(
        self,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
    ) -> Tuple[datetime, datetime]:
        """Validate and standardize date inputs"""
        try:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            if end_date is None:
                end_date = datetime.now(self.utc_tz)

            # Convert to UTC datetime objects
            if isinstance(start_date, date):
                start_date = datetime.combine(start_date, datetime.min.time())
            if isinstance(end_date, date):
                end_date = datetime.combine(end_date, datetime.max.time())

            # Ensure timezone awareness
            if start_date.tzinfo is None:
                start_date = self.eastern_tz.localize(start_date)
            if end_date.tzinfo is None:
                end_date = self.eastern_tz.localize(end_date)

            # Convert to UTC
            start_date = start_date.astimezone(self.utc_tz)
            end_date = end_date.astimezone(self.utc_tz)

            return start_date, end_date

        except Exception as e:
            self.logger.error(f"Error validating date range: {str(e)}")
            raise ValueError("Invalid date range provided")

    def _is_regular_session(self, dt: datetime) -> bool:
        """Check if timestamp is during regular trading hours (9:30-16:00 ET)"""
        try:
            if dt.tzinfo is None:
                dt = self.eastern_tz.localize(dt)
            elif dt.tzinfo != self.eastern_tz:
                dt = dt.astimezone(self.eastern_tz)

            # Check for weekends
            if dt.weekday() >= 5:
                return False

            # Standard market hours
            market_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = dt.replace(hour=16, minute=0, second=0, microsecond=0)

            # Check for early closes
            if (dt.month == 12 and dt.day == 24) or (  # Christmas Eve
                dt.month == 11 and dt.weekday() == 4 and dt.day >= 23 and dt.day <= 29
            ):  # Day after Thanksgiving
                market_close = dt.replace(hour=13, minute=0, second=0, microsecond=0)

            return market_open <= dt <= market_close

        except Exception as e:
            self.logger.error(f"Error checking market hours: {str(e)}")
            return False

    def _validate_option_data(self, option: OptionData, days_to_expiry: int) -> bool:
        """Validate option data with expiry-dependent criteria."""
        try:
            # Basic field validation
            if option.strike <= 0 or option.underlying_price <= 0:
                return False

            if option.contract_type not in ["call", "put"]:
                return False

            # Adjust IV bounds based on time to expiry
            max_iv = 5.0  # 500% vol cap for short dated
            if days_to_expiry > 60:
                max_iv = 2.0  # 200% vol cap for longer dated
            elif days_to_expiry > 180:
                max_iv = 1.5  # 150% vol cap for very long dated

            if (
                option.implied_volatility is None
                or option.implied_volatility <= 0
                or option.implied_volatility > max_iv
            ):
                return False

            # Greeks validation - more permissive for longer dated
            if option.delta is not None:
                if not -1 <= option.delta <= 1:
                    return False

            if option.gamma is not None:
                if option.gamma < 0:
                    return False
                # Add upper bound check for gamma
                if days_to_expiry <= 30 and option.gamma > 1:
                    return False

            # Market data validation
            if option.volume is not None:
                if option.volume < 0:
                    return False

            if option.bid is not None and option.ask is not None:
                if option.bid > option.ask:
                    return False

            # Expiration validation
            now = datetime.now(self.utc_tz)
            if option.expiration < now:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating option data: {str(e)}, {option}")
            return False

    def _validate_bar_data(self, bar: BarData) -> bool:
        """Validate price bar data for consistency"""
        try:
            # Price consistency
            if not (
                bar.low <= bar.high
                and bar.low <= bar.open
                and bar.low <= bar.close
                and bar.high >= bar.open
                and bar.high >= bar.close
            ):
                return False

            # Volume should be non-negative
            if bar.volume < 0:
                return False

            # VWAP should be within high/low range if present
            if bar.vwap is not None:
                if not (bar.low <= bar.vwap <= bar.high):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating bar data: {str(e)}")
            return False
