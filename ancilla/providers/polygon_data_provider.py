# ancilla/providers/polygon_data_provider.py
from typing import Optional, List, Dict, Union, Tuple, Callable, Any
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from polygon import RESTClient
import pytz
from functools import lru_cache
import time
from collections import defaultdict
from scipy.interpolate import griddata

from ancilla.models import OptionData, BarData, MarketSnapshot
from ancilla.utils.logging import MarketDataLogger

class PolygonDataProvider:
    """
    A robust data provider for Polygon.io API with standardized outputs and error handling.

    Features:
    - Consistent timezone handling (all timestamps in UTC)
    - Automatic rate limiting and retry logic
    - Robust error handling and logging
    - Caching of frequently accessed data
    - Data validation and cleaning
    """

    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Polygon data provider.

        Args:
            api_key: Polygon.io API key
            max_retries: Maximum number of API retry attempts
            retry_delay: Base delay between retries (uses exponential backoff)
        """
        self.client = RESTClient(api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.utc_tz = pytz.UTC

        # Set up logging
        self.logger = MarketDataLogger("polygon").get_logger()
        self.logger.info("Initializing Polygon data provider")

        # Cache settings
        self.cache_ttl = 300  # 5 minutes cache for prices
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Initialize option expirations cache
        self._option_expirations_cache = {}
        self._cache_update_time = None

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
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)

    def _validate_date_range(
        self,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None
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
            if ((dt.month == 12 and dt.day == 24) or  # Christmas Eve
                (dt.month == 11 and dt.weekday() == 4 and dt.day >= 23 and dt.day <= 29)):  # Day after Thanksgiving
                market_close = dt.replace(hour=13, minute=0, second=0, microsecond=0)

            return market_open <= dt <= market_close

        except Exception as e:
            self.logger.error(f"Error checking market hours: {str(e)}")
            return False

    def _validate_option_data(self, option: OptionData) -> bool:
        """Validate option data for reasonable values and consistency"""
        try:
            # Basic field validation
            if option.strike <= 0 or option.underlying_price <= 0:
                return False

            if option.contract_type not in ['call', 'put']:
                return False

            if option.implied_volatility <= 0 or option.implied_volatility > 5:
                return False

            # Greeks validation
            if option.delta is not None:
                if not -1 <= option.delta <= 1:
                    return False

            if option.gamma is not None:
                if option.gamma < 0:
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
            self.logger.error(f"Error validating option data: {str(e)}")
            return False

    def _validate_bar_data(self, bar: BarData) -> bool:
        """Validate price bar data for consistency"""
        try:
            # Price consistency
            if not (bar.low <= bar.high and
                   bar.low <= bar.open and
                   bar.low <= bar.close and
                   bar.high >= bar.open and
                   bar.high >= bar.close):
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

    @lru_cache(maxsize=128)
    def get_current_price(self, ticker: str) -> Optional[MarketSnapshot]:
        """
        Get the current price snapshot for a ticker with caching.

        Args:
            ticker: Stock symbol

        Returns:
            MarketSnapshot object or None if data unavailable
        """
        try:
            self.logger.debug(f"Fetching current price for {ticker}")
            snapshot = self._retry_with_backoff(
                self.client.get_snapshot_ticker,
                "stocks",
                ticker
            )

            if snapshot:
                now = datetime.now(self.utc_tz)

                # Extract price from session or fall back to prev_day
                price = None
                session = getattr(snapshot, 'session', None)
                if session:
                    price = session.close or session.last or None
                if price is None and hasattr(snapshot, 'prev_day'):
                    prev_day = snapshot.prev_day
                    price = prev_day.close if hasattr(prev_day, 'close') else None

                if price is None:
                    self.logger.warning(f"No valid price data available for {ticker}")
                    return None

                return MarketSnapshot(
                    timestamp=now,
                    price=price,
                    bid=float(session.bid) if session and hasattr(session, 'bid') else None,
                    ask=float(session.ask) if session and hasattr(session, 'ask') else None,
                    bid_size=int(session.bid_size) if session and hasattr(session, 'bid_size') else None,
                    ask_size=int(session.ask_size) if session and hasattr(session, 'ask_size') else None,
                    volume=int(session.volume) if session and hasattr(session, 'volume') else None,
                    vwap=float(session.vwap) if session and hasattr(session, 'vwap') else None
                )

            self.logger.warning(f"No snapshot data available for {ticker}")
            return None

        except Exception as e:
            self.logger.error(f"Error fetching current price for {ticker}: {str(e)}")
            return None

    def get_options_expiration(self, ticker: str) -> Optional[List[datetime]]:
        """
        Get available option expiration dates for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            List of expiration dates in UTC
        """
        try:
            self.logger.debug(f"Fetching option expirations for {ticker}")

            # Check cache
            now = datetime.now(self.utc_tz)
            if (ticker in self._option_expirations_cache and
                self._cache_update_time and
                (now - self._cache_update_time).total_seconds() < self.cache_ttl):
                return self._option_expirations_cache[ticker]

            # Get reference data for options
            contracts = self._retry_with_backoff(
                self.client.list_options_contracts,
                underlying_ticker=ticker,
                limit=1000  # Get a large number to ensure we get all expirations
            )

            if not contracts:
                self.logger.warning(f"No option contracts found for {ticker}")
                return None

            # Extract unique expiration dates
            expirations = set()
            for contract in contracts:
                try:
                    if hasattr(contract, 'expiration_date') and contract.expiration_date:
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

            # Update cache
            self._option_expirations_cache[ticker] = sorted_expirations
            self._cache_update_time = now

            self.logger.debug(f"Found {len(sorted_expirations)} expiration dates for {ticker}")
            return sorted_expirations

        except Exception as e:
            self.logger.error(f"Error fetching option expirations for {ticker}: {str(e)}")
            return None

    def get_options_chain(
        self,
        ticker: str,
        expiration_range_days: int = 90,
        delta_range: Tuple[float, float] = (0.1, 0.9),
        min_volume: int = 10
    ) -> Optional[List[OptionData]]:
        """
        Get the full options chain for a ticker with filtering.

        Args:
            ticker: Stock symbol
            expiration_range_days: How many days forward to fetch expirations
            delta_range: Only include options within this delta range
            min_volume: Minimum option volume to include

        Returns:
            List of OptionData objects
        """
        try:
            # Get current price first
            snapshot = self.get_current_price(ticker)
            if not snapshot:
                self.logger.error(f"Could not get current price for {ticker}")
                return None

            current_price = snapshot.price

            # Set up date range
            today = datetime.now(self.utc_tz)
            end_date = today + timedelta(days=expiration_range_days)

            self.logger.debug(f"Fetching options chain for {ticker}")
            options_data = []

            # Get options chain
            chain = self._retry_with_backoff(
                self.client.list_snapshot_options_chain,
                ticker,
                params={
                    "expiration_date.gte": today.strftime('%Y-%m-%d'),
                    "expiration_date.lte": end_date.strftime('%Y-%m-%d'),
                }
            )

            processed_count = 0
            skipped_count = 0

            for option in chain:
                try:
                    if not hasattr(option, 'details') or not option.details:
                        # self.logger.debug(f"Skipping option: Missing details for {ticker}")
                        skipped_count += 1
                        continue

                    details = option.details
                    contract_type = details.contract_type.lower()
                    strike = float(details.strike_price)
                    expiration = pd.to_datetime(details.expiration_date).tz_localize(self.eastern_tz)

                    # Greeks
                    if hasattr(option, 'greeks') and option.greeks:
                        delta = float(option.greeks.delta) if option.greeks.delta else None
                        gamma = float(option.greeks.gamma) if option.greeks.gamma else None
                        theta = float(option.greeks.theta) if option.greeks.theta else None
                        vega = float(option.greeks.vega) if option.greeks.vega else None
                    else:
                        delta = gamma = theta = vega = None

                    # Log why options are skipped due to delta range
                    if delta and not (delta_range[0] <= abs(delta) <= delta_range[1]):
                        # self.logger.debug(f"Skipping option: Delta {delta} out of range for {ticker}")
                        skipped_count += 1
                        continue

                    # Market data
                    if hasattr(option, 'last_quote') and option.last_quote:
                        bid = float(option.last_quote.bid) if option.last_quote.bid else None
                        ask = float(option.last_quote.ask) if option.last_quote.ask else None
                    else:
                        bid = ask = None

                    # Volume and OI
                    if hasattr(option, 'day') and option.day:
                        volume = int(option.day.volume) if option.day.volume else 0
                        open_interest = int(option.day.open_interest) if hasattr(option.day, 'open_interest') else None
                        if volume < min_volume:
                            # self.logger.debug(f"Skipping option: Volume {volume} below minimum for {ticker}")
                            skipped_count += 1
                            continue
                    else:
                        volume = 0
                        open_interest = None

                    # Implied volatility
                    if hasattr(option, 'implied_volatility') and option.implied_volatility:
                        iv = float(option.implied_volatility) / 100  # Convert to decimal
                    else:
                        # self.logger.debug(f"Skipping option: Missing implied volatility for {ticker}")
                        skipped_count += 1
                        continue

                    # Validate and add to options_data
                    opt_data = OptionData(
                        strike=strike,
                        expiration=expiration,
                        contract_type=contract_type,
                        implied_volatility=iv,
                        underlying_price=current_price,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                        bid=bid,
                        ask=ask,
                        volume=volume,
                        open_interest=open_interest,
                        last_trade=None  # Handle if last_trade is not present
                    )

                    if self._validate_option_data(opt_data):
                        options_data.append(opt_data)
                        processed_count += 1
                    else:
                        # self.logger.debug(f"Skipping option: Validation failed for {ticker}")
                        skipped_count += 1

                except Exception as e:
                    self.logger.warning(f"Error processing individual option for {ticker}: {str(e)}")
                    skipped_count += 1
                    continue

            self.logger.info(f"Processed {processed_count} options, skipped {skipped_count} invalid/filtered options")
            return options_data if options_data else None

        except Exception as e:
            self.logger.error(f"Error fetching options chain for {ticker}: {str(e)}")
            return None

    def get_daily_bars(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
        adjusted: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV bars for a ticker.

        Args:
            ticker: Stock symbol
            start_date: Start date
            end_date: End date (defaults to today)
            adjusted: Whether to return adjusted prices

        Returns:
            DataFrame with columns: [open, high, low, close, volume, vwap, returns, realized_vol]
        """
        try:
            self.logger.debug(f"Fetching daily bars for {ticker}")
            start_date, end_date = self._validate_date_range(start_date, end_date)

            aggs = self._retry_with_backoff(
                self.client.list_aggs,
                ticker,
                1,
                'day',
                start_date,
                end_date,
                adjusted=adjusted
            )

            if not aggs:
                self.logger.warning(f"No daily bars data for {ticker}")
                return None

            # Convert to BarData objects for validation
            bars = []
            for agg in aggs:
                try:
                    bar = BarData(
                        timestamp=pd.to_datetime(agg.timestamp, unit='ms', utc=True),
                        open=float(agg.open),
                        high=float(agg.high),
                        low=float(agg.low),
                        close=float(agg.close),
                        volume=int(agg.volume),
                        vwap=float(agg.vwap) if hasattr(agg, 'vwap') else None
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
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Calculate returns and volatility
            df['returns'] = df['close'].pct_change()
            df['realized_vol'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching daily bars for {ticker}: {str(e)}")
            return None

    def get_intraday_bars(
            self,
            ticker: str,
            start_date: Union[str, datetime, date],
            end_date: Optional[Union[str, datetime, date]] = None,
            interval: str = '1min',
            adjusted: bool = True
        ) -> Optional[pd.DataFrame]:
        """
        Get intraday price bars for a ticker.

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
            '1min': 'minute',
            '5min': 'minute',
            '15min': 'minute',
            '30min': 'minute',
            '1hour': 'hour'
        }

        try:
            if interval not in interval_map:
                raise ValueError(f"Invalid interval: {interval}")

            self.logger.debug(f"Fetching {interval} bars for {ticker}")
            start_date, end_date = self._validate_date_range(start_date, end_date)

            aggs = self._retry_with_backoff(
                self.client.list_aggs,
                ticker,
                interval_map[interval],
                interval.split('min')[0] if 'min' in interval else '1',
                start_date,
                end_date,
                adjusted=adjusted
            )

            if not aggs:
                self.logger.warning(f"No intraday bars data for {ticker}")
                return None

            bars = []
            for agg in aggs:
                try:
                    bar = BarData(
                        timestamp=pd.to_datetime(agg.timestamp, unit='ms', utc=True),
                        open=float(agg.open),
                        high=float(agg.high),
                        low=float(agg.low),
                        close=float(agg.close),
                        volume=int(agg.volume),
                        vwap=float(agg.vwap) if hasattr(agg, 'vwap') else None,
                        trades=int(agg.transactions) if hasattr(agg, 'transactions') else None
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
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Add trading session indicators
            df['regular_session'] = df.index.map(lambda x:
                self._is_regular_session(x.astimezone(self.eastern_tz)))

            return df

        except Exception as e:
            self.logger.error(f"Error fetching intraday bars for {ticker}: {str(e)}")
            return None

    def get_volatility_surface(
            self,
            ticker: str,
            target_date: datetime,
            moneyness_range: Tuple[float, float] = (0.7, 1.3)
        ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create a volatility surface from options data for a specific date/time.

        Args:
            ticker: Stock symbol.
            target_date: The reference date/time for calculating time to expiry.
            moneyness_range: Range of moneyness to include (K/S).

        Returns:
            Tuple of (X, Y, Z) arrays for surface plotting where:
            X = moneyness grid,
            Y = time to expiry grid,
            Z = implied volatility values.
        """
        try:
            self.logger.debug(f"Creating volatility surface for {ticker} based on {target_date}")

            # Ensure target_date is timezone-aware
            if target_date.tzinfo is None:
                target_date = target_date.replace(tzinfo=self.utc_tz)

            options_data = self.get_options_chain(ticker)
            if not options_data:
                return None

            # Convert to DataFrame for easier processing
            df: pd.DataFrame = pd.DataFrame([{
                'strike': opt.strike,
                'expiration': opt.expiration,
                'iv': opt.implied_volatility,
                'underlying_price': opt.underlying_price
            } for opt in options_data])

            # Ensure 'expiration' is timezone-aware
            if df['expiration'].dt.tz is None:
                df['expiration'] = df['expiration'].dt.tz_localize(self.utc_tz)

            # Calculate moneyness and time to expiry relative to target_date
            df['moneyness'] = df['strike'] / df['underlying_price']
            df['tte'] = df['expiration'].apply(lambda x: (x - target_date).total_seconds() / (365.0 * 24 * 3600))

            # Filter by moneyness and positive time to expiry
            df = df.loc[(df['moneyness'].between(*moneyness_range)) & (df['tte'] > 0)]

            if df.empty:
                self.logger.warning("No valid data points for volatility surface")
                return None

            # Create grid for surface
            money_points = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 50)
            tte_points = np.linspace(df['tte'].min(), df['tte'].max(), 50)
            X, Y = np.meshgrid(money_points, tte_points)

            # Interpolate implied volatilities
            moneyness_values = df['moneyness'].to_numpy(dtype=float)
            tte_values = df['tte'].to_numpy(dtype=float)
            iv_values = df['iv'].to_numpy(dtype=float)

            Z = griddata(
                (moneyness_values, tte_values),
                iv_values,
                (X, Y),
                method='cubic',
                fill_value=np.nan
            )

            # Fill any remaining NaN values using nearest neighbor
            mask = np.isnan(Z)
            if mask.any():
                Z[mask] = griddata(
                    (moneyness_values, tte_values),
                    iv_values,
                    (X[mask], Y[mask]),
                    method='nearest'
                )

            return X, Y, Z

        except Exception as e:
            self.logger.error(f"Error creating volatility surface for {ticker}: {str(e)}")
            return None

    def get_option_chain_stats(
        self,
        ticker: str
    ) -> Optional[Dict[str, Any]]:
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
                    if opt.contract_type == 'call':
                        total_call_volume += opt.volume
                    else:
                        total_put_volume += opt.volume

                # Volatility skew (25-delta options)
                if opt.delta:
                    abs_delta = abs(opt.delta)
                    if 0.2 <= abs_delta <= 0.3:
                        if opt.contract_type == 'call':
                            near_25_delta_calls.append(opt.implied_volatility)
                        else:
                            near_25_delta_puts.append(opt.implied_volatility)

                # ATM options for term structure
                moneyness = abs(opt.strike / current_price - 1)
                if moneyness < 0.02:  # Within 2% of ATM
                    atm_options.append({
                        'expiry': opt.expiration,
                        'iv': opt.implied_volatility
                    })

                # Gamma exposure
                if opt.gamma is not None and opt.volume:
                    total_gamma[opt.strike] += opt.gamma * opt.volume * 100  # Convert to 100 shares

            # Calculate statistics
            stats = {}

            # Put/Call ratio
            if total_call_volume > 0:
                stats['put_call_ratio'] = total_put_volume / total_call_volume
            else:
                stats['put_call_ratio'] = None

            # Volatility skew
            if near_25_delta_calls and near_25_delta_puts:
                stats['implied_volatility_skew'] = (
                    np.mean(near_25_delta_puts) / np.mean(near_25_delta_calls)
                )
            else:
                stats['implied_volatility_skew'] = None

            # Term structure
            term_structure = pd.DataFrame(atm_options)
            if not term_structure.empty:
                term_structure = (term_structure
                    .sort_values('expiry')
                    .set_index('expiry')['iv'])
                stats['term_structure'] = term_structure
            else:
                stats['term_structure'] = None

            # Gamma exposure
            stats['total_gamma'] = pd.Series(total_gamma).sort_index()

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating option chain statistics for {ticker}: {str(e)}")
            return None

    def get_historical_volatility(
            self,
            ticker: str,
            start_date: Union[str, datetime, date],
            end_date: Optional[Union[str, datetime, date]] = None
        ) -> Optional[pd.DataFrame]:
            """
            Calculate historical volatility metrics for a stock.

            Args:
                ticker: Stock symbol
                start_date: Start date
                end_date: End date (defaults to today)

            Returns:
                DataFrame with columns: [open, high, low, close, volume, vwap, returns, realized_vol, parkinson_vol, garman_klass_vol]
            """
            df = self.get_daily_bars(ticker, start_date, end_date, adjusted=True)
            if df is None or df.empty:
                return None
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['realized_vol'] = df['log_returns'].rolling(20).std() * np.sqrt(252)
            df['park_r'] = np.log(df['high']/df['low'])**2
            df['parkinson_vol'] = np.sqrt(df['park_r'].rolling(20).mean()/(4*np.log(2))) * np.sqrt(252)
            df['gk'] = 0.5 * (np.log(df['high']/df['low'])**2) - \
                       (2*np.log(2)-1)*((np.log(df['close']/df['open']))**2)
            df['garman_klass_vol'] = np.sqrt(df['gk'].rolling(20).mean()) * np.sqrt(252)
            return df

    def get_market_hours(
        self,
        date_input: Union[str, date],
        include_holidays: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Get market open and close times for a specific date.

        Args:
            date_input: Date to check
            include_holidays: Include holidays as market closed
        """
        if isinstance(date_input, str):
            date_input = pd.to_datetime(date_input).date()
        if date_input.weekday() >= 5:
            return None
        holidays_2024 = {
            # month/day
            (1, 1),   # New Year's (observed)
            (1, 15),  # MLK day
            (2, 19),  # Presidents Day
            (3, 29),  # Good Friday
            (5, 27),  # Memorial Day
            (6, 19),  # Juneteenth
            (7, 4),   # Independence Day
            (9, 2),   # Labor Day
            (11, 28), # Thanksgiving
            (12, 25), # Christmas
        }
        if (date_input.month, date_input.day) in holidays_2024:
            if include_holidays:
                return {'is_holiday': True}
            return None
        # Normal open/close times
        market_open_est = datetime(date_input.year, date_input.month, date_input.day, 9, 30, tzinfo=self.eastern_tz)
        market_close_est = datetime(date_input.year, date_input.month, date_input.day, 16, 0, tzinfo=self.eastern_tz)
        # Early close checks
        if (date_input.month == 12 and date_input.day == 24):
            market_close_est = market_close_est.replace(hour=13, minute=0)
        # Convert to UTC
        market_open_utc = market_open_est.astimezone(self.utc_tz)
        market_close_utc = market_close_est.astimezone(self.utc_tz)
        return {
            'market_open': market_open_utc,
            'market_close': market_close_utc
        }

    def clean_timeseries(
        self,
        df: pd.DataFrame,
        handle_missing: str = 'ffill',
        handle_outliers: bool = True,
        outlier_std: float = 3.0
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
        if handle_missing == 'ffill':
            df = df.ffill()
        elif handle_missing == 'bfill':
            df = df.bfill()
        else:
            df = df.dropna()
        if 'volume' in df.columns:
            df.loc[df['volume'] < 0, 'volume'] = 0
            df['volume'] = np.where(np.isinf(df['volume']), np.nan, df['volume'])
            df['volume'] = df['volume'].fillna(0)
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
