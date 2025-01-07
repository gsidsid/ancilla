from datetime import datetime
from typing import Optional, Dict, Any
import pytz
import time
import pandas as pd

from fredapi import Fred
from ancilla.utils.caching import HybridCache
from ancilla.utils.logging import MarketDataLogger


class FREDDataProvider:
    def __init__(self, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the FRED data provider with caching capabilities.

        Args:
            api_key: FRED API key
            max_retries: Maximum number of API retry attempts
            retry_delay: Base delay between retries (uses exponential backoff)
        """
        self.fred = Fred(api_key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set up timezone
        self.eastern_tz = pytz.timezone("US/Eastern")
        self.utc_tz = pytz.UTC

        # Set up logging
        self.logger = MarketDataLogger("fred").get_logger()
        self.logger.debug("Initializing Fred data provider")

        # Initialize cache
        self.cache = HybridCache()

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

    def get_series(
        self, series_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.Series]:
        """
        Get FRED time series data with caching.

        Args:
            series_id: FRED series identifier
            start_date: Start date for the series
            end_date: End date for the series

        Returns:
            Optional[pd.Series]: Time series data if available
        """
        try:
            # Generate cache key
            cache_params = {
                "series_id": series_id,
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            }
            cache_key = (
                f"series_{series_id}_{cache_params['start']}_{cache_params['end']}"
            )

            # Check cache
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                return pd.Series(cached_data)

            # Fetch from API
            series = self._retry_with_backoff(
                self.fred.get_series,
                series_id,
                observation_start=cache_params["start"],
                observation_end=cache_params["end"],
            )

            if series is not None:
                # Cache the result
                self._cache_data(series.to_dict(), cache_key)
                return series

            return None

        except Exception as e:
            self.logger.error(f"Error fetching series {series_id}: {str(e)}")
            self.logger.error(
                f"Request: {locals().get('cache_params', 'Not available')}"
            )
            return None

    def get_fed_funds_rate(
        self, start_date: datetime, end_date: datetime
    ) -> Optional[pd.Series]:
        """
        Convenience method to get Fed Funds rates for a given date range.

        Args:
            start_date: Start date for the rate series
            end_date: End date for the rate series

        Returns:
            Optional[pd.Series]: Fed funds rate time series if available
        """
        return self.get_series("FEDFUNDS", start_date, end_date)

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Ensure minimum time between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)

                result = func(*args, **kwargs)
                self.last_request_time = time.time()
                return result

            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(
                        f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
                    raise e

                wait_time = self.retry_delay * (2**attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}"
                )
                time.sleep(wait_time)

    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache if available."""
        return self.cache.get(cache_key)

    def _cache_data(self, data: Dict[str, Any], cache_key: str):
        """Store data in cache."""
        self.cache.set(cache_key, data)
