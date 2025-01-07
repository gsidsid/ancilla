import os
import pytest
import dotenv
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict

from ancilla.providers.polygon import PolygonDataProvider
from ancilla.models import MarketSnapshot, OptionData

UTC = pytz.UTC
EST = pytz.timezone("US/Eastern")

dotenv.load_dotenv()

############################################################
#                   MOCK-BASED UNIT TESTS
############################################################


@pytest.fixture
def mock_client():
    """Create a mock Polygon REST client."""
    return Mock()


@pytest.fixture
def provider(mock_client):
    """
    Create a PolygonDataProvider instance with a mock client
    for unit tests.
    """
    with patch("ancilla.providers.polygon.RESTClient") as mock_rest:
        mock_rest.return_value = mock_client
        pdp = PolygonDataProvider(api_key="test-api-key")
        pdp.cache.clear()
        yield pdp


def create_mock_agg(timestamp: int, open_, high, low, close, volume, vwap=None) -> Mock:
    agg = Mock()
    agg.timestamp = timestamp
    agg.open = open_
    agg.high = high
    agg.low = low
    agg.close = close
    agg.volume = volume
    agg.vwap = vwap
    return agg


def create_mock_option(
    strike: float,
    expiry: str,
    contract_type: str,
    iv: float,
    greeks: Dict[str, float],
    quote: Dict[str, float],
    volume: int,
) -> Mock:
    option = Mock()
    option.details = Mock()
    option.details.strike_price = strike
    option.details.expiration_date = expiry
    option.details.contract_type = contract_type
    option.greeks = Mock()
    for greek, value in greeks.items():
        setattr(option.greeks, greek, value)
    option.last_quote = Mock()
    option.last_quote.bid = quote.get("bid")
    option.last_quote.ask = quote.get("ask")
    option.day = Mock()
    option.day.volume = volume
    option.implied_volatility = iv * 100  # stored as a percentage
    return option


class TestPolygonDataProviderMock:
    """Mock-based unit tests for PolygonDataProvider."""

    def test_init(self, provider):
        assert provider.max_retries == 3
        assert provider.retry_delay == 1.0
        assert provider.eastern_tz == EST
        assert provider.utc_tz == UTC

    def test_retry_with_backoff(self, provider):
        """Ensure retry logic will re-attempt the correct number of times."""
        mock_func = Mock(
            side_effect=[Exception("Test error"), Exception("Test error"), "success"]
        )
        result = provider._retry_with_backoff(mock_func)
        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_with_backoff_max_retries(self, provider):
        """Ensure retry logic raises after max retries are reached."""
        mock_func = Mock(side_effect=Exception("Test error"))
        with pytest.raises(Exception):
            provider._retry_with_backoff(mock_func)
        assert mock_func.call_count == provider.max_retries

    def test_validate_date_range(self, provider):
        start, end = provider._validate_date_range("2024-01-01", "2024-01-31")
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start.tzinfo == UTC
        assert end.tzinfo == UTC

    def test_is_regular_session(self, provider):
        # 2:30 PM EST on a weekday
        dt = EST.localize(datetime(2024, 1, 2, 14, 30))
        assert provider._is_regular_session(dt)
        # 9:00 AM EST (before open)
        dt = EST.localize(datetime(2024, 1, 2, 9, 0))
        assert not provider._is_regular_session(dt)

    def test_error_handling(self, provider, mock_client):
        """Test error handling & retries in get_current_price."""
        mock_client.get_snapshot_ticker.side_effect = Exception("API Error")
        snapshot = provider.get_current_price("AAPL")
        assert snapshot is None
        assert mock_client.get_snapshot_ticker.call_count == provider.max_retries

    def test_data_cleaning(self, provider):
        test_data = pd.DataFrame(
            {
                "close": [100.0, 101.0, np.nan, 102.0, 1000.0, 103.0],
                "volume": [1000, -100, 1200, np.inf, 1400, 1500],
                "vwap": [100.1, 101.1, 101.5, np.nan, np.nan, 103.1],
            },
            index=pd.date_range("2024-01-01", periods=6, freq="D", tz=UTC),
        )

        cleaned = provider.clean_timeseries(
            test_data, handle_missing="ffill", handle_outliers=True, outlier_std=3.0
        )

        # Check that NaN values were forward-filled
        assert not cleaned["close"].isna().any()
        assert not cleaned["vwap"].isna().any()
        # Check volumes are >= 0 and not inf
        assert (cleaned["volume"] >= 0).all()
        assert not np.isinf(cleaned["volume"]).any()

    def test_option_greeks_validation(self, provider):
        valid_option = OptionData(
            ticker="AAPL",
            strike=100.0,
            expiration=datetime.now(UTC) + timedelta(days=30),
            contract_type="call",
            implied_volatility=0.25,
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            underlying_price=100.0,
            volume=100,
        )
        assert provider._validate_option_data(valid_option, days_to_expiry=30)

        invalid_option = OptionData(
            ticker="AAPL",
            strike=100.0,
            expiration=datetime.now(UTC) + timedelta(days=30),
            contract_type="call",
            implied_volatility=6.0,  # >5 is invalid
            delta=0.5,
            gamma=0.02,
            theta=-0.05,
            vega=0.15,
            underlying_price=100.0,
            volume=100,
        )
        assert not provider._validate_option_data(invalid_option, days_to_expiry=30)

    def test_intraday_bars_mock(self, provider, mock_client):
        """
        Unit test for intraday bars with mocked aggregates.
        """
        market_open = EST.localize(datetime(2024, 1, 2, 9, 30))
        mock_aggs = []
        current_time = market_open
        while current_time.hour < 16 or (
            current_time.hour == 16 and current_time.minute == 0
        ):
            mock_aggs.append(
                create_mock_agg(
                    int(current_time.timestamp() * 1000),
                    100 + np.random.normal(0, 0.5),
                    100 + np.random.normal(0, 0.5),
                    100 + np.random.normal(0, 0.5),
                    100 + np.random.normal(0, 0.5),
                    int(np.random.normal(1000, 100)),
                    100 + np.random.normal(0, 0.5),
                )
            )
            current_time += timedelta(minutes=5)

        mock_client.list_aggs.return_value = mock_aggs
        df = provider.get_intraday_bars(
            "AAPL", "2024-01-02", "2024-01-02", interval="5min"
        )
        if df is not None and not df.empty:
            assert "regular_session" in df.columns
            assert df.index.tz.zone == "UTC"

    def test_get_options_chain_mock(self, provider, mock_client):
        """
        Unit test for options chain retrieval with mocked data.
        """
        # Mock current price
        with patch.object(provider, "get_current_price") as mock_price:
            mock_price.return_value = MarketSnapshot(
                timestamp=datetime.now(UTC), price=150.0, bid=149.95, ask=150.05
            )
            mock_options = [
                create_mock_option(
                    strike=145.0,
                    expiry="2024-02-16",
                    contract_type="call",
                    iv=0.25,
                    greeks={"delta": 0.65, "gamma": 0.04, "theta": -0.15, "vega": 0.30},
                    quote={"bid": 6.80, "ask": 7.00},
                    volume=500,
                ),
                create_mock_option(
                    strike=155.0,
                    expiry="2024-02-16",
                    contract_type="put",
                    iv=0.28,
                    greeks={
                        "delta": -0.40,
                        "gamma": 0.03,
                        "theta": -0.12,
                        "vega": 0.25,
                    },
                    quote={"bid": 4.20, "ask": 4.40},
                    volume=300,
                ),
            ]
            mock_client.list_snapshot_options_chain.return_value = mock_options
            options = provider.get_options_chain("AAPL")

            if options:
                assert all(isinstance(opt, OptionData) for opt in options)
                for opt in options:
                    assert opt.contract_type in ["call", "put"]
                    assert opt.volume >= 0
                    if opt.bid is not None and opt.ask is not None:
                        assert opt.bid <= opt.ask

    def test_get_daily_bars_mock(self, provider, mock_client):
        """
        Unit test for daily bars using mocked data.
        """
        # Mock daily aggregates
        mock_aggs = [
            create_mock_agg(
                int(datetime(2024, 1, 1).timestamp() * 1000),
                150,
                155,
                149,
                153,
                1000000,
                152.5,
            ),
            create_mock_agg(
                int(datetime(2024, 1, 2).timestamp() * 1000),
                153,
                158,
                152,
                157,
                1200000,
                155.5,
            ),
        ]
        mock_client.list_aggs.return_value = mock_aggs
        df = provider.get_daily_bars("AAPL", "2024-01-01", "2024-01-02")
        assert df is not None
        assert not df.empty
        assert "realized_vol" in df.columns

    def test_get_historical_volatility_mock(self, provider):
        """
        Unit test for historical volatility calculations
        using mocked daily bar data.
        """
        mock_data = pd.DataFrame(
            {
                "open": [
                    100 * (1 + 0.01 * i + np.random.normal(0, 0.02)) for i in range(50)
                ],
                "close": [
                    100 * (1 + 0.01 * i + np.random.normal(0, 0.02)) for i in range(50)
                ],
                "high": [
                    105 * (1 + 0.01 * i + np.random.normal(0, 0.02)) for i in range(50)
                ],
                "low": [
                    95 * (1 + 0.01 * i + np.random.normal(0, 0.02)) for i in range(50)
                ],
                "volume": [
                    1000000 + np.random.randint(-100000, 100000) for _ in range(50)
                ],
            },
            index=pd.date_range("2024-01-01", periods=50, tz=UTC),
        )

        with patch.object(provider, "get_daily_bars") as mock_bars:
            mock_bars.return_value = mock_data
            result = provider.get_historical_volatility("AAPL", "2024-01-01")
            assert result is not None
            assert not result.empty
            for col in ["realized_vol", "parkinson_vol", "garman_klass_vol"]:
                assert col in result.columns


############################################################
#               LIVE INTEGRATION TESTS
############################################################


@pytest.fixture(scope="session")
def live_provider():
    """
    Provides a PolygonDataProvider that uses the real Polygon.io REST client.
    Requires the environment variable POLYGON_API_KEY to be set.
    """
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        pytest.skip("POLYGON_API_KEY not set. Skipping live integration tests.")
    return PolygonDataProvider(api_key=api_key)


def test_init(live_provider):
    live_provider.cache.clear()
    assert live_provider.max_retries == 3
    assert live_provider.retry_delay == 1.0
    assert live_provider.eastern_tz == EST
    assert live_provider.utc_tz == UTC


def test_get_current_price(live_provider):
    ticker = "AAPL"
    snapshot = live_provider.get_current_price(ticker)
    assert snapshot is not None, f"get_current_price returned None for {ticker}"
    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.price is not None
    assert snapshot.timestamp.tzinfo == UTC


def test_get_daily_bars(live_provider):
    ticker = "AAPL"
    df = live_provider.get_daily_bars(ticker, "2024-01-01", "2024-02-01")
    assert df is not None, "get_daily_bars returned None"
    assert not df.empty, "No daily data returned from Polygon"
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns, f"Missing column {col} in daily bars"


def test_get_options_expiration(live_provider):
    ticker = "AAPL"
    expirations = live_provider.get_options_expiration(ticker)
    assert expirations is not None, "get_options_expiration returned None"
    assert len(expirations) > 0, "No expirations returned"
    assert all(exp.tzinfo == UTC for exp in expirations)


def test_get_options_chain(live_provider):
    ticker = "AAPL"
    options = live_provider.get_options_chain(ticker)
    if options is not None and len(options) > 0:
        assert isinstance(options[0], OptionData)
        for opt in options:
            assert opt.contract_type in ["call", "put"]
            assert opt.implied_volatility > 0, "Implied volatility must be positive"


def test_get_intraday_bars(live_provider):
    ticker = "AAPL"
    df = live_provider.get_intraday_bars(
        ticker, "2024-01-03", "2024-01-03", interval="5min"
    )
    if df is not None and not df.empty:
        assert "open" in df.columns
        assert "close" in df.columns
        assert "regular_session" in df.columns


def test_market_hours(live_provider):
    result = live_provider.get_market_hours("2024-01-02")
    assert result is not None, "Should be a normal trading day"
    assert "market_open" in result
    assert "market_close" in result
    assert result["market_open"].tzinfo == UTC
    assert result["market_close"].tzinfo == UTC
