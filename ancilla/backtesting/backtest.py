from ancilla.providers.polygon_data_provider import PolygonDataProvider
from ancilla.models.option_data import OptionData
from typing import List

import bt
import pandas as pd

class Backtest:
    """
    A backtesting engine that wraps the bt library and integrates with DataProviders.
    """

    def __init__(self, data_provider_instance: PolygonDataProvider, starting_capital: float):
        print(f"Initializing BacktestEngine with starting capital: {starting_capital}")
        self.data_provider = data_provider_instance
        self.starting_capital = starting_capital

    def fetch_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV data for a list of tickers from the PolygonDataProvider.

        Args:
            tickers: List of stock symbols.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            A pandas DataFrame with the combined data for the tickers.
        """
        print(f"Fetching data for tickers: {tickers} from {start_date} to {end_date}")
        combined_data = []
        for ticker in tickers:
            print(f"Fetching data for ticker: {ticker}")
            bars = self.data_provider.get_daily_bars(ticker, start_date, end_date)
            if bars is not None:
                print(f"Data fetched for {ticker}: {len(bars)} records")
                bars["ticker"] = ticker
                combined_data.append(bars)
            else:
                print(f"No data available for ticker: {ticker}")

        if not combined_data:
            raise ValueError("No data available for the given tickers and date range.")

        return pd.concat(combined_data)

    def fetch_options_data(self, ticker: str, expiration_days: int = 90) -> List[OptionData]:
        """
        Fetch options chain data for a given ticker.

        Args:
            ticker: Stock symbol.
            expiration_days: Number of days ahead to include in the options chain.

        Returns:
            List of OptionData objects.

        Raises:
            ValueError: If no options data is available.
        """
        print(f"Fetching options data for ticker: {ticker} with expiration range: {expiration_days} days")
        options_data = self.data_provider.get_options_chain(ticker, expiration_range_days=expiration_days)
        if options_data is None:
            return []
        return options_data

    def run(self, strategy: bt.Strategy, data: pd.DataFrame):
        """
        Run the backtest on a strategy and dataset.

        Args:
            strategy: A bt.Strategy instance.
            data: The dataset to backtest against.

        Returns:
            bt.Result: The backtest results.
        """
        print(f"Running backtest for strategy: {strategy.name}")
        backtest = bt.Backtest(strategy, data)
        result = bt.run(backtest)
        print(f"Backtest completed for strategy: {strategy.name}")
        return result
