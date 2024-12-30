# experiments/test_backtest.py
from datetime import datetime
import pytz
from typing import Dict, Any, List
import os
import dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ancilla.backtesting.simulation import CommissionConfig, SlippageConfig
from ancilla.providers.polygon_data_provider import PolygonDataProvider
from ancilla.backtesting.engine import BacktestEngine
from ancilla.backtesting.strategy import Strategy
from ancilla.models import OptionData

dotenv.load_dotenv()


class SimpleTestStrategy(Strategy):
    """Simple test strategy that buys and holds stocks."""

    def __init__(self, data_provider, position_size: float = 0.2):
        super().__init__(data_provider, name="simpletest")
        self.position_size = position_size
        self.entry_prices = {}  # Track entry prices for each ticker

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Buy and hold stocks with basic position sizing."""
        self.logger.debug(f"Processing market data for {timestamp}")
        for ticker, data in market_data.items():
            # Log market data
            self.logger.debug(f"{ticker} Data - Open: {data.get('open', data['close'])}, High: {data.get('high', data['close'])}, Low: {data.get('low', data['close'])}, Close: {data['close']:.2f}")

            # Skip if we already have a position
            if ticker in self.portfolio.positions:
                continue

            # Calculate position size based on portfolio value
            portfolio_value = self.portfolio.get_total_value()
            position_value = portfolio_value * self.position_size
            shares = int(position_value / data['close'])

            if shares > 0:
                # Open position
                self.logger.info(
                    f"Opening position in {ticker}: {shares} shares @ ${data['close']:.2f}"
                )
                success = self.engine.execute_order(
                    ticker=ticker,
                    quantity=shares,
                    price=data['close'],
                    timestamp=timestamp,
                    market_data=market_data,
                    position_type='stock'
                )
                if success:
                    self.entry_prices[ticker] = data['close']
                    self.logger.info(f"Successfully opened position in {ticker}")
                else:
                    self.logger.warning(f"Failed to open position in {ticker}")

    def on_option_data(self, timestamp: datetime, options_data: List[OptionData]) -> None:
        """Not using options in this test."""
        pass


class VolatilityBasedStrategy(Strategy):
    """Volatility-based strategy that adjusts position sizes based on ATR."""

    def __init__(self, data_provider, position_size: float = 0.2, atr_period: int = 14, atr_multiplier: float = 1.5):
        super().__init__(data_provider, name="volatility_based")
        self.base_position_size = position_size
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.price_history: Dict[str, pd.DataFrame] = {}  # Store price history for each ticker
        self.entry_prices = {}  # Track entry prices for each ticker

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Adjust positions based on volatility."""
        self.logger.debug(f"Processing market data for {timestamp}")
        for ticker, data in market_data.items():
            self.logger.debug(f"{ticker} Data - Open: {data.get('open', data['close'])}, High: {data.get('high', data['close'])}, Low: {data.get('low', data['close'])}, Close: {data['close']:.2f}")

            # Update price history
            if ticker not in self.price_history:
                # Initialize with correct data types to prevent FutureWarnings
                self.price_history[ticker] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])

            # Create a new DataFrame for the new row
            new_row = pd.DataFrame([{
                'timestamp': timestamp,
                'open': data.get('open', data['close']),
                'high': data.get('high', data['close']),
                'low': data.get('low', data['close']),
                'close': data['close']
            }])

            # Check if the existing DataFrame is empty
            if self.price_history[ticker].empty:
                self.price_history[ticker] = new_row
                self.logger.debug(f"Initialized price history for {ticker} with first data point.")
            else:
                # Concatenate the new row to the existing DataFrame
                self.price_history[ticker] = pd.concat([self.price_history[ticker], new_row], ignore_index=True)

            # Ensure we have enough data to calculate ATR
            if len(self.price_history[ticker]) < self.atr_period + 1:
                self.logger.debug(f"Not enough data to calculate ATR for {ticker}. Needed: {self.atr_period + 1}, Available: {len(self.price_history[ticker])}")
                continue  # Need at least atr_period + 1 data points for ATR calculation

            # Calculate ATR manually
            atr = self.calculate_atr(ticker)

            if atr is None or atr == 0:
                self.logger.debug(f"ATR is {atr} for {ticker} at {timestamp}, skipping trade logic.")
                continue  # Skip if ATR is not available or zero

            self.logger.debug(f"{ticker} ATR: {atr:.2f}")

            # Determine entry and exit thresholds
            entry_threshold = data['close'] + self.atr_multiplier * atr
            exit_threshold = data['close'] - self.atr_multiplier * atr

            self.logger.debug(f"{ticker} Entry Threshold: {entry_threshold:.2f}, Exit Threshold: {exit_threshold:.2f}")

            # Check if we have an open position
            if ticker in self.portfolio.positions:
                position = self.portfolio.positions[ticker]
                # Exit condition
                if data['close'] < exit_threshold:
                    self.logger.info(f"Exiting position in {ticker}: {position.quantity} shares @ ${data['close']:.2f}")
                    success = self.engine.execute_order(
                        ticker=ticker,
                        quantity=-position.quantity,  # Sell all shares
                        price=data['close'],
                        timestamp=timestamp,
                        market_data=market_data,
                        position_type='stock'
                    )
                    if success:
                        del self.entry_prices[ticker]
                        self.logger.info(f"Successfully exited position in {ticker}")
                    else:
                        self.logger.warning(f"Failed to exit position in {ticker}")
            else:
                # Entry condition
                if data['close'] > entry_threshold:
                    # Adjust position size inversely with ATR
                    portfolio_value = self.portfolio.get_total_value()
                    volatility_adjustment = self.base_position_size / atr
                    position_size = min(volatility_adjustment, self.base_position_size)  # Cap the position size
                    position_value = portfolio_value * position_size
                    shares = int(position_value / data['close'])

                    if shares > 0:
                        self.logger.info(
                            f"Opening position in {ticker}: {shares} shares @ ${data['close']:.2f} (ATR: {atr:.2f})"
                        )
                        success = self.engine.execute_order(
                            ticker=ticker,
                            quantity=shares,
                            price=data['close'],
                            timestamp=timestamp,
                            market_data=market_data,
                            position_type='stock'
                        )
                        if success:
                            self.entry_prices[ticker] = data['close']
                            self.logger.info(f"Successfully opened position in {ticker}")
                        else:
                            self.logger.warning(f"Failed to open position in {ticker}")

    def calculate_atr(self, ticker: str) -> float:
        """Calculate the Average True Range (ATR) for the given ticker."""
        df = self.price_history[ticker].tail(self.atr_period + 1).copy()
        df.reset_index(drop=True, inplace=True)

        if len(df) < self.atr_period + 1:
            raise ValueError(f"Not enough data to calculate ATR for {ticker}")

        # Calculate True Range (TR)
        tr = []
        for i in range(1, len(df)):
            current_high = df.at[i, 'high']
            current_low = df.at[i, 'low']
            previous_close = df.at[i-1, 'close']

            tr1 = current_high - current_low
            tr2 = abs(current_high - previous_close)
            tr3 = abs(current_low - previous_close)

            tr_value = max(tr1, tr2, tr3)
            tr.append(tr_value)

        # Remove NaN or infinite TR values
        tr = [value for value in tr if not pd.isna(value) and np.isfinite(value)]

        if len(tr) < self.atr_period:
            raise ValueError(f"Not enough valid TR values to calculate ATR for {ticker}")

        # Calculate ATR as the average of TR over the period
        atr = sum(tr[-self.atr_period:]) / self.atr_period
        return atr


def run_backtest(strategy_class):
    """Run a backtest using the specified strategy class."""

    # Initialize data provider
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy instance
    if strategy_class == SimpleTestStrategy:
        strategy = SimpleTestStrategy(
            data_provider=data_provider,
            position_size=0.2  # 20% of portfolio per position
        )
    elif strategy_class == VolatilityBasedStrategy:
        strategy = VolatilityBasedStrategy(
            data_provider=data_provider,
            position_size=0.2,       # Base position size (20% of portfolio)
            atr_period=14,            # ATR period
            atr_multiplier=1.5        # Multiplier for entry/exit thresholds
        )
    else:
        raise ValueError("Unsupported strategy class provided.")

    # Set up test parameters
    tickers = ["AAPL", "MSFT"]  # Reduced ticker list for testing
    start_date = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 1, 31, tzinfo=pytz.UTC)  # Shorter test period
    initial_capital = 100000

    # Initialize backtest engine
    engine = BacktestEngine(
        data_provider=data_provider,
        strategy=strategy,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        commission_config=CommissionConfig(
            min_commission=1.0,
            per_share=0.005,
            per_contract=0.65,
            percentage=0.0001
        ),
        slippage_config=SlippageConfig(
            base_points=1.0,
            vol_impact=0.1,
            spread_factor=0.5,
            market_impact=0.1
        )
    )

    # Run backtest
    results = engine.run()

    return results


def test_backtest():
    """Run backtests for both strategies to verify the engine works."""
    print("Running SimpleTestStrategy Backtest...")
    simple_results = run_backtest(SimpleTestStrategy)

    # plot simple_results['equity_curve']
    plt.plot(simple_results['equity_curve']['equity'])
    plt.title('SimpleTestStrategy Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('$')
    plt.show()

    print("\nRunning VolatilityBasedStrategy Backtest...")
    volatility_results = run_backtest(VolatilityBasedStrategy)



if __name__ == "__main__":
    test_backtest()
