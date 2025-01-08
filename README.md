# Ancilla
A Polygon.io and FRED-based financial data library for backtesting trading strategies.

<p>
    <a href="https://pypi.python.org/pypi/ancilla" alt="PyPI">
        <img src="https://img.shields.io/pypi/v/ancilla.svg" /></a>
    <a href="https://github.com/gsidsid/ancilla" alt="License">
        <img src="https://img.shields.io/github/license/gsidsid/ancilla" /></a>
</p>

Ancilla is a straightforward Python library for querying/visualizing financial data and executing backtests to survey trading strategies. Under the hood, Ancilla uses Polygon.io to fetch historical stock and options data, and FRED, the official data source for the Federal Reserve, to fetch interest rates.

## Installation

```
pip install ancilla
```

Create a `.env` file in the root directory with
```
POLYGON_API_KEY=your_api_key
FRED_API_KEY=your_api_key
```

Using Ancilla requires paid Polygon.io data subscriptions, which can be obtained [here](https://polygon.io/), and a free FRED API key, which can be obtained [here](https://fred.stlouisfed.org/).

## Usage

```python
# experiments/test_backtest.py
from datetime import datetime
from typing import Dict, Any
import pytz
import os
import dotenv

from ancilla.backtesting.configuration import CommissionConfig, SlippageConfig
from ancilla.backtesting import Backtest, Strategy
from ancilla.providers import PolygonDataProvider

dotenv.load_dotenv()


class HoldSpyStrategy(Strategy):
    """Simple test strategy that buys and holds SPY."""

    def __init__(self, data_provider, position_size: float = 0.2):
        super().__init__(data_provider, name="hold_spy")
        self.position_size = position_size
        self.entry_prices = {}  # Track entry prices for each ticker

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Buy and hold stocks with basic position sizing."""
        self.logger.debug(f"Processing market data for {timestamp}")
        for ticker, data in market_data.items():
            # Log market data
            self.logger.debug(f"{ticker} price: ${data['close']:.2f}")

            # Skip if we already have a position
            if ticker in self.portfolio.positions:
                continue

            # Calculate position size based on portfolio value
            portfolio_value = self.portfolio.get_total_value()
            position_value = portfolio_value * self.position_size
            shares = int(position_value / data["close"])

            if shares > 0:
                # Open position
                self.logger.info(
                    f"Opening position in {ticker}: {shares} shares @ ${data['close']:.2f}"
                )
                success = self.engine.buy_stock(ticker=ticker, quantity=shares)
                if success:
                    self.entry_prices[ticker] = data["close"]
                    self.logger.info(f"Successfully opened position in {ticker}")
                else:
                    self.logger.warning(f"Failed to open position in {ticker}")


if __name__ == "__main__":
    # Initialize data provider
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy
    strategy = HoldSpyStrategy(
        data_provider=data_provider, position_size=0.5  # 50% of portfolio per position
    )

    # Set up test parameters
    tickers = ["SPY"]  # Reduced ticker list for testing
    start_date = datetime(2023, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 12, 31, tzinfo=pytz.UTC)  # Shorter test period

    # Initialize backtest engine
    simple_backtest = Backtest(
        strategy=strategy,
        initial_capital=100000,
        frequency="1hour",
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        commission_config=CommissionConfig(
            min_commission=1.0, per_share=0.005, per_contract=0.65, percentage=0.0001
        ),
        slippage_config=SlippageConfig(
            base_points=1.0, vol_impact=0.1, spread_factor=0.5, market_impact=0.05
        ),
    )

    # Run backtest
    results = simple_backtest.run()

    # Plot results
    results.plot(include_drawdown=True)
```

## Features

- A Polygon.io data wrapper with retries & automatic caching
- Basic visualizations for IV surfaces, liquidity, and price data
- Backtesting for equities and options
  - Batch requests and data caching for speed
  - Configurable strategy frequency (30min, 1hour, etc.)
  - Models slippage, commissions, price impact
  - Fill probability based on volume (or deterministic fills for testing)
  - Adjusts stock prices for splits and dividends, pays dividends
  - Monthly interest using accurate rates
  - Automatic ITM options exercise and assignment
  - Detailed trade history visualizations, performance metrics, logs
