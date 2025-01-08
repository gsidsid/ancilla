# experiments/test_backtest.py
from datetime import datetime
import pytz
from typing import Dict, Any
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
