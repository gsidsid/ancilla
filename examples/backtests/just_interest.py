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


class JustInterestStrategy(Strategy):
    """Simple test strategy that buys and holds stocks."""

    def __init__(
        self,
        data_provider,
    ):
        super().__init__(data_provider, name="just_interest")

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """No strategy here."""
        pass


if __name__ == "__main__":
    # Initialize data provider
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy
    strategy = JustInterestStrategy(data_provider=data_provider)
    start_date = datetime(2020, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 8, 31, tzinfo=pytz.UTC)  # Shorter test period

    # Initialize backtest engine
    simple_backtest = Backtest(
        strategy=strategy,
        initial_capital=100000,
        frequency="1hour",
        start_date=start_date,
        end_date=end_date,
        tickers=[],
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
