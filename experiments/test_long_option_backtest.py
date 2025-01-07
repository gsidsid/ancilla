from datetime import datetime
import pytz
from typing import Dict, Any
import os
import dotenv

from ancilla.backtesting.configuration import CommissionConfig, SlippageConfig
from ancilla.backtesting import Backtest, Strategy
from ancilla.providers import PolygonDataProvider

dotenv.load_dotenv()


class LongOptionStrategy(Strategy):
    """
    Simple Long Option Strategy that:
    1. Buys ATM calls with specified duration
    2. Holds them to observe theta decay
    3. Exits before expiration
    """

    def __init__(
        self,
        data_provider,
        position_size: float = 0.1,  # Smaller position size since options are leveraged
        target_days_to_expiry: int = 30,
        exit_dte_threshold: int = 5,
        strike_flex_pct: float = 0.02,
        trading_hours: tuple[int, int] = (10, 15),
    ):
        super().__init__(data_provider, name="long_option")
        self.position_size = position_size
        self.target_days_to_expiry = target_days_to_expiry
        self.exit_dte_threshold = exit_dte_threshold
        self.strike_flex_pct = strike_flex_pct
        self.trading_hours = trading_hours
        self.active_options = {}

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process hourly market data updates."""
        if not (self.trading_hours[0] <= timestamp.hour <= self.trading_hours[1]):
            return

        market_data_snapshot = dict(market_data)

        # First manage existing positions
        for ticker in list(self.active_options.keys()):
            if ticker in market_data_snapshot:
                current_price = market_data_snapshot[ticker]["close"]
                self._manage_existing_position(ticker, current_price, timestamp)

        # Then look for new positions
        for ticker, data in market_data_snapshot.items():
            # Skip if it's an options ticker or we already have a position
            if len(ticker) > 5 or ticker in self.active_options:
                continue

            current_price = data["close"]
            self._enter_option_position(ticker, current_price, timestamp)

    def _enter_option_position(
        self, ticker: str, current_price: float, timestamp: datetime
    ) -> None:
        """Enter a new ATM call option position."""
        portfolio_value = self.portfolio.get_total_value()
        position_value = portfolio_value * self.position_size

        # Look for ATM calls
        strike_range = (
            current_price * (1 - self.strike_flex_pct),
            current_price * (1 + self.strike_flex_pct),
        )

        available_calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=timestamp,
            strike_range=strike_range,
            max_expiration_days=self.target_days_to_expiry + 5,
            contract_type="call",
        )

        if not available_calls:
            return

        # Filter for options close to target DTE
        valid_calls = [
            call
            for call in available_calls
            if abs(
                (call.expiration.replace(tzinfo=pytz.UTC) - timestamp).days
                - self.target_days_to_expiry
            )
            <= 5
        ]

        if not valid_calls:
            return

        # Select the call closest to ATM
        selected_call = min(valid_calls, key=lambda x: abs(x.strike - current_price))

        contracts = 1

        success = self.engine.buy_option(option=selected_call, quantity=contracts)

        if success:
            self.active_options[ticker] = selected_call
            self.logger.info(
                f"Bought {contracts} {ticker} calls @ strike {selected_call.strike} "
                f"expiring {selected_call.expiration.date()}"
            )

    def _manage_existing_position(
        self, ticker: str, current_price: float, timestamp: datetime
    ) -> None:
        """Manage existing option position, exit if close to expiration."""
        option = self.active_options[ticker]

        if timestamp > option.expiration:
            self.logger.info(f"Option expired for {ticker}")
            self.active_options.pop(ticker)
            return

        dte = (option.expiration.replace(tzinfo=pytz.UTC) - timestamp).days

        if dte <= self.exit_dte_threshold:
            self.logger.info(f"Exiting position with {dte} DTE remaining")

            # Find position quantity
            position = None
            for pos in self.portfolio.positions.values():
                if pos.instrument == option:
                    position = pos
                    break

            if position:
                success = self.engine.sell_option(
                    option=option, quantity=position.quantity
                )

                if success:
                    self.logger.info(f"Sold calls for {ticker}")
                    self.active_options.pop(ticker)
                else:
                    self.logger.warning(f"Failed to sell calls for {ticker}")


def test_long_option_strategy():
    """Run backtest with the long option strategy."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy instance
    strategy = LongOptionStrategy(
        data_provider=data_provider,
        position_size=0.1,  # 10% of portfolio per position
        target_days_to_expiry=30,  # Target 30 DTE options
        exit_dte_threshold=5,  # Exit with 5 or fewer days left
        strike_flex_pct=0.02,  # Allow ±2% flexibility in strike selection
    )

    # Set up test parameters
    tickers = ["AAPL"]  # Test with a liquid stock
    start_date = datetime(2023, 11, 1)
    end_date = datetime(2023, 12, 30)
    initial_capital = 100000

    # Initialize backtest engine
    long_option_backtest = Backtest(
        strategy=strategy,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        commission_config=CommissionConfig(
            min_commission=1.0, per_share=0.005, per_contract=0.65, percentage=0.0001
        ),
        slippage_config=SlippageConfig(
            base_points=1.0, vol_impact=0.1, spread_factor=0.5, market_impact=0.1
        ),
    )

    # Run backtest
    results = long_option_backtest.run()

    # Plot results
    results.plot(include_drawdown=True)


if __name__ == "__main__":
    results = test_long_option_strategy()
