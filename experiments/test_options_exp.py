from datetime import datetime
from typing import Dict, Optional, Any
import dotenv
import pytz
import os

from ancilla.models import Stock, Option
from ancilla.backtesting.configuration import CommissionConfig, SlippageConfig
from ancilla.backtesting import Backtest, Strategy
from ancilla.providers import PolygonDataProvider

dotenv.load_dotenv()

class OptionExpirationTestStrategy(Strategy):
    """
    Test strategy that focuses on option expiration behavior:
    1. Buys stock and immediately writes ATM calls
    2. Holds through expiration to test assignment
    3. Logs detailed position states around expiration
    4. Tests multiple expiration scenarios (ITM, ATM, OTM)
    """

    def __init__(
        self,
        data_provider,
        position_size: float = 0.3,
        test_scenarios: list = ['ITM', 'ATM', 'OTM'],
        trading_hours: tuple = (10, 15)
    ):
        super().__init__(data_provider, name="expiration_test")
        self.position_size = position_size
        self.test_scenarios = test_scenarios
        self.trading_hours = trading_hours
        self.stock_positions = {}
        self.active_calls = {}
        self.scenario_status = {}

    def _has_pending_expiration(self, timestamp: datetime) -> bool:
        """Check if we have any positions expiring today."""
        for position in self.portfolio.positions.values():
            if (isinstance(position.instrument, Option) and
                position.instrument.expiration.date() == timestamp.date()):
                return True
        return False

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process market data updates with focus on expiration behavior."""
        if not (self.trading_hours[0] <= timestamp.hour <= self.trading_hours[1]):
            return

        # Create a snapshot of market data
        market_data_snapshot = dict(market_data)

        # First check existing positions for expiration
        for ticker in list(self.active_calls.keys()):
            if ticker in market_data_snapshot:
                self._check_expiration(ticker, market_data_snapshot[ticker]['close'], timestamp)

        # Only establish new positions if:
        # 1. No pending expirations today
        # 2. Haven't tested all scenarios
        if (not self._has_pending_expiration(timestamp) and
            len(self.scenario_status) < len(self.test_scenarios)):

            # Then establish new test positions if needed
            for ticker, data in market_data_snapshot.items():
                if len(ticker) > 5:  # Skip option tickers
                    continue

                current_price = data['close']

                # Enter new position if we don't have one
                if ticker not in self.stock_positions:
                    self._establish_test_position(ticker, current_price, timestamp)

    def _establish_test_position(self, ticker: str, price: float, timestamp: datetime) -> None:
        """Establish a new test position with specific strike selection based on test scenario."""
        # Calculate position size
        portfolio_value = self.portfolio.get_total_value()
        position_value = portfolio_value * self.position_size
        shares = min(100, int(position_value / price))

        if shares < 100:
            return

        # Determine which scenario we're testing
        current_scenario = self.test_scenarios[len(self.scenario_status)]

        # Buy stock position
        self.logger.info(f"Buying {shares} shares of {ticker} @ ${price:.2f} for {current_scenario} test")
        stock = Stock(ticker)
        success = self.engine.buy_stock(
            ticker=ticker,
            quantity=shares
        )

        if not success:
            return

        self.stock_positions[ticker] = stock

        # Select strike based on test scenario
        strike_multiplier = {
            'ITM': 0.95,  # 5% in-the-money
            'ATM': 1.00,  # at-the-money
            'OTM': 1.05   # 5% out-of-the-money
        }

        target_strike = price * strike_multiplier[current_scenario]

        # Get available calls
        available_calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=timestamp,
            strike_range=(target_strike * 0.98, target_strike * 1.02),
            max_expiration_days=30,
            contract_type='call'
        )

        if not available_calls:
            return

        # Select call closest to target strike
        selected_call = min(available_calls, key=lambda x: abs(x.strike - target_strike))

        # Write call
        contracts = shares // 100
        success = self.engine.sell_option(
            option=selected_call,
            quantity=contracts
        )

        if success:
            self.active_calls[ticker] = selected_call
            self.scenario_status[ticker] = current_scenario
            self.logger.info(
                f"Sold {current_scenario} call for {ticker} @ strike {selected_call.strike} "
                f"expiring {selected_call.expiration.date()}"
            )

    def _check_expiration(self, ticker: str, current_price: float, timestamp: datetime) -> None:
        """Monitor and log position behavior around expiration."""
        call = self.active_calls[ticker]
        scenario = self.scenario_status[ticker]

        # Log detailed position state in the days leading up to expiration
        days_to_expiry = (call.expiration.replace(tzinfo=pytz.UTC) - timestamp).days

        if days_to_expiry <= 5:
            self.logger.info(f"\nExpiration check for {ticker} ({scenario} test):")
            self.logger.info(f"  Days to expiry: {days_to_expiry}")
            self.logger.info(f"  Current price: ${current_price:.2f}")
            self.logger.info(f"  Strike price: ${call.strike:.2f}")
            self.logger.info(f"  Moneyness: {(current_price - call.strike):.2f}")

            # Log all current positions
            self.logger.info("Current positions:")
            for pos_ticker, position in self.portfolio.positions.items():
                self.logger.info(
                    f"  {pos_ticker}: {type(position.instrument).__name__}, "
                    f"{position.quantity} units"
                )

        # After expiration, clean up and log final state
        if timestamp > call.expiration:
            self.logger.info(f"\nPost-expiration state for {ticker} ({scenario} test):")
            self.logger.info(f"  Final price: ${current_price:.2f}")
            self.logger.info(f"  Strike price: ${call.strike:.2f}")
            self.logger.info("Final positions:")
            for pos_ticker, position in self.portfolio.positions.items():
                self.logger.info(
                    f"  {pos_ticker}: {type(position.instrument).__name__}, "
                    f"{position.quantity} units"
                )

            # Clean up tracking
            self.active_calls.pop(ticker)
            self.stock_positions.pop(ticker, None)

def test_option_expiration():
    """Run backtest with the option expiration test strategy."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy instance
    strategy = OptionExpirationTestStrategy(
        data_provider=data_provider,
        position_size=0.3,
        test_scenarios=['ITM', 'ATM', 'OTM']
    )

    # Set up test parameters
    tickers = ["AAPL"]  # Use a liquid stock
    start_date = datetime(2023, 11, 1)
    end_date = datetime(2023, 12, 30)
    initial_capital = 100000

    # Initialize backtest engine
    option_expiry_backtest = Backtest(
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
    results = option_expiry_backtest.run()

    # Plot results
    results.plot(include_drawdown=True)

    return results

if __name__ == "__main__":
    results = test_option_expiration()
