from datetime import datetime
import pytz
from typing import Dict, Optional, Any
import os
import dotenv

from ancilla.models import Option
from ancilla.backtesting.configuration import CommissionConfig, SlippageConfig
from ancilla.backtesting import Backtest, Strategy
from ancilla.providers import PolygonDataProvider

dotenv.load_dotenv()

class MetaLongCallStrategy(Strategy):
    """
    Simple Long Call Option Strategy for META that:
    1. Buys ATM calls with 30-day duration
    2. Holds to observe theta decay
    3. Exits before expiration
    """

    def __init__(
        self,
        data_provider,
        position_size: float = 0.1,
        target_days_to_expiry: int = 30,
        exit_dte_threshold: int = 5,
        strike_flex_pct: float = 0.02
    ):
        super().__init__(data_provider, name="meta_long_call")
        self.position_size = position_size
        self.target_days_to_expiry = target_days_to_expiry
        self.exit_dte_threshold = exit_dte_threshold
        self.strike_flex_pct = strike_flex_pct
        self.active_option = None
        self.entry_price = None
        self.has_position = False

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process market data updates."""
        if 'META' not in market_data:
            return

        current_price = market_data['META']['close']

        # Manage existing position
        if self.has_position:
            self._manage_existing_position(current_price, timestamp)
            return

        # Enter new position if we don't have one
        if not self.has_position:
            self._enter_option_position(current_price, timestamp)

    def _enter_option_position(self, current_price: float, timestamp: datetime) -> None:
        """Enter a new ATM call option position."""
        portfolio_value = self.portfolio.get_total_value()
        position_value = portfolio_value * self.position_size

        # Look for ATM calls
        strike_range = (
            current_price * (1 - self.strike_flex_pct),
            current_price * (1 + self.strike_flex_pct)
        )

        available_calls = self.data_provider.get_options_contracts(
            ticker='META',
            as_of=timestamp,
            strike_range=strike_range,
            max_expiration_days=self.target_days_to_expiry + 5,
            contract_type='call'
        )

        if not available_calls:
            return

        # Filter for options close to target DTE
        valid_calls = [
            call for call in available_calls
            if abs((call.expiration.replace(tzinfo=pytz.UTC) - timestamp).days - self.target_days_to_expiry) <= 5
        ]

        if not valid_calls:
            return

        # Select the call closest to ATM
        selected_call: Optional[Option] = min(
            valid_calls,
            key=lambda x: abs(x.strike - current_price)
        )

        contracts = 1  # Start with 1 contract

        success = self.engine.buy_option(
            option=selected_call,
            quantity=contracts
        )

        if success:
            self.active_option = selected_call
            self.entry_price = current_price
            self.has_position = True
            self.logger.info(
                f"Bought {contracts} META calls @ strike {selected_call.strike} "
                f"expiring {selected_call.expiration.date()}"
            )

    def _manage_existing_position(self, current_price: float, timestamp: datetime) -> None:
        """Manage existing option position, exit if close to expiration."""
        if not self.active_option:
            return

        if timestamp > self.active_option.expiration:
            self.logger.info("Option expired")
            self.active_option = None
            self.has_position = False
            return

        dte = (self.active_option.expiration.replace(tzinfo=pytz.UTC) - timestamp).days

        if dte <= self.exit_dte_threshold:
            self.logger.info(f"Exiting position with {dte} DTE remaining")

            # Find position quantity
            position = None
            for pos in self.portfolio.positions.values():
                if pos.instrument == self.active_option:
                    position = pos
                    break

            if position:
                success = self.engine.sell_option(
                    option=self.active_option,
                    quantity=position.quantity
                )

                if success:
                    self.logger.info("Sold META calls")
                    self.active_option = None
                    self.has_position = False
                else:
                    self.logger.warning("Failed to sell META calls")

def run_meta_backtest():
    """Run backtest with the META long call strategy."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy instance
    strategy = MetaLongCallStrategy(
        data_provider=data_provider,
        position_size=0.1,            # 10% of portfolio per position
        target_days_to_expiry=30,     # Target 30 DTE options
        exit_dte_threshold=-1,         # Exit with 5 or fewer days left
        strike_flex_pct=0.02          # Allow ±2% flexibility in strike selection
    )

    # Set up test parameters
    start_date = datetime(2023, 12, 1)
    end_date = datetime(2024, 4, 1)
    initial_capital = 100000

    # Initialize backtest engine
    engine = Backtest(
        data_provider=data_provider,
        strategy=strategy,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        tickers=['META'],
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

    # Plot results
    results.plot(include_drawdown=True)

    return results

if __name__ == "__main__":
    results = run_meta_backtest()
