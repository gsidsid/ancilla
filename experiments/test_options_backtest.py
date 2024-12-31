# experiments/test_covered_call.py
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List, Tuple
import os
import dotenv
from ancilla.backtesting.simulation import CommissionConfig, SlippageConfig
from ancilla.providers.polygon_data_provider import PolygonDataProvider
from ancilla.backtesting.engine import BacktestEngine
from ancilla.backtesting.strategy import Strategy
from ancilla.backtesting.instruments import Stock, Option

dotenv.load_dotenv()

class CoveredCallStrategy(Strategy):
    """
    Covered Call Strategy that:
    1. Takes long stock positions with specified portfolio allocation
    2. Writes out-of-the-money calls against stock positions
    3. Manages positions and rolls options near expiration
    """

    def __init__(
        self,
        data_provider,
        position_size: float = 0.2,
        otm_pct: float = 0.05,
        min_days_to_expiry: int = 20,
        max_days_to_expiry: int = 45,
        roll_dte_threshold: int = 5,
        strike_flex_pct: float = 0.02,
        trading_hours: Tuple[int, int] = (10, 15)  # Only trade between 10AM and 3PM
    ):
        super().__init__(data_provider, name="covered_call")
        self.position_size = position_size
        self.otm_pct = otm_pct
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.roll_dte_threshold = roll_dte_threshold
        self.strike_flex_pct = strike_flex_pct
        self.trading_hours = trading_hours

        # Track positions by instrument type
        self.stock_positions = {}  # ticker -> Stock instrument
        self.active_calls = {}     # ticker -> Option instrument

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Process hourly market data updates."""
        # Only trade during specified hours
        if not (self.trading_hours[0] <= timestamp.hour <= self.trading_hours[1]):
            return

        for ticker, data in market_data.items():
            current_price = data['close']

            # First check if we need to roll any existing calls
            if ticker in self.active_calls:
                self._manage_existing_call(ticker, current_price, timestamp, market_data)

            # If we don't have a stock position, consider entering one
            # Make sure it's not an options ticker
            if ticker not in self.stock_positions and len(ticker) <= 5:
                self._enter_stock_position(ticker, current_price, timestamp, market_data)

            # If we have stock but no call, write a new covered call
            elif ticker not in self.active_calls:
                self._write_new_call(ticker, current_price, timestamp, market_data)

    def _enter_stock_position(self, ticker: str, price: float, timestamp: datetime,
                            market_data: Dict[str, Any]) -> None:
        """Enter a new stock position."""
        portfolio_value = self.portfolio.get_total_value()
        position_value = portfolio_value * self.position_size

        # Let's limit to just 1 covered call for now
        shares = min(100, int(position_value / price))

        if shares >= 100:  # Only enter if we can write at least one call
            self.logger.info(f"Buying {shares} shares of {ticker} @ ${price:.2f}")
            stock = Stock(ticker)
            success = self.engine.buy_stock(
                ticker=ticker,
                quantity=shares,
                timestamp=timestamp,
                market_data=market_data
            )

            if success:
                self.stock_positions[ticker] = stock
                # Immediately try to write a call
                self._write_new_call(ticker, price, timestamp, market_data)

    def _write_new_call(self, ticker: str, current_price: float, timestamp: datetime,
                           market_data: Dict[str, Any]) -> None:
        """Write a new covered call against an existing stock position."""
        stock_position = None
        for pos in self.portfolio.positions.values():
            if isinstance(pos.instrument, Stock) and pos.instrument.ticker == ticker:
                stock_position = pos
                break

        if not stock_position:
            return

        shares = stock_position.quantity
        contracts = shares // 100

        if contracts == 0:
            self.logger.info(f"Not enough shares ({shares}) for a covered call")
            return

        target_strike = current_price * (1 + self.otm_pct)

        strike_range = (
            target_strike * (1 - self.strike_flex_pct),
            target_strike * (1 + self.strike_flex_pct)
        )

        available_calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=timestamp,
            strike_range=strike_range,
            max_expiration_days=self.max_days_to_expiry,
            contract_type='call'
        )

        if not available_calls:
            return

        valid_calls = [
            call for call in available_calls
            if (call.expiration.replace(tzinfo=pytz.UTC) - timestamp).days >= self.min_days_to_expiry
        ]

        if not valid_calls:
            return

        sorted_calls = sorted(
            valid_calls,
            key=lambda x: (
                x.expiration.replace(tzinfo=pytz.UTC),
                abs(x.strike - target_strike)
            )
        )

        selected_call = sorted_calls[0]

        success = self.engine.sell_option(
            option=selected_call,
            quantity=contracts,
            timestamp=timestamp,
            market_data=market_data
        )

        if success:
            self.active_calls[ticker] = selected_call
            self.logger.info(f"Sold call for {ticker} @ strike {selected_call.strike} expiring {selected_call.expiration.date()}")

    def _manage_existing_call(self, ticker: str, current_price: float,
                                timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Manage existing call position, rolling if necessary."""

        call = self.active_calls[ticker]

        # If the call is expired, handle expiration
        if timestamp > call.expiration:
            for pos_ticker, position in self.portfolio.positions.items():
                self.logger.info(f"  {pos_ticker}: {type(position.instrument).__name__}, {position.quantity} units")

            # On expiration, all we need to do is remove it from tracking
            self.active_calls.pop(ticker)

            # Can write new calls against our stock position
            if any(isinstance(pos.instrument, Stock) for pos in self.portfolio.positions.values()):
                self._write_new_call(ticker, current_price, timestamp, market_data)
            return

        dte = (call.expiration - timestamp).days

        # Check if we need to roll
        if dte <= self.roll_dte_threshold:
            self.logger.info(f"Rolling call with {dte} DTE remaining")

            # Look specifically for stock positions
            stock_position = None
            for pos in self.portfolio.positions.values():
                self.logger.info(f"Checking position type: {type(pos.instrument).__name__}, ticker: {pos.instrument.ticker}")
                if isinstance(pos.instrument, Stock) and pos.instrument.ticker == ticker:
                    stock_position = pos
                    break

            if not stock_position:
                self.logger.warning("No stock position found")
                return

            self.logger.info(f"Found stock position of {stock_position.quantity} shares")

            contracts = stock_position.quantity // 100
            if contracts == 0:
                self.logger.warning("Not enough shares for covered call")
                return

            # Buy back current call
            self.logger.info(f"Buying back {contracts} contract(s)")
            success = self.engine.buy_option(
                option=call,
                quantity=contracts,
                timestamp=timestamp,
                market_data=market_data
            )

            if success:
                self.logger.info(f"Bought back calls for {ticker}")
                self.active_calls.pop(ticker)
                # Write new call
                self._write_new_call(ticker, current_price, timestamp, market_data)
            else:
                self.logger.warning(f"Failed to buy back calls for {ticker}")

        else:
            self.logger.debug(f"No roll needed yet ({dte} DTE)")
            self.logger.debug("Current positions:")
            for pos_ticker, position in self.portfolio.positions.items():
                self.logger.debug(f"  {pos_ticker}: {type(position.instrument).__name__}, {position.quantity} units")

def test_covered_call_strategy():
    """Run backtest with the covered call strategy."""
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable not set")

    data_provider = PolygonDataProvider(api_key)

    # Create strategy instance with standard parameters
    strategy = CoveredCallStrategy(
        data_provider=data_provider,
        position_size=0.2,           # 20% of portfolio per position
        otm_pct=0.05,               # 5% OTM calls
        min_days_to_expiry=20,      # Don't write calls with less than 20 DTE
        max_days_to_expiry=45,      # Don't look further than 45 days out
        roll_dte_threshold=5,        # Roll calls with 5 or fewer days left
        strike_flex_pct=0.02        # Allow ±2% flexibility in strike selection
    )

    # Set up test parameters for Q4 2023
    tickers = ["AAPL"]  # Test with a liquid stock
    start_date = datetime(2023, 10, 1, tzinfo=pytz.UTC)
    end_date = datetime(2023, 12, 30, tzinfo=pytz.UTC)
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

    # Plot results
    fig = results.plot_equity_curve(include_drawdown=True)
    fig.show()

    return results


if __name__ == "__main__":
    results = test_covered_call_strategy()
