import unittest
from datetime import datetime
import pytz
import logging
from ancilla.backtesting.portfolio import Portfolio
from ancilla.backtesting.instruments import InstrumentType, Option, Stock
from ancilla.models import Trade, Position

class TestBacktestAlignment(unittest.TestCase):
    def setUp(self):
        # Set up logging
        self.logger = logging.getLogger('TestBacktestAlignment')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def test_option_roll_alignment(self):
        self.logger.info("=== Starting Option Roll Alignment Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("test_portfolio", initial_capital)

        # Create original option
        option1 = Option(
            ticker="AAPL",
            strike=180.0,
            expiration=datetime(2023, 11, 24, tzinfo=pytz.UTC)
        )

        # Create roll option
        option2 = Option(
            ticker="AAPL",
            strike=200.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC)
        )

        self.logger.info("\n=== Trade 1: Initial Option Short ===")
        # Open first option position
        portfolio.open_position(
            instrument=option1,
            quantity=-1,
            price=2.09,
            timestamp=datetime(2023, 11, 1, 15, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00
        )
        self.logger.info(f"Cash after first option open: ${portfolio.cash:,.2f}")
        self.logger.info(f"Position value: ${portfolio.get_position_value():,.2f}")

        # Close first option (buy back)
        portfolio.close_position(
            instrument=option1,
            price=11.46,  # Higher price - a loss
            timestamp=datetime(2023, 11, 22, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00
        )
        self.logger.info(f"Cash after first option close: ${portfolio.cash:,.2f}")

        # Open second option (roll)
        portfolio.open_position(
            instrument=option2,
            quantity=-1,
            price=0.45,
            timestamp=datetime(2023, 11, 22, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00
        )
        self.logger.info(f"Cash after roll option open: ${portfolio.cash:,.2f}")

        # Final calculations
        self.logger.info("\n=== Final Position State ===")
        current_capital = portfolio.cash + portfolio.get_position_value()
        self.logger.info(f"Current cash: ${portfolio.cash:,.2f}")
        self.logger.info(f"Position value: ${portfolio.get_position_value():,.2f}")
        self.logger.info(f"Current capital: ${current_capital:,.2f}")

        # Log trades
        self.logger.info("\n=== Trade History ===")
        for i, trade in enumerate(portfolio.trades, 1):
            self.logger.info(f"Trade {i}:")
            self.logger.info(f"Instrument: {trade.instrument.ticker}")
            self.logger.info(f"Entry price: ${trade.entry_price:,.2f}")
            self.logger.info(f"Exit price: ${trade.exit_price:,.2f}")
            self.logger.info(f"Quantity: {trade.quantity}")
            self.logger.info(f"PnL: ${trade.pnl:,.2f}")

    def test_backtest_alignment(self):
        self.logger.info("=== Starting Backtest Alignment Test ===")

        # Initialize portfolio
        initial_capital = 100000
        self.logger.info(f"Initializing portfolio with capital: ${initial_capital:,.2f}")
        portfolio = Portfolio("test_portfolio", initial_capital)

        # Create instruments
        option = Option(
            ticker="AAPL",
            strike=205.0,
            expiration=datetime(2024, 1, 12, tzinfo=pytz.UTC),
        )
        stock = Stock(ticker="AAPL")

        self.logger.info("\n=== Trade 1: Stock Transaction ===")
        # Log initial state
        self.logger.info(f"Initial cash before stock trade: ${portfolio.cash:,.2f}")

        # Open stock position
        stock_quantity = 100
        stock_price = 197.45
        stock_costs = 2.97
        expected_stock_cash_impact = -(stock_price * stock_quantity) - stock_costs
        self.logger.info(f"Opening stock position:")
        self.logger.info(f"Quantity: {stock_quantity}")
        self.logger.info(f"Price: ${stock_price:,.2f}")
        self.logger.info(f"Transaction costs: ${stock_costs:,.2f}")
        self.logger.info(f"Expected cash impact: ${expected_stock_cash_impact:,.2f}")

        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=stock_price,
            timestamp=datetime(2023, 12, 20, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=stock_costs
        )
        self.logger.info(f"Cash after stock open: ${portfolio.cash:,.2f}")
        self.logger.info(f"Stock position value: ${portfolio.get_position_value():,.2f}")

        # Close stock position
        self.logger.info("\nClosing stock position:")
        portfolio.close_position(
            instrument=stock,
            price=stock_price,
            timestamp=datetime(2023, 12, 31, 0, 0, tzinfo=pytz.UTC),
            transaction_costs=stock_costs
        )
        self.logger.info(f"Cash after stock close: ${portfolio.cash:,.2f}")

        self.logger.info("\n=== Trade 2: Option Transaction ===")
        # Log state before option trade
        self.logger.info(f"Cash before option trade: ${portfolio.cash:,.2f}")

        # Open option position
        option_quantity = -1
        option_price = 0.55
        option_costs = 1.00
        multiplier = option.get_multiplier()
        expected_option_cash_impact = (option_price * abs(option_quantity) * multiplier) - option_costs
        self.logger.info(f"Opening option position:")
        self.logger.info(f"Quantity: {option_quantity}")
        self.logger.info(f"Price: ${option_price:,.2f}")
        self.logger.info(f"Multiplier: {multiplier}")
        self.logger.info(f"Transaction costs: ${option_costs:,.2f}")
        self.logger.info(f"Expected cash impact: ${expected_option_cash_impact:,.2f}")

        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=option_price,
            timestamp=datetime(2023, 12, 20, 15, 26, tzinfo=pytz.UTC),
            transaction_costs=option_costs
        )
        self.logger.info(f"Cash after option open: ${portfolio.cash:,.2f}")
        self.logger.info(f"Option position value: ${portfolio.get_position_value():,.2f}")

        # Close option position
        self.logger.info("\nClosing option position:")
        portfolio.close_position(
            instrument=option,
            price=option_price,
            timestamp=datetime(2023, 12, 31, 0, 0, tzinfo=pytz.UTC),
            transaction_costs=option_costs
        )
        self.logger.info(f"Cash after option close: ${portfolio.cash:,.2f}")

        # Final calculations
        self.logger.info("\n=== Final Calculations ===")
        actual_final_capital = portfolio.cash + portfolio.get_position_value()
        self.logger.info(f"Final cash: ${portfolio.cash:,.2f}")
        self.logger.info(f"Final position value: ${portfolio.get_position_value():,.2f}")
        self.logger.info(f"Actual final capital: ${actual_final_capital:,.2f}")

        # Log all trades
        self.logger.info("\n=== Trade History ===")
        for i, trade in enumerate(portfolio.trades, 1):
            self.logger.info(f"Trade {i}:")
            self.logger.info(f"Instrument: {trade.instrument.ticker}")
            self.logger.info(f"Entry price: ${trade.entry_price:,.2f}")
            self.logger.info(f"Exit price: ${trade.exit_price:,.2f}")
            self.logger.info(f"Quantity: {trade.quantity}")
            self.logger.info(f"PnL: ${trade.pnl:,.2f}")

        # Log all cash flows
        self.logger.info("\n=== Cash Flow History ===")
        self.logger.info(f"Opening cash flows: {portfolio.opening_cash_flows}")
        self.logger.info(f"Total opening cash flows: ${sum(portfolio.opening_cash_flows):,.2f}")

        # Run assertions
        self.logger.info("\n=== Running Assertions ===")
        self.assertEqual(len(portfolio.positions), 0, "All positions should be closed at the end of the backtest.")
        self.assertAlmostEqual(actual_final_capital, 80302.06, places=2,
                             msg=f"Final capital mismatch: Expected 80,302.06, Got {actual_final_capital}")

        expected_pnls = [-2.97, -1.00]
        actual_pnls = [t.pnl for t in portfolio.trades]
        for i, (expected, actual) in enumerate(zip(expected_pnls, actual_pnls)):
            self.assertAlmostEqual(actual, expected, places=2,
                                 msg=f"Trade {i+1} PnL mismatch: Expected {expected}, Got {actual}")

        expected_opening_cash_flows = -19693.97
        actual_opening_cash_flows = sum(portfolio.opening_cash_flows)
        self.assertAlmostEqual(actual_opening_cash_flows, expected_opening_cash_flows, places=2,
                             msg=f"Opening cash flows mismatch: Expected {expected_opening_cash_flows}, Got {actual_opening_cash_flows}")

        expected_net_pnl = -19697.94
        actual_net_pnl = actual_final_capital - portfolio.initial_capital
        self.assertAlmostEqual(actual_net_pnl, expected_net_pnl, places=2,
                             msg=f"Net PnL mismatch: Expected {expected_net_pnl}, Got {actual_net_pnl}")
