from datetime import datetime
import unittest
import pytz
import sys
import logging

from ancilla.backtesting.portfolio import Portfolio
from ancilla.models import Option, Stock, InstrumentType


class TestBacktesting(unittest.TestCase):
    """
    Test suite for portfolio functionality during backtesting.
    """

    def setUp(self):
        # Set up logging with method name
        self.logger = logging.getLogger(self._testMethodName)
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Add formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Only add StreamHandler if not running under pytest
        if "pytest" not in sys.modules:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    ############################################################
    #               STOCKS – LONG AND SHORT
    ############################################################

    def test_profitable_stock_trade(self):
        self.logger.info("=== Starting Profitable Stock Trade Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_stock_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="GOOGL")

        # Open stock position
        stock_quantity = 50
        buy_price = 1500.00
        buy_costs = 10.00
        expected_buy_cash_impact = -(buy_price * stock_quantity) - buy_costs

        self.logger.info("\n=== Trade 1: Buy GOOGL ===")
        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying stock: ${portfolio.cash:,.2f}")

        # Close stock position at a higher price
        sell_price = 1600.00
        sell_costs = 10.00
        expected_sell_cash_impact = (sell_price * stock_quantity) - sell_costs

        self.logger.info("\n=== Trade 2: Sell GOOGL ===")
        portfolio.close_position(
            instrument=stock,
            price=sell_price,
            timestamp=datetime(2023, 12, 15, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling stock: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        expected_final_capital = (
            initial_capital
            + ((sell_price - buy_price) * stock_quantity)
            - (buy_costs + sell_costs)
        )
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        expected_pnl = (sell_price - buy_price) * stock_quantity - (
            buy_costs + sell_costs
        )
        actual_pnl = portfolio.trades[0].pnl
        self.assertAlmostEqual(
            actual_pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {actual_pnl}",
        )

    def test_losing_stock_trade(self):
        self.logger.info("=== Starting Losing Stock Trade Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("losing_stock_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="MSFT")

        # Open stock position
        stock_quantity = 100
        buy_price = 250.00
        buy_costs = 15.00
        expected_buy_cash_impact = -(buy_price * stock_quantity) - buy_costs

        self.logger.info("\n=== Trade 1: Buy MSFT ===")
        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 1, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying stock: ${portfolio.cash:,.2f}")

        # Close stock position at a lower price
        sell_price = 240.00
        sell_costs = 15.00
        expected_sell_cash_impact = (sell_price * stock_quantity) - sell_costs

        self.logger.info("\n=== Trade 2: Sell MSFT ===")
        portfolio.close_position(
            instrument=stock,
            price=sell_price,
            timestamp=datetime(2023, 12, 20, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling stock: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        expected_final_capital = (
            initial_capital
            + ((sell_price - buy_price) * stock_quantity)
            - (buy_costs + sell_costs)
        )
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        expected_pnl = (sell_price - buy_price) * stock_quantity - (
            buy_costs + sell_costs
        )
        actual_pnl = portfolio.trades[0].pnl
        self.assertAlmostEqual(
            actual_pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {actual_pnl}",
        )

    def test_profitable_short_stock_trade(self):
        self.logger.info("=== Starting Profitable Short Stock Trade Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_short_stock_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="TSLA")

        # Open short stock position
        stock_quantity = -50  # Negative for short
        sell_price = 800.00
        sell_costs = 15.00

        self.logger.info("\n=== Trade 1: Short Sell TSLA ===")
        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after short selling stock: ${portfolio.cash:,.2f}")

        # Close short stock position by buying back at lower price for profit
        buy_back_price = 750.00
        buy_back_costs = 15.00

        self.logger.info("\n=== Trade 2: Buy Back TSLA Short Position ===")
        portfolio.close_position(
            instrument=stock,
            price=buy_back_price,
            timestamp=datetime(2023, 12, 15, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_back_costs,
        )
        self.logger.info(f"Cash after buying back stock: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = ((sell_price - buy_back_price) * abs(stock_quantity)) - (
            sell_costs + buy_back_costs
        )
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_losing_short_stock_trade(self):
        self.logger.info("=== Starting Losing Short Stock Trade Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("losing_short_stock_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="TSLA")

        # Open short stock position
        stock_quantity = -30  # Negative for short
        sell_price = 800.00
        sell_costs = 10.00

        self.logger.info("\n=== Trade 1: Short Sell TSLA ===")
        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 1, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after short selling stock: ${portfolio.cash:,.2f}")

        # Close short stock position by buying back at higher price for loss
        buy_back_price = 850.00
        buy_back_costs = 10.00

        self.logger.info("\n=== Trade 2: Buy Back TSLA Short Position ===")
        portfolio.close_position(
            instrument=stock,
            price=buy_back_price,
            timestamp=datetime(2023, 12, 20, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_back_costs,
        )
        self.logger.info(f"Cash after buying back stock: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = ((sell_price - buy_back_price) * abs(stock_quantity)) - (
            sell_costs + buy_back_costs
        )
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    ############################################################
    #               OPTIONS - LONG AND SHORT
    ############################################################

    def test_option_roll_alignment(self):
        self.logger.info("=== Starting Option Roll Alignment Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("test_portfolio", initial_capital)

        # Create original option
        option1 = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=180.0,
            expiration=datetime(2023, 11, 24, tzinfo=pytz.UTC),
        )

        # Create roll option
        option2 = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=200.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC),
        )

        self.logger.info("\n=== Trade 1: Initial Option Short ===")
        # Open first option position
        portfolio.open_position(
            instrument=option1,
            quantity=-1,
            price=2.09,
            timestamp=datetime(2023, 11, 1, 15, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00,
        )
        self.logger.info(f"Cash after first option open: ${portfolio.cash:,.2f}")
        self.logger.info(f"Position value: ${portfolio.get_position_value():,.2f}")

        # Close first option (buy back)
        portfolio.close_position(
            instrument=option1,
            price=11.46,  # Higher price - a loss
            timestamp=datetime(2023, 11, 22, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00,
        )
        self.logger.info(f"Cash after first option close: ${portfolio.cash:,.2f}")

        # Open second option (roll)
        portfolio.open_position(
            instrument=option2,
            quantity=-1,
            price=0.45,
            timestamp=datetime(2023, 11, 22, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=1.00,
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

    def test_profitable_option_trade_long(self):
        self.logger.info("=== Starting Profitable Option Trade (Long) Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_option_long_portfolio", initial_capital)

        # Create option instrument
        option = Option(
            ticker="TSLA",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=800.0,
            expiration=datetime(2024, 1, 15, tzinfo=pytz.UTC),
        )

        # Open option position (buy)
        option_quantity = 2
        buy_price = 20.00
        buy_costs = 2.00
        multiplier = option.get_multiplier()
        expected_buy_cash_impact = (
            -(buy_price * option_quantity * multiplier) - buy_costs
        )

        self.logger.info("\n=== Trade 1: Buy TSLA Call Option ===")
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 5, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying option: ${portfolio.cash:,.2f}")

        # Close option position at a higher price
        sell_price = 25.00
        sell_costs = 2.00
        expected_sell_cash_impact = (
            sell_price * option_quantity * multiplier
        ) - sell_costs

        self.logger.info("\n=== Trade 2: Sell TSLA Call Option ===")
        portfolio.close_position(
            instrument=option,
            price=sell_price,
            timestamp=datetime(2023, 12, 25, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling option: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        expected_final_capital = (
            initial_capital
            + ((sell_price - buy_price) * option_quantity * multiplier)
            - (buy_costs + sell_costs)
        )
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        expected_pnl = ((sell_price - buy_price) * option_quantity * multiplier) - (
            buy_costs + sell_costs
        )
        actual_pnl = portfolio.trades[0].pnl
        self.assertAlmostEqual(
            actual_pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {actual_pnl}",
        )

    def test_losing_option_trade_short(self):
        self.logger.info("=== Starting Losing Option Trade (Short) Test ===")

        initial_capital = 100000
        portfolio = Portfolio("losing_option_short_portfolio", initial_capital)

        # Create naked short call option
        option = Option(
            ticker="NFLX",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=600.0,
            expiration=datetime(2024, 2, 15, tzinfo=pytz.UTC),
            naked=True,  # Explicitly mark as naked call
        )

        # Trade parameters (same as before)
        option_quantity = -3
        sell_price = 30.00
        sell_costs = 3.00
        buy_back_price = 35.00
        buy_back_costs = 3.00
        multiplier = option.get_multiplier()

        # Execute trades (same as before)
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 10, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )

        portfolio.close_position(
            instrument=option,
            price=buy_back_price,
            timestamp=datetime(2023, 12, 30, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_back_costs,
        )

        # Final calculations with full transaction costs
        final_capital = portfolio.cash + portfolio.get_position_value()

        # Calculate expected capital including all costs
        trade_pnl = (
            (sell_price - buy_back_price) * abs(option_quantity) * multiplier
        )  # Base PnL
        total_costs = sell_costs + buy_back_costs  # Include both transaction costs
        expected_final_capital = initial_capital + trade_pnl - total_costs

        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

    def test_short_option_pnl_correctness(self):
        """
        Comprehensive test for short option PnL correctness in both profit and loss scenarios.
        """
        self.logger.info(
            "=== Starting Comprehensive Short Option PnL Correctness Test ==="
        )

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("comprehensive_short_option_portfolio", initial_capital)

        # Create two naked short call options
        option_loss = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=150.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC),
            naked=True,  # Mark as naked
        )

        option_profit = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=155.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC),
            naked=True,  # Mark as naked
        )

        # Open short option position that will incur a loss: Sell 1 call at $5.00
        option_loss_quantity = -1
        option_loss_sell_price = 5.00
        option_loss_sell_costs = 1.00
        portfolio.open_position(
            instrument=option_loss,
            quantity=option_loss_quantity,
            price=option_loss_sell_price,
            timestamp=datetime(2023, 11, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=option_loss_sell_costs,
        )
        self.logger.info(
            f"Cash after selling loss-incurring option: ${portfolio.cash:,.2f}"
        )

        # Open short option position that will incur a profit: Sell 1 call at $6.00
        option_profit_quantity = -1
        option_profit_sell_price = 6.00
        option_profit_sell_costs = 1.00
        portfolio.open_position(
            instrument=option_profit,
            quantity=option_profit_quantity,
            price=option_profit_sell_price,
            timestamp=datetime(2023, 11, 1, 10, 5, tzinfo=pytz.UTC),
            transaction_costs=option_profit_sell_costs,
        )
        self.logger.info(
            f"Cash after selling profit-incurring option: ${portfolio.cash:,.2f}"
        )

        # Close loss-incurring short option at $7.00 premium (loss)
        option_loss_buy_back_price = 7.00
        option_loss_buy_back_costs = 1.00
        portfolio.close_position(
            instrument=option_loss,
            price=option_loss_buy_back_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=option_loss_buy_back_costs,
        )
        self.logger.info(
            f"Cash after buying back loss-incurring option: ${portfolio.cash:,.2f}"
        )

        # Close profit-incurring short option at $5.50 premium (profit)
        option_profit_buy_back_price = 5.50
        option_profit_buy_back_costs = 1.00
        portfolio.close_position(
            instrument=option_profit,
            price=option_profit_buy_back_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=option_profit_buy_back_costs,
        )
        self.logger.info(
            f"Cash after buying back profit-incurring option: ${portfolio.cash:,.2f}"
        )

        # Calculate expected PnLs
        # Option Loss:
        # PnL = (5.00 - 7.00) * 1 * 100 = -200.00
        # Transaction Costs = 1.00 + 1.00 = 2.00
        # Realized PnL = -200.00 - 2.00 = -202.00

        # Option Profit:
        # PnL = (6.00 - 5.50) * 1 * 100 = 50.00
        # Transaction Costs = 1.00 + 1.00 = 2.00
        # Realized PnL = 50.00 - 2.00 = 48.00

        expected_pnl_loss = -202.00
        expected_pnl_profit = 48.00

        total_expected_pnl = expected_pnl_loss + expected_pnl_profit
        expected_final_capital = initial_capital + total_expected_pnl

        # Final capital
        final_capital = portfolio.cash + portfolio.get_position_value()

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify individual trade PnLs
        self.assertEqual(
            len(portfolio.trades), 2, "There should be exactly two trades recorded."
        )

        trade_loss = portfolio.trades[0]
        trade_profit = portfolio.trades[1]

        self.assertAlmostEqual(
            trade_loss.pnl,
            expected_pnl_loss,
            places=2,
            msg=f"Loss Trade PnL mismatch: Expected {expected_pnl_loss}, Got {trade_loss.pnl}",
        )

        self.assertAlmostEqual(
            trade_profit.pnl,
            expected_pnl_profit,
            places=2,
            msg=f"Profit Trade PnL mismatch: Expected {expected_pnl_profit}, Got {trade_profit.pnl}",
        )

    def test_short_option_pnl_loss(self):
        """
        Test that closing a short option position at a higher price results in a negative PnL.
        """
        self.logger.info("=== Starting Losing Short Option PnL Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("losing_short_option_portfolio", initial_capital)

        # Create a naked short call option
        option = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=150.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC),
            naked=True,  # Mark as naked
        )

        # Open short option position: Sell 1 call option at $5.00 premium
        option_quantity = -1  # Negative for short
        sell_price = 5.00
        sell_costs = 1.00  # Transaction costs for selling
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 11, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling option: ${portfolio.cash:,.2f}")

        # Close short option position: Buy back at $7.00 premium (loss)
        buy_back_price = 7.00
        buy_back_costs = 1.00  # Transaction costs for buying back
        portfolio.close_position(
            instrument=option,
            price=buy_back_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_back_costs,
        )
        self.logger.info(f"Cash after buying back option: ${portfolio.cash:,.2f}")

        # Calculate expected PnL:
        # PnL per contract = (Sell Price - Buy Price) * Multiplier
        # Total PnL = (5.00 - 7.00) * 1 * 100 = -200.00
        # Total Transaction Costs = 1.00 + 1.00 = 2.00
        # Realized PnL = -200.00 - 2.00 = -202.00
        expected_pnl = -202.00

        # Final capital
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_short_option_pnl_profit(self):
        """
        Test that closing a short option position at a lower price results in a positive PnL.
        """
        self.logger.info("=== Starting Profitable Short Option PnL Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_short_option_portfolio", initial_capital)

        # Create a naked short call option
        option = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=150.0,
            expiration=datetime(2023, 12, 15, tzinfo=pytz.UTC),
            naked=True,  # Mark as naked
        )

        # Open short option position: Sell 2 call options at $4.50 premium
        option_quantity = -2  # Negative for short
        sell_price = 4.50
        sell_costs = 1.50  # Transaction costs for selling
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 11, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling options: ${portfolio.cash:,.2f}")

        # Close short option position: Buy back at $3.00 premium (profit)
        buy_back_price = 3.00
        buy_back_costs = 1.50  # Transaction costs for buying back
        portfolio.close_position(
            instrument=option,
            price=buy_back_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_back_costs,
        )
        self.logger.info(f"Cash after buying back options: ${portfolio.cash:,.2f}")

        # Calculate expected PnL:
        # PnL per contract = (Sell Price - Buy Price) * Multiplier
        # Total PnL = (4.50 - 3.00) * 2 * 100 = 300.00
        # Total Transaction Costs = 1.50 + 1.50 = 3.00
        # Realized PnL = 300.00 - 3.00 = 297.00
        expected_pnl = 297.00

        # Final capital
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_profitable_put_option_trade_long(self):
        self.logger.info("=== Starting Profitable Put Option Trade (Long) Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_put_long_portfolio", initial_capital)

        # Create put option instrument
        put_option = Option(
            ticker="IBM",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=130.0,
            expiration=datetime(2024, 1, 20, tzinfo=pytz.UTC),
        )

        # Open put option position (buy)
        option_quantity = 3
        buy_price = 5.00
        buy_costs = 1.50
        multiplier = put_option.get_multiplier()

        self.logger.info("\n=== Trade 1: Buy IBM Put Option ===")
        portfolio.open_position(
            instrument=put_option,
            quantity=option_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 1, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying put option: ${portfolio.cash:,.2f}")

        # Close put option position (sell) at higher premium for profit
        sell_price = 7.00
        sell_costs = 1.50

        self.logger.info("\n=== Trade 2: Sell IBM Put Option ===")
        portfolio.close_position(
            instrument=put_option,
            price=sell_price,
            timestamp=datetime(2023, 12, 15, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling put option: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = ((sell_price - buy_price) * option_quantity * multiplier) - (
            buy_costs + sell_costs
        )
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_losing_put_option_trade_long(self):
        self.logger.info("=== Starting Losing Put Option Trade (Long) Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("losing_put_long_portfolio", initial_capital)

        # Create put option instrument
        put_option = Option(
            ticker="IBM",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=130.0,
            expiration=datetime(2024, 1, 20, tzinfo=pytz.UTC),
        )

        # Open put option position (buy)
        option_quantity = 2
        buy_price = 6.00
        buy_costs = 1.00
        multiplier = put_option.get_multiplier()

        self.logger.info("\n=== Trade 1: Buy IBM Put Option ===")
        portfolio.open_position(
            instrument=put_option,
            quantity=option_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 1, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying put option: ${portfolio.cash:,.2f}")

        # Close put option position (sell) at lower premium for loss
        sell_price = 4.00
        sell_costs = 1.00

        self.logger.info("\n=== Trade 2: Sell IBM Put Option ===")
        portfolio.close_position(
            instrument=put_option,
            price=sell_price,
            timestamp=datetime(2023, 12, 16, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling put option: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = ((sell_price - buy_price) * option_quantity * multiplier) - (
            buy_costs + sell_costs
        )
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_profitable_put_option_trade_short(self):
        self.logger.info("=== Starting Profitable Put Option Trade (Short) Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("profitable_put_short_portfolio", initial_capital)

        # Create put option instrument
        put_option = Option(
            ticker="IBM",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=125.0,
            expiration=datetime(2024, 2, 20, tzinfo=pytz.UTC),
        )

        # Open put option position (sell)
        option_quantity = -2  # Negative for short
        sell_price = 7.00
        sell_costs = 1.50
        multiplier = put_option.get_multiplier()

        self.logger.info("\n=== Trade 1: Sell IBM Put Option ===")
        portfolio.open_position(
            instrument=put_option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 5, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling put option: ${portfolio.cash:,.2f}")

        # Close put option position (buy) at lower premium for profit
        buy_price = 5.00
        buy_costs = 1.50

        self.logger.info("\n=== Trade 2: Buy IBM Put Option ===")
        portfolio.close_position(
            instrument=put_option,
            price=buy_price,
            timestamp=datetime(2023, 12, 20, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying put option: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = (
            (sell_price - buy_price) * abs(option_quantity) * multiplier
        ) - (sell_costs + buy_costs)
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    def test_losing_put_option_trade_short(self):
        self.logger.info("=== Starting Losing Put Option Trade (Short) Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("losing_put_short_portfolio", initial_capital)

        # Create put option instrument
        put_option = Option(
            ticker="IBM",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=135.0,
            expiration=datetime(2024, 3, 20, tzinfo=pytz.UTC),
        )

        # Open put option position (sell)
        option_quantity = -1  # Negative for short
        sell_price = 4.50
        sell_costs = 1.00
        multiplier = put_option.get_multiplier()

        self.logger.info("\n=== Trade 1: Sell IBM Put Option ===")
        portfolio.open_position(
            instrument=put_option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 10, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(f"Cash after selling put option: ${portfolio.cash:,.2f}")

        # Close put option position (buy) at higher premium for loss
        buy_price = 6.00
        buy_costs = 1.00

        self.logger.info("\n=== Trade 2: Buy IBM Put Option ===")
        portfolio.close_position(
            instrument=put_option,
            price=buy_price,
            timestamp=datetime(2023, 12, 25, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(f"Cash after buying put option: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        expected_pnl = (
            (sell_price - buy_price) * abs(option_quantity) * multiplier
        ) - (sell_costs + buy_costs)
        expected_final_capital = initial_capital + expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify PnL
        self.assertEqual(
            len(portfolio.trades), 1, "There should be exactly one trade recorded."
        )
        trade = portfolio.trades[0]
        self.assertAlmostEqual(
            trade.pnl,
            expected_pnl,
            places=2,
            msg=f"Trade PnL mismatch: Expected {expected_pnl}, Got {trade.pnl}",
        )

    # Assignment is managed by the broker engine,
    # difficult to test here
    def test_covered_call_no_assignment(self):
        self.logger.info("=== Starting Backtest Alignment Test ===")

        # Initialize portfolio
        initial_capital = 100000
        self.logger.info(
            f"Initializing portfolio with capital: ${initial_capital:,.2f}"
        )
        portfolio = Portfolio("test_portfolio", initial_capital)

        # Create instruments - make it a naked call since we don't have the shares
        option = Option(
            ticker="AAPL",
            strike=205.0,
            instrument_type=InstrumentType.CALL_OPTION,
            expiration=datetime(2024, 1, 12, tzinfo=pytz.UTC),
            naked=True,  # Add this for naked call
        )
        stock = Stock(ticker="AAPL")

        # Open stock position (same parameters)
        stock_quantity = 100
        stock_price = 197.45
        stock_costs = 2.97

        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=stock_price,
            timestamp=datetime(2023, 12, 20, 14, 26, tzinfo=pytz.UTC),
            transaction_costs=stock_costs,
        )

        # Close stock position (same parameters)
        portfolio.close_position(
            instrument=stock,
            price=stock_price,
            timestamp=datetime(2023, 12, 31, 0, 0, tzinfo=pytz.UTC),
            transaction_costs=stock_costs,
        )

        # Open and close option position (same parameters)
        option_quantity = -1
        option_price = 0.55
        option_costs = 1.00

        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=option_price,
            timestamp=datetime(2023, 12, 20, 15, 26, tzinfo=pytz.UTC),
            transaction_costs=option_costs,
        )

        portfolio.close_position(
            instrument=option,
            price=option_price,
            timestamp=datetime(2023, 12, 31, 0, 0, tzinfo=pytz.UTC),
            transaction_costs=option_costs,
        )

        # Final calculations
        actual_final_capital = portfolio.cash + portfolio.get_position_value()

        # Updated assertions with proper transaction cost handling
        self.assertEqual(
            len(portfolio.positions),
            0,
            "All positions should be closed at the end of the backtest.",
        )

        # Same final capital check (this was already correct)
        self.assertAlmostEqual(
            actual_final_capital,
            99992.06,
            places=2,
            msg=f"Final capital mismatch: Expected 99,992.06, Got {actual_final_capital}",
        )

        # Updated expected PnLs to include both transaction costs
        expected_pnls = [
            -(stock_costs + stock_costs),  # Stock trade: both entry and exit costs
            -(option_costs + option_costs),  # Option trade: both entry and exit costs
        ]
        actual_pnls = [round(t.pnl, 2) for t in portfolio.trades]

        for i, (expected, actual) in enumerate(zip(expected_pnls, actual_pnls)):
            self.assertAlmostEqual(
                actual,
                expected,
                places=2,
                msg=f"Trade {i+1} PnL mismatch: Expected {expected}, Got {actual}",
            )

    ############################################################
    #               CONCURRENT POSITION PNL
    ############################################################

    def test_multiple_instruments_concurrent_positions(self):
        self.logger.info(
            "=== Starting Multiple Instruments Concurrent Positions Test ==="
        )

        # Initialize portfolio
        initial_capital = 150000
        portfolio = Portfolio("multiple_instruments_portfolio", initial_capital)

        # Create stock and option instruments
        stock_a = Stock(ticker="AAPL")
        option_a = Option(
            ticker="AAPL",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=180.0,
            expiration=datetime(2024, 1, 25, tzinfo=pytz.UTC),
        )

        stock_b = Stock(ticker="GOOGL")
        option_b = Option(
            ticker="GOOGL",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=2500.0,
            expiration=datetime(2024, 2, 20, tzinfo=pytz.UTC),
        )

        # Open multiple positions
        # Trade 1: Buy AAPL stock
        aapl_quantity = 100
        aapl_buy_price = 170.00
        aapl_buy_costs = 10.00

        self.logger.info("\n=== Trade 1: Buy AAPL Stock ===")
        portfolio.open_position(
            instrument=stock_a,
            quantity=aapl_quantity,
            price=aapl_buy_price,
            timestamp=datetime(2023, 12, 1, 9, 30, tzinfo=pytz.UTC),
            transaction_costs=aapl_buy_costs,
        )
        self.logger.info(f"Cash after buying AAPL: ${portfolio.cash:,.2f}")

        # Trade 2: Sell GOOGL put option
        googl_option_quantity = -1
        googl_option_sell_price = 50.00
        googl_option_sell_costs = 2.00

        self.logger.info("\n=== Trade 2: Sell GOOGL Put Option ===")
        portfolio.open_position(
            instrument=option_b,
            quantity=googl_option_quantity,
            price=googl_option_sell_price,
            timestamp=datetime(2023, 12, 1, 9, 35, tzinfo=pytz.UTC),
            transaction_costs=googl_option_sell_costs,
        )
        self.logger.info(f"Cash after selling GOOGL put option: ${portfolio.cash:,.2f}")

        # Trade 3: Buy GOOGL stock
        googl_quantity = 50
        googl_buy_price = 2550.00
        googl_buy_costs = 15.00

        self.logger.info("\n=== Trade 3: Buy GOOGL Stock ===")
        portfolio.open_position(
            instrument=stock_b,
            quantity=googl_quantity,
            price=googl_buy_price,
            timestamp=datetime(2023, 12, 5, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=googl_buy_costs,
        )
        self.logger.info(f"Cash after buying GOOGL: ${portfolio.cash:,.2f}")

        # Trade 4: Sell AAPL call option
        aapl_option_quantity = -1
        aapl_option_sell_price = 10.00
        aapl_option_sell_costs = 1.00

        self.logger.info("\n=== Trade 4: Sell AAPL Call Option ===")
        portfolio.open_position(
            instrument=option_a,
            quantity=aapl_option_quantity,
            price=aapl_option_sell_price,
            timestamp=datetime(2023, 12, 5, 10, 5, tzinfo=pytz.UTC),
            transaction_costs=aapl_option_sell_costs,
        )
        self.logger.info(f"Cash after selling AAPL call option: ${portfolio.cash:,.2f}")

        # Close positions in various orders
        # Close AAPL call option
        aapl_option_buy_price = 8.00
        aapl_option_buy_costs = 1.00

        self.logger.info("\n=== Trade 5: Buy Back AAPL Call Option ===")
        portfolio.close_position(
            instrument=option_a,
            price=aapl_option_buy_price,
            timestamp=datetime(2023, 12, 20, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=aapl_option_buy_costs,
        )
        self.logger.info(
            f"Cash after buying back AAPL call option: ${portfolio.cash:,.2f}"
        )

        # Close GOOGL put option
        googl_option_buy_price = 55.00
        googl_option_buy_costs = 2.00

        self.logger.info("\n=== Trade 6: Buy Back GOOGL Put Option ===")
        portfolio.close_position(
            instrument=option_b,
            price=googl_option_buy_price,
            timestamp=datetime(2023, 12, 25, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=googl_option_buy_costs,
        )
        self.logger.info(
            f"Cash after buying back GOOGL put option: ${portfolio.cash:,.2f}"
        )

        # Close AAPL stock
        aapl_sell_price = 175.00
        aapl_sell_costs = 10.00

        self.logger.info("\n=== Trade 7: Sell AAPL Stock ===")
        portfolio.close_position(
            instrument=stock_a,
            price=aapl_sell_price,
            timestamp=datetime(2023, 12, 30, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=aapl_sell_costs,
        )
        self.logger.info(f"Cash after selling AAPL: ${portfolio.cash:,.2f}")

        # Close GOOGL stock
        googl_sell_price = 2600.00
        googl_sell_costs = 15.00

        self.logger.info("\n=== Trade 8: Sell GOOGL Stock ===")
        portfolio.close_position(
            instrument=stock_b,
            price=googl_sell_price,
            timestamp=datetime(2024, 1, 5, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=googl_sell_costs,
        )
        self.logger.info(f"Cash after selling GOOGL: ${portfolio.cash:,.2f}")

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()
        total_expected_pnl = (
            (aapl_sell_price - aapl_buy_price) * aapl_quantity
            - (aapl_buy_costs + aapl_sell_costs)
            + (googl_option_sell_price - googl_option_buy_price)
            * 100
            * abs(googl_option_quantity)
            - (googl_option_sell_costs + googl_option_buy_costs)
            + (googl_sell_price - googl_buy_price) * googl_quantity
            - (googl_buy_costs + googl_sell_costs)
            + (aapl_option_sell_price - aapl_option_buy_price)
            * 100
            * abs(aapl_option_quantity)
            - (aapl_option_sell_costs + aapl_option_buy_costs)
        )

        expected_final_capital = initial_capital + total_expected_pnl

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

    def test_concurrent_positions_same_instrument(self):
        self.logger.info("=== Starting Concurrent Positions Same Instrument Test ===")

        # Initialize portfolio
        initial_capital = 120000
        portfolio = Portfolio("concurrent_same_instrument_portfolio", initial_capital)

        # Create stock and option instruments
        stock = Stock(ticker="NFLX")
        call_option = Option(
            ticker="NFLX",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=550.0,  # Example strike price
            expiration=datetime(2024, 1, 25, tzinfo=pytz.UTC),
        )

        # Trade 1: Buy NFLX Stock
        nflx_quantity = 200  # Increased to 200 to cover 2 call options
        nflx_buy_price = 500.00
        nflx_buy_costs = 5.00

        self.logger.info("\n=== Trade 1: Buy NFLX Stock ===")
        portfolio.open_position(
            instrument=stock,
            quantity=nflx_quantity,
            price=nflx_buy_price,
            timestamp=datetime(2023, 12, 1, 9, 0, tzinfo=pytz.UTC),
            transaction_costs=nflx_buy_costs,
        )
        self.logger.info(f"Cash after buying NFLX: ${portfolio.cash:,.2f}")

        # Trade 2: Sell NFLX Call Options (Covered Call)
        option_quantity = -2  # Selling 2 call options
        option_sell_price = 10.00
        option_sell_costs = 1.00

        self.logger.info("\n=== Trade 2: Sell NFLX Call Option ===")
        portfolio.open_position(
            instrument=call_option,  # Correct instrument for options
            quantity=option_quantity,
            price=option_sell_price,
            timestamp=datetime(2023, 12, 5, 10, 5, tzinfo=pytz.UTC),
            transaction_costs=option_sell_costs,
        )
        self.logger.info(f"Cash after selling NFLX call option: ${portfolio.cash:,.2f}")

        # Trade 3: Sell NFLX Stock
        nflx_sell_price = 530.00
        nflx_sell_costs = 4.00

        self.logger.info("\n=== Trade 3: Sell NFLX Stock ===")
        portfolio.close_position(
            instrument=stock,
            price=nflx_sell_price,
            timestamp=datetime(2023, 12, 20, 11, 0, tzinfo=pytz.UTC),
            transaction_costs=nflx_sell_costs,
        )
        self.logger.info(f"Cash after selling NFLX stock: ${portfolio.cash:,.2f}")

        # Trade 4: Close NFLX Call Options (Buy Back)
        option_buy_price = 8.00
        option_buy_costs = 1.00

        self.logger.info("\n=== Trade 4: Buy Back NFLX Call Option ===")
        portfolio.close_position(
            instrument=call_option,
            price=option_buy_price,
            timestamp=datetime(2023, 12, 20, 11, 30, tzinfo=pytz.UTC),
            transaction_costs=option_buy_costs,
        )
        self.logger.info(
            f"Cash after buying back NFLX call option: ${portfolio.cash:,.2f}"
        )

        # Final calculations
        final_capital = portfolio.cash + portfolio.get_position_value()

        # Detailed Cash Flow Calculation:
        # Initial Capital: $120,000

        # Trade 1: Buy 200 NFLX at $500 + $5 = $100,005
        # Cash: 120,000 -100,005 =19,995

        # Trade 2: Sell 2 options at $10 + $1 = +1,999
        # Cash:19,995 +1,999 =21,994

        # Trade 3: Sell 200 NFLX at $530 + $4 = +105,996
        # Cash:21,994 +105,996 =127,990

        # Trade 4: Buy back 2 options at $8 + $1 = -1,601
        # Cash:127,990 -1,601 =126,389

        # Final Cash:126,389
        # Final Position Value:0
        # Final Capital:126,389

        # Expected Final Capital:126,389

        expected_final_capital = (
            initial_capital
            - (nflx_quantity * nflx_buy_price + nflx_buy_costs)
            + (abs(option_quantity) * option_sell_price * 100 - option_sell_costs)
            + (nflx_quantity * nflx_sell_price - nflx_sell_costs)
            - (abs(option_quantity) * option_buy_price * 100 + option_buy_costs)
        )

        self.logger.info("\n=== Final Position State ===")
        self.logger.info(f"Final Cash: ${portfolio.cash:,.2f}")
        self.logger.info(
            f"Final Position Value: ${portfolio.get_position_value():,.2f}"
        )
        self.logger.info(f"Final Capital: ${final_capital:,.2f}")

        # Assertions
        self.assertAlmostEqual(
            final_capital,
            expected_final_capital,
            places=2,
            msg=f"Final capital mismatch: Expected {expected_final_capital}, Got {final_capital}",
        )

        # Verify number of trades
        self.assertEqual(
            len(portfolio.trades), 2, "There should be exactly four trades recorded."
        )

        # Verify individual trade PnLs
        trade1 = portfolio.trades[0]  # Buy Stock + Sell Stock
        trade2 = portfolio.trades[1]  # Sell Call Option + Buy Back Call Option

        # Trade 1 PnL: 0.00
        self.assertAlmostEqual(
            trade1.pnl,
            (nflx_sell_price - nflx_buy_price) * nflx_quantity
            - (nflx_buy_costs + nflx_sell_costs),
            places=2,
            msg=f"Trade 1 PnL mismatch: Expected 0.00, Got {trade1.pnl}",
        )

        # Trade 2 PnL: +1,999.00
        self.assertAlmostEqual(
            trade2.pnl,
            (option_sell_price - option_buy_price) * abs(option_quantity) * 100
            - (option_sell_costs + option_buy_costs),
            places=2,
            msg=f"Trade 2 PnL mismatch: Expected 1999.00, Got {trade2.pnl}",
        )

    ############################################################
    #               TRADE ENTRY CONSTRAINTS
    ############################################################

    def test_insufficient_funds_for_trade(self):
        self.logger.info("=== Starting Insufficient Funds for Trade Test ===")

        # Initialize portfolio with limited cash
        initial_capital = 1000  # Limited capital to trigger insufficient funds
        portfolio = Portfolio("insufficient_funds_portfolio", initial_capital)

        # Create expensive stock instrument
        expensive_stock = Stock(ticker="AMZN")

        # Attempt to open a large position that exceeds available cash
        stock_quantity = 1000  # Large quantity
        buy_price = 2000.00  # High price per share
        buy_costs = 50.00

        self.logger.info("\n=== Trade 1: Attempt to Buy AMZN Stock ===")
        # Since Portfolio.open_position does not raise an exception, verify via position and cash
        portfolio.open_position(
            instrument=expensive_stock,
            quantity=stock_quantity,
            price=buy_price,
            timestamp=datetime(2023, 12, 1, 12, 0, tzinfo=pytz.UTC),
            transaction_costs=buy_costs,
        )
        self.logger.info(
            f"Cash after attempting to buy AMZN stock: ${portfolio.cash:,.2f}"
        )

        # Verify that the position was not added
        self.assertNotIn(
            expensive_stock.ticker,
            portfolio.positions,
            "Expensive stock position should not be added due to insufficient funds.",
        )

        # Verify that cash remains unchanged
        self.assertEqual(
            portfolio.cash,
            initial_capital,
            "Cash should remain unchanged after failed trade.",
        )

        # Attempt to open a put option that exceeds available cash
        put_option = Option(
            ticker="AMZN",
            instrument_type=InstrumentType.PUT_OPTION,
            strike=3000.0,
            expiration=datetime(2024, 1, 20, tzinfo=pytz.UTC),
        )

        option_quantity = 10
        buy_price_option = 50.00
        buy_costs_option = 10.00

        self.logger.info("\n=== Trade 2: Attempt to Buy AMZN Put Option ===")
        portfolio.open_position(
            instrument=put_option,
            quantity=option_quantity,
            price=buy_price_option,
            timestamp=datetime(2023, 12, 1, 12, 30, tzinfo=pytz.UTC),
            transaction_costs=buy_costs_option,
        )
        self.logger.info(
            f"Cash after attempting to buy AMZN put option: ${portfolio.cash:,.2f}"
        )

        # Verify that the option position was not added
        self.assertNotIn(
            put_option.format_option_ticker(),
            portfolio.positions,
            "AMZN put option position should not be added due to insufficient funds.",
        )

        # Verify that cash remains unchanged
        self.assertEqual(
            portfolio.cash,
            initial_capital,
            "Cash should remain unchanged after failed option trade.",
        )

    def test_short_option_without_underlying(self):
        self.logger.info("=== Starting Short Option without Underlying Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio(
            "short_option_no_underlying_portfolio",
            initial_capital,
            enable_naked_options=False,
        )

        # Create a naked short
        # Option without underlying stock
        option = Option(
            ticker="TSLA",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=800.0,
            expiration=datetime(2024, 1, 20, tzinfo=pytz.UTC),
            naked=False,  # Mark as non naked to prevent entry
        )

        # Attempt to open a naked short option position
        option_quantity = -1
        sell_price = 10.00
        sell_costs = 1.00

        self.logger.info("\n=== Trade 1: Attempt to Sell TSLA Call Option ===")
        # Since Portfolio.open_position does not raise an exception, verify via position and cash
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 1, 13, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(
            f"Cash after attempting to sell TSLA call option: ${portfolio.cash:,.2f}"
        )

        # Verify that the position was not added
        self.assertNotIn(
            option.format_option_ticker(),
            portfolio.positions,
            "Naked short option position should not be added without underlying stock.",
        )

        # Verify that cash remains unchanged
        self.assertEqual(
            portfolio.cash,
            initial_capital,
            "Cash should remain unchanged after failed naked short option trade.",
        )

    def test_short_option_with_underlying(self):
        self.logger.info("=== Starting Short Option with Underlying Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("short_option_with_underlying_portfolio", initial_capital)

        # Create stock and option instruments
        stock = Stock(ticker="TSLA")
        option = Option(
            ticker="TSLA",
            instrument_type=InstrumentType.CALL_OPTION,
            strike=800.0,
            expiration=datetime(2024, 1, 20, tzinfo=pytz.UTC),
        )

        # Open stock position
        stock_quantity = 100
        stock_price = 750.00
        stock_costs = 5.00

        self.logger.info("\n=== Trade 1: Buy TSLA Stock ===")
        portfolio.open_position(
            instrument=stock,
            quantity=stock_quantity,
            price=stock_price,
            timestamp=datetime(2023, 12, 1, 14, 0, tzinfo=pytz.UTC),
            transaction_costs=stock_costs,
        )
        self.logger.info(f"Cash after buying TSLA stock: ${portfolio.cash:,.2f}")

        # Attempt to open a short option position
        option_quantity = -1
        sell_price = 10.00
        sell_costs = 1.00

        self.logger.info("\n=== Trade 2: Attempt to Sell TSLA Call Option ===")
        portfolio.open_position(
            instrument=option,
            quantity=option_quantity,
            price=sell_price,
            timestamp=datetime(2023, 12, 5, 10, 0, tzinfo=pytz.UTC),
            transaction_costs=sell_costs,
        )
        self.logger.info(
            f"Cash after attempting to sell TSLA call option: ${portfolio.cash:,.2f}"
        )

        # Verify that the position was added
        self.assertIn(
            option.format_option_ticker(),
            portfolio.positions,
            "Short option position should be added with underlying stock.",
        )

        # Verify that cash impact is correct
        expected_cash = initial_capital - (
            stock_quantity * stock_price + stock_costs
        )  # Cash reduced by stock purchase
        expected_cash += (
            sell_price * abs(option_quantity) * option.get_multiplier() - sell_costs
        )

        self.assertEqual(
            portfolio.cash,
            expected_cash,
            "Cash should be reduced by transaction costs after successful short option trade.",
        )

    ############################################################
    #               TRADE OPENING & CLOSING
    ############################################################

    def test_long_position_additions_and_closures(self):
        self.logger.info("=== Starting Position Additions and Closures Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("position_additions_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="AAPL")

        # Initial position open
        initial_quantity = 100
        initial_price = 175.00
        initial_costs = 5.00

        self.logger.info("\n=== Trade 1: Initial AAPL Stock Purchase ===")
        portfolio.open_position(
            instrument=stock,
            quantity=initial_quantity,
            price=initial_price,
            timestamp=datetime(2023, 12, 1, 14, 0, tzinfo=pytz.UTC),
            transaction_costs=initial_costs,
        )

        # Verify initial position
        self.assertIn(
            stock.ticker, portfolio.positions, "Initial stock position should be added"
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            initial_quantity,
            "Initial quantity should be correct",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            initial_price,
            "Initial entry price should be correct",
        )

        # Add to position
        additional_quantity = 50
        additional_price = 180.00
        additional_costs = 3.00

        self.logger.info("\n=== Trade 2: Adding to AAPL Position ===")
        portfolio.open_position(
            instrument=stock,
            quantity=additional_quantity,
            price=additional_price,
            timestamp=datetime(2023, 12, 2, 14, 0, tzinfo=pytz.UTC),
            transaction_costs=additional_costs,
        )

        # Calculate expected weighted average price
        expected_total_value = (initial_quantity * initial_price) + (
            additional_quantity * additional_price
        )
        expected_total_quantity = initial_quantity + additional_quantity
        expected_avg_price = expected_total_value / expected_total_quantity
        expected_total_costs = initial_costs + additional_costs

        # Verify combined position
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            expected_total_quantity,
            "Combined quantity should be correct",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            expected_avg_price,
            "Weighted average price should be correct",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_transaction_costs,
            expected_total_costs,
            "Combined transaction costs should be correct",
        )

        # Partial close
        close_quantity = 75
        close_price = 185.00
        close_costs = 4.00

        self.logger.info("\n=== Trade 3: Partial Close of AAPL Position ===")
        portfolio.close_position(
            instrument=stock,
            price=close_price,
            timestamp=datetime(2023, 12, 3, 14, 0, tzinfo=pytz.UTC),
            quantity=close_quantity,
            transaction_costs=close_costs,
        )

        # Verify remaining position
        expected_remaining_quantity = expected_total_quantity - close_quantity
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            expected_remaining_quantity,
            "Remaining quantity should be correct",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            expected_avg_price,
            "Entry price should remain unchanged",
        )

        # Calculate expected remaining entry costs (proportional)
        expected_remaining_costs = expected_total_costs * (
            expected_remaining_quantity / expected_total_quantity
        )
        self.assertAlmostEqual(
            portfolio.positions[stock.ticker].entry_transaction_costs,
            expected_remaining_costs,
            places=2,
            msg="Remaining entry costs should be proportional",
        )

        # Full close
        final_close_price = 190.00
        final_close_costs = 3.00

        self.logger.info("\n=== Trade 4: Full Close of Remaining AAPL Position ===")
        portfolio.close_position(
            instrument=stock,
            price=final_close_price,
            timestamp=datetime(2023, 12, 4, 14, 0, tzinfo=pytz.UTC),
            quantity=expected_remaining_quantity,
            transaction_costs=final_close_costs,
        )

        # Verify position is fully closed
        self.assertNotIn(
            stock.ticker, portfolio.positions, "Position should be fully closed"
        )

        # Verify all trades were recorded
        self.assertEqual(
            len(portfolio.trades), 2, "Should have two closing trades recorded"
        )

        # Verify trade objects
        partial_close_trade = portfolio.trades[0]
        self.assertEqual(
            partial_close_trade.quantity,
            close_quantity,
            "First trade quantity should match partial close",
        )
        self.assertEqual(
            partial_close_trade.exit_price,
            close_price,
            "First trade exit price should match partial close",
        )

        final_close_trade = portfolio.trades[1]
        self.assertEqual(
            final_close_trade.quantity,
            expected_remaining_quantity,
            "Second trade quantity should match final close",
        )
        self.assertEqual(
            final_close_trade.exit_price,
            final_close_price,
            "Second trade exit price should match final close",
        )

        # Verify cash impacts
        expected_final_cash = (
            initial_capital
            - (initial_quantity * initial_price + initial_costs)  # Initial purchase
            - (
                additional_quantity * additional_price + additional_costs
            )  # Addition to position
            + (close_quantity * close_price - close_costs)  # Partial close
            + (
                expected_remaining_quantity * final_close_price - final_close_costs
            )  # Final close
        )

        self.assertAlmostEqual(
            portfolio.cash,
            expected_final_cash,
            places=2,
            msg="Final cash should reflect all transactions",
        )

    def test_short_position_additions_and_closures(self):
        self.logger.info("=== Starting Short Position Additions and Closures Test ===")

        # Initialize portfolio
        initial_capital = 100000
        portfolio = Portfolio("short_position_portfolio", initial_capital)

        # Create stock instrument
        stock = Stock(ticker="AAPL")

        # Initial short position
        initial_quantity = -100  # Start with short 100
        initial_price = 175.00
        initial_costs = 5.00

        self.logger.info("\n=== Trade 1: Initial AAPL Short Sale ===")
        portfolio.open_position(
            instrument=stock,
            quantity=initial_quantity,
            price=initial_price,
            timestamp=datetime(2023, 12, 1, 14, 0, tzinfo=pytz.UTC),
            transaction_costs=initial_costs,
        )

        # Verify initial position
        self.assertIn(
            stock.ticker, portfolio.positions, "Initial short position should be added"
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            initial_quantity,
            "Initial short quantity should be negative",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            initial_price,
            "Initial entry price should be correct",
        )

        # Add to short position (make it more negative)
        additional_quantity = -50  # Short 50 more
        additional_price = 180.00
        additional_costs = 3.00

        self.logger.info("\n=== Trade 2: Adding to AAPL Short Position ===")
        portfolio.open_position(
            instrument=stock,
            quantity=additional_quantity,
            price=additional_price,
            timestamp=datetime(2023, 12, 2, 14, 0, tzinfo=pytz.UTC),
            transaction_costs=additional_costs,
        )

        # Calculate expected weighted average price
        expected_total_value = (abs(initial_quantity) * initial_price) + (
            abs(additional_quantity) * additional_price
        )
        expected_total_quantity = (
            initial_quantity + additional_quantity
        )  # Should be -150
        expected_avg_price = expected_total_value / abs(expected_total_quantity)
        expected_total_costs = initial_costs + additional_costs

        # Verify combined position
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            expected_total_quantity,
            "Combined short quantity should be -150",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            expected_avg_price,
            "Weighted average price should be correct",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_transaction_costs,
            expected_total_costs,
            "Combined transaction costs should be correct",
        )

        # Partial cover (close part of short)
        cover_quantity = 60  # Cover 60 shares (should make position less negative)
        cover_price = 170.00  # Covering at a profit
        cover_costs = 4.00

        self.logger.info("\n=== Trade 3: Partial Cover of AAPL Short Position ===")
        portfolio.close_position(
            instrument=stock,
            price=cover_price,
            timestamp=datetime(2023, 12, 3, 14, 0, tzinfo=pytz.UTC),
            quantity=cover_quantity,
            transaction_costs=cover_costs,
        )

        # Verify remaining position
        expected_remaining_quantity = (
            expected_total_quantity + cover_quantity
        )  # Should be -90
        self.assertEqual(
            portfolio.positions[stock.ticker].quantity,
            expected_remaining_quantity,
            "Remaining short quantity should be -90",
        )
        self.assertEqual(
            portfolio.positions[stock.ticker].entry_price,
            expected_avg_price,
            "Entry price should remain unchanged",
        )

        # Calculate expected remaining entry costs (proportional)
        expected_remaining_costs = expected_total_costs * (
            abs(expected_remaining_quantity) / abs(expected_total_quantity)
        )
        self.assertAlmostEqual(
            portfolio.positions[stock.ticker].entry_transaction_costs,
            expected_remaining_costs,
            places=2,
            msg="Remaining entry costs should be proportional",
        )

        # Full cover (close remaining short)
        final_cover_price = 165.00  # Covering remaining at a profit
        final_cover_costs = 3.00

        self.logger.info(
            "\n=== Trade 4: Full Cover of Remaining AAPL Short Position ==="
        )
        portfolio.close_position(
            instrument=stock,
            price=final_cover_price,
            timestamp=datetime(2023, 12, 4, 14, 0, tzinfo=pytz.UTC),
            quantity=abs(expected_remaining_quantity),  # Cover the remaining 90 shares
            transaction_costs=final_cover_costs,
        )

        # Verify position is fully closed
        self.assertNotIn(
            stock.ticker, portfolio.positions, "Position should be fully closed"
        )

        # Verify all trades were recorded
        self.assertEqual(
            len(portfolio.trades), 2, "Should have two covering trades recorded"
        )

        # Verify trade objects
        partial_cover_trade = portfolio.trades[0]
        self.assertEqual(
            partial_cover_trade.quantity,
            cover_quantity,
            "First trade quantity should match partial cover",
        )
        self.assertEqual(
            partial_cover_trade.exit_price,
            cover_price,
            "First trade exit price should match partial cover",
        )

        final_cover_trade = portfolio.trades[1]
        self.assertEqual(
            final_cover_trade.quantity,
            abs(expected_remaining_quantity),
            "Second trade quantity should match final cover",
        )
        self.assertEqual(
            final_cover_trade.exit_price,
            final_cover_price,
            "Second trade exit price should match final cover",
        )

        # Verify cash impacts
        expected_final_cash = (
            initial_capital
            + (
                abs(initial_quantity) * initial_price - initial_costs
            )  # Initial short sale
            + (
                abs(additional_quantity) * additional_price - additional_costs
            )  # Addition to short
            - (cover_quantity * cover_price + cover_costs)  # Partial cover
            - (
                abs(expected_remaining_quantity) * final_cover_price + final_cover_costs
            )  # Final cover
        )

        self.assertAlmostEqual(
            portfolio.cash,
            expected_final_cash,
            places=2,
            msg="Final cash should reflect all short transactions",
        )
