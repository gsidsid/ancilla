# ancilla/backtesting/portfolio.py
from datetime import datetime
from typing import Dict, Optional, List, TYPE_CHECKING

from ancilla.backtesting.configuration import Broker
from ancilla.models import Trade, Position, Instrument, InstrumentType, Stock
from ancilla.utils.logging import BookLogger

if TYPE_CHECKING:
    from ancilla.backtesting.configuration import Broker


class Portfolio:
    """Manages capital, positions, and trades during backtests."""

    def __init__(
        self, name: str, initial_capital: float, enable_naked_options: bool = True
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.opening_cash_flows: List[float] = []
        self.logger = BookLogger(name)
        self.logger.capital_update(datetime.now(), self.cash, 0, self.initial_capital)
        self.enable_naked_options = enable_naked_options

    def open_position(
        self,
        instrument: Instrument,
        quantity: int,
        price: float,
        timestamp: datetime,
        transaction_costs: float = 0.0,
        allow_naked_calls: bool = False,
        is_assignment: bool = False,
        is_exercise: bool = False,
    ) -> bool:
        """Open or add to a position with accurate cash flow tracking."""
        if instrument.is_option and quantity < 0:  # Short options
            if instrument.instrument_type == InstrumentType.CALL_OPTION:
                # COVERED CALL
                required_shares = quantity * 100
                share_check = (
                    self.positions[instrument.underlying_ticker].quantity
                    < required_shares
                    if instrument.underlying_ticker in self.positions
                    else True
                )
                warning_msg = (
                    f"Insufficient shares for covered call: need {required_shares}, "
                )
            else:
                # CASH SECURED PUT
                required_shares = quantity * 100 * -1
                share_check = (
                    self.positions[instrument.underlying_ticker].quantity
                    > required_shares
                    if instrument.underlying_ticker in self.positions
                    else True
                )
                warning_msg = f"Insufficient shares for cash-secured put: need {required_shares}, "
            if share_check:
                self.logger.get_logger().warning(
                    warning_msg
                    + f"have {self.positions.get(instrument.underlying_ticker, Position(instrument, 0, 0, timestamp)).quantity}. "
                    f"May automatically liquidate position."
                )
                if not self.enable_naked_options:
                    self.logger.get_logger().error(
                        "Naked options are not enabled. This position will not be opened."
                    )
                    return False

        multiplier = instrument.get_multiplier()
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        cash_impact = -(price * quantity * multiplier) - transaction_costs

        # Check for sufficient cash if buying
        if quantity > 0 and (-cash_impact) > self.cash:
            self.logger.get_logger().error(
                f"Insufficient cash for {instrument.ticker}: "
                f"need ${-cash_impact:,.2f}, have ${self.cash:,.2f}"
            )
            return False

        # Update cash
        self.cash += cash_impact

        # Record the opening cash flow
        if cash_impact > 0:
            self.opening_cash_flows.append(cash_impact)

        # Update or create the position
        if ticker in self.positions:
            existing_position = self.positions[ticker]
            # Calculate new average entry price
            total_value = (
                existing_position.entry_price * existing_position.quantity
            ) + (price * quantity)
            new_quantity = existing_position.quantity + quantity
            new_price = total_value / new_quantity if new_quantity != 0 else price

            # Update the position
            self.positions[ticker] = Position(
                instrument=instrument,
                quantity=new_quantity,
                entry_price=new_price,
                entry_date=existing_position.entry_date,  # Keep original entry date
                entry_transaction_costs=existing_position.entry_transaction_costs
                + transaction_costs,
                assignment=existing_position.assignment or is_assignment,
                exercised=existing_position.exercised or is_exercise,
            )
        else:
            # Create new position
            self.positions[ticker] = Position(
                instrument=instrument,
                quantity=quantity,
                entry_price=price,
                entry_date=timestamp,
                entry_transaction_costs=transaction_costs,
                assignment=is_assignment,
                exercised=is_exercise,
            )

        # Log the transaction
        self.logger.position_open(
            timestamp=timestamp,
            ticker=instrument.ticker,
            quantity=quantity,  # Log the increment amount
            price=price,
            position_type="option" if instrument.is_option else "stock",
            capital=self.cash,
        )
        self.logger.capital_update(
            timestamp, self.cash, self.get_position_value(), self.get_total_value()
        )

        self.log_position_status()

        return True

    def close_position(
        self,
        instrument: Instrument,
        price: float,
        timestamp: datetime,
        quantity: Optional[int] = None,
        transaction_costs: float = 0.0,
        realized_pnl: Optional[float] = None,
        is_assignment: bool = False,
        is_exercise: bool = False,
    ) -> bool:
        """Close a specified quantity of a position with accurate PnL tracking."""
        ticker = (
            instrument.ticker
            if not instrument.is_option
            else instrument.format_option_ticker()
        )
        if ticker not in self.positions:
            self.logger.get_logger().warning(f"No open position found for {ticker}")
            return False

        position = self.positions[ticker]
        position_quantity = position.quantity
        position_type = "option" if instrument.is_option else "stock"
        multiplier = instrument.get_multiplier()
        if quantity is None:
            quantity = position_quantity

        # Ensure the close quantity does not exceed the position quantity
        # TODO: Should abs(quantity) > abs(position_quantity) imply opening a new position?
        # I think it's better to force positions to be addressed independently
        if abs(quantity) > abs(position_quantity):
            self.logger.get_logger().warning(
                f"Attempting to close {quantity} of {ticker}, but only {position_quantity} is available."
            )
            return False

        # Update cash, PnL
        entry_price = position.entry_price
        pnl = 0.0
        cash_impact = 0.0
        if position_quantity < 0:
            # Closing a short option position (buying back)
            pnl = (position.entry_price - price) * abs(quantity) * multiplier
            cash_impact = -(price * abs(quantity) * multiplier) - transaction_costs
        else:
            # Closing a long option position (selling)
            pnl = (price - position.entry_price) * quantity * multiplier
            cash_impact = (price * quantity * multiplier) - transaction_costs
        self.cash += cash_impact

        # Calculate proportional entry transaction costs for the quantity being closed
        ratio = abs(quantity / position_quantity)
        proportional_entry_costs = position.entry_transaction_costs * ratio
        total_transaction_costs = proportional_entry_costs + transaction_costs

        # Close the position by finalizing a trade
        realized_pnl = pnl - total_transaction_costs
        closing_trade = Trade(
            instrument=instrument,
            entry_time=position.entry_date,
            exit_time=timestamp,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            transaction_costs=total_transaction_costs,  # Now includes proportional entry costs
            realized_pnl=realized_pnl,
            assignment=is_assignment or position.assignment,
            exercised=is_exercise or position.exercised,
        )
        self.trades.append(closing_trade)

        # Retain directionality
        if (position_quantity > 0 and quantity > 0) or (
            position_quantity < 0 and quantity < 0
        ):
            remaining_quantity = position_quantity - quantity
        else:
            remaining_quantity = position_quantity + quantity

        if remaining_quantity == 0:
            # Fully close the position
            del self.positions[ticker]
            self.logger.position_close(
                timestamp=timestamp,
                ticker=ticker,
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=self.cash,
            )
        else:
            # Partially close
            remaining_ratio = abs(remaining_quantity / position_quantity)
            remaining_entry_costs = position.entry_transaction_costs * remaining_ratio
            position.quantity = remaining_quantity
            position.entry_transaction_costs = (
                remaining_entry_costs  # Update remaining entry costs
            )
            self.positions[ticker] = position
            self.logger.get_logger().info(
                f"Partially closed {quantity} of {ticker}. Remaining quantity: {remaining_quantity}"
            )
            self.logger.position_close(
                timestamp=timestamp,
                ticker=ticker,
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=self.cash,
            )

        # Log the closure
        self.logger.trade_complete(timestamp, closing_trade)
        self.logger.capital_update(
            timestamp, self.cash, self.get_position_value(), self.get_total_value()
        )
        self.log_position_status()

        return True

    def handle_assignment(
        self,
        option: Instrument,
        strike_price: float,
        timestamp: datetime,
        is_call: bool,
        broker: Broker,
    ) -> bool:
        """
        Handle the assignment of a short option position.
        """
        ticker = option.format_option_ticker()
        underlying_ticker = option.underlying_ticker
        contract_quantity = self.positions[ticker].quantity
        share_quantity = int(contract_quantity * option.get_multiplier())

        if is_call:
            # Short Call Assignment: Sell underlying stock at strike price
            self.logger.get_logger().info(
                f"Assigning short Call Option for {ticker}: Selling {share_quantity} shares at ${strike_price:.2f}"
            )
            # Close the corresponding covered position
            success = self.close_position(
                instrument=Stock(underlying_ticker),
                price=strike_price,
                timestamp=timestamp,
                quantity=-share_quantity,
                transaction_costs=broker.calculate_commission(
                    strike_price, share_quantity, "stock"
                ),
                is_assignment=True,
                realized_pnl=0,
            )
        else:
            # Short Put Assignment: Buy underlying stock at strike price
            self.logger.get_logger().info(
                f"Assigning short Put Option for {ticker}: Buying {share_quantity} shares at ${strike_price:.2f}"
            )
            # Open a new position for the underlying stock
            success = self.open_position(
                instrument=Stock(underlying_ticker),
                quantity=share_quantity,
                price=strike_price,
                timestamp=timestamp,
                transaction_costs=broker.calculate_commission(
                    strike_price, share_quantity, "stock"
                ),
                is_assignment=True,
            )

        # After assignment, close the option position
        if self.close_position(
            instrument=option,
            price=0.0,  # Option is assigned/exercised; no residual value
            timestamp=timestamp,
            transaction_costs=broker.calculate_commission(
                0.0, contract_quantity, "option"
            ),
            is_assignment=True,
        ):
            self.logger.get_logger().info(
                f"Option {ticker} position closed due to assignment."
            )
            if ticker in self.positions:
                del self.positions[ticker]
            return True
        else:
            self.logger.get_logger().error(
                f"Failed to close option position {ticker} after assignment."
            )
            return False

    def handle_exercise(
        self,
        option: Instrument,
        strike_price: float,
        timestamp: datetime,
        is_call: bool,
        intrinsic_value: float,
        broker: Broker,
    ) -> bool:
        """
        Handle the exercise of a long option position.
        """
        ticker = option.format_option_ticker()
        underlying_ticker = option.underlying_ticker
        contract_quantity = abs(self.positions[ticker].quantity)
        share_quantity = int(contract_quantity * option.get_multiplier())

        if is_call:
            # Long Call Exercise: Buy underlying stock at strike price
            self.logger.get_logger().info(
                f"Exercising long Call Option for {ticker}: Buying {share_quantity} shares at ${strike_price:.2f}"
            )
            # Open a new position for the underlying stock
            success = self.open_position(
                instrument=Stock(underlying_ticker),
                quantity=share_quantity,
                price=strike_price,
                timestamp=timestamp,
                transaction_costs=broker.calculate_commission(
                    strike_price, share_quantity, "stock"
                ),
                is_exercise=True,
            )
        else:
            # Long Put Exercise: Sell underlying stock at strike price
            self.logger.get_logger().info(
                f"Exercising long Put Option for {ticker}: Selling {share_quantity} shares at ${strike_price:.2f}"
            )
            # Close the corresponding stock position if exists
            success = self.close_position(
                instrument=Stock(underlying_ticker),
                quantity=share_quantity,
                price=strike_price,
                timestamp=timestamp,
                transaction_costs=broker.calculate_commission(
                    strike_price, share_quantity, "stock"
                ),
                is_exercise=True,
            )

        # After exercise, close the option position
        if self.close_position(
            instrument=option,
            price=intrinsic_value or 0.0,
            timestamp=timestamp,
            transaction_costs=broker.calculate_commission(
                intrinsic_value or 0.0, contract_quantity, "option"
            ),
            is_exercise=True,
        ):
            self.logger.get_logger().info(
                f"Option {ticker} position closed due to exercise."
            )
            if ticker in self.positions:
                del self.positions[ticker]

            return True
        else:
            self.logger.get_logger().error(
                f"Failed to close option position {ticker} after exercise."
            )
            return False

    def payout_interest(self, interest_rate: float) -> float:
        """Calculate monthly interest on uninvested cash using federal funds rate."""
        monthly_payout = self.cash * interest_rate / 12
        self.cash += monthly_payout
        return monthly_payout

    def payout_dividend(self, dividend_ticker: str, cash_amount: float) -> float:
        """Payout dividends to the portfolio by crediting cash per held share."""
        dividend_payout = 0.0
        if dividend_ticker in self.positions:
            position = self.positions[dividend_ticker]
            if position.quantity > 0:
                dividend_payout = cash_amount * position.quantity
                self.cash += dividend_payout
                self.logger.get_logger().info(
                    f"Dividend payout for holding {dividend_ticker}. Total paid ${dividend_payout:.2f}"
                )
        return dividend_payout

    def get_position_value(
        self, market_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """Get the total value of all open positions."""
        total_value = 0
        for ticker, position in self.positions.items():
            if market_prices and ticker in market_prices:
                price = market_prices[ticker]
            else:
                price = position.entry_price
            if price is not None:
                multiplier = position.instrument.get_multiplier()
                value = position.quantity * price * multiplier
                total_value += value
        return total_value

    def get_total_value(
        self, market_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """Get the total value of the portfolio."""
        position_value = self.get_position_value(market_prices)
        total = self.cash + position_value
        return total

    def update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Update equity curve with current market prices."""
        current_equity = self.get_total_value(market_prices)
        self.equity_curve.append((timestamp, current_equity))

    def log_position_status(self):
        """Add this to Portfolio class to help debug position states"""
        self.logger.get_logger().info("Current Portfolio Positions:")
        for ticker, position in self.positions.items():
            position_type = "option" if position.instrument.is_option else "stock"
            self.logger.get_logger().info(
                f"{ticker}: {position.quantity} units @ {position.entry_price} ({position_type})"
            )
