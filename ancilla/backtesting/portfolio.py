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

    def __init__(self, name: str, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.opening_cash_flows: List[float] = []
        self.logger = BookLogger(name)
        self.logger.capital_update(
            datetime.now(),
            self.cash,
            0,
            self.initial_capital
        )

    def get_position_value(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Get the total value of all open positions."""
        total_value = 0
        for ticker, position in self.positions.items():
            if not position.instrument.is_option:
                if market_prices and ticker in market_prices:
                    price = market_prices[ticker]
                else:
                    price = position.entry_price
                if price is not None:
                    value = position.quantity * price
                    total_value += value
        for ticker, position in self.positions.items():
            if position.instrument.is_option:
                if market_prices and ticker in market_prices:
                    price = market_prices[ticker]
                else:
                    price = position.entry_price
                if price is not None:
                    multiplier = position.instrument.get_multiplier()
                    value = position.quantity * price * multiplier
                    total_value += value
        return total_value

    def get_total_value(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Get the total value of the portfolio."""
        position_value = self.get_position_value(market_prices)
        total = self.cash + position_value
        return total

    def open_position(
        self,
        instrument: Instrument,
        quantity: int,
        price: float,
        timestamp: datetime,
        transaction_costs: float = 0.0,
        allow_naked_calls: bool = False,
        is_assignment: bool = False,
        is_exercise: bool = False
    ) -> bool:
        """Open or add to a position with accurate cash flow tracking."""
        if (instrument.is_option
            and instrument.instrument_type == InstrumentType.CALL_OPTION
            and quantity < 0  # Short call
            and not instrument.naked):  # type: ignore

            required_shares = abs(quantity) * instrument.get_multiplier()
            if (instrument.underlying_ticker not in self.positions
                or self.positions[instrument.underlying_ticker].quantity < required_shares):
                    self.logger.get_logger().warning(
                        f"Insufficient shares for covered call: need {required_shares}, "
                        f"have {self.positions.get(instrument.underlying_ticker, Position(instrument, 0, 0, timestamp)).quantity}. "
                        f"May automatically liquidate position."
                    )

        multiplier = instrument.get_multiplier()
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        cash_impact = 0.0

        if instrument.is_option:
            is_covered_call = (
                instrument.instrument_type == InstrumentType.CALL_OPTION
                and quantity < 0
                and instrument.underlying_ticker in self.positions
                and self.positions[instrument.underlying_ticker].quantity > 0
            )

            if is_covered_call:
                # Selling a covered call: Receive premium
                cash_impact = (price * abs(quantity) * multiplier) - transaction_costs
            else:
                if quantity > 0:
                    # Buying an option: Pay premium
                    cash_impact = -(price * quantity * multiplier) - transaction_costs
                else:
                    # Selling an option: Receive premium
                    cash_impact = (price * abs(quantity) * multiplier) - transaction_costs
        else:
            if quantity > 0:
                # Buying stock: Pay for stocks
                cash_impact = -(price * quantity) - transaction_costs
            else:
                # Selling stock: Receive cash
                cash_impact = (price * abs(quantity)) - transaction_costs

        # Check for sufficient cash if buying
        if quantity > 0 and (-cash_impact) > self.cash:
            self.logger.get_logger().warning(
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
            total_value = (existing_position.entry_price * existing_position.quantity) + (price * quantity)
            new_quantity = existing_position.quantity + quantity
            new_price = total_value / new_quantity if new_quantity != 0 else price

            # Update the position
            self.positions[ticker] = Position(
                instrument=instrument,
                quantity=new_quantity,
                entry_price=new_price,
                entry_date=existing_position.entry_date,  # Keep original entry date
                entry_transaction_costs=existing_position.entry_transaction_costs + transaction_costs,
                assignment=existing_position.assignment or is_assignment,
                exercised=existing_position.exercised or is_exercise
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
                exercised=is_exercise
            )

        # Log the transaction
        self.logger.position_open(
            timestamp=timestamp,
            ticker=instrument.ticker,
            quantity=quantity,  # Log the increment amount
            price=price,
            position_type='option' if instrument.is_option else 'stock',
            capital=self.cash
        )
        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(),
            self.get_total_value()
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
        skip_cash_impact: bool = False
    ) -> bool:
        """Close a specified quantity of a position with accurate PnL tracking."""
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        if ticker not in self.positions:
            self.logger.get_logger().warning(f"No open position found for {ticker}")
            return False

        position = self.positions[ticker]
        position_quantity = position.quantity
        if quantity is None:
            # Close the entire position
            quantity = position_quantity

        # Ensure the close quantity does not exceed the position quantity
        if (position_quantity > 0 and quantity > position_quantity) or \
            (position_quantity < 0 and quantity < position_quantity):
            self.logger.get_logger().warning(
                f"Attempting to close {quantity} of {ticker}, but only {position_quantity} is available."
            )
            return False

        entry_price = position.entry_price
        pnl = 0.0
        cash_impact = 0.0

        # Calculate P&L for the specified quantity
        if instrument.is_option:
            multiplier = instrument.get_multiplier()
            if position_quantity < 0:
                # Closing a short option position (buying back)
                pnl = (position.entry_price - price) * abs(quantity) * multiplier
                cash_impact = -(price * abs(quantity) * multiplier) - transaction_costs
            else:
                # Closing a long option position (selling)
                pnl = (price - position.entry_price) * quantity * multiplier
                cash_impact = (price * quantity * multiplier) - transaction_costs
        else:
            # For stocks
            if position_quantity < 0:
                # Closing a short stock position (buying back)
                pnl = (position.entry_price - price) * abs(quantity)
                cash_impact = -(price * abs(quantity)) - transaction_costs  # Always pay when covering
            else:
                # Closing a long stock position (selling)
                pnl = (price - position.entry_price) * quantity
                cash_impact = (price * quantity) - transaction_costs  # Receive when selling

        # Update cash only if not skipping impact
        if not skip_cash_impact:
            self.cash += cash_impact

        # Calculate proportional entry transaction costs for the quantity being closed
        ratio = abs(quantity / position_quantity)
        proportional_entry_costs = position.entry_transaction_costs * ratio
        total_transaction_costs = proportional_entry_costs + transaction_costs

        # Always calculate realized PnL regardless of skip_cash_impact
        realized_pnl = pnl - total_transaction_costs

        # Create Trade object for the closed quantity
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
            exercised=is_exercise or position.exercised
        )
        self.trades.append(closing_trade)

        # Retain directionality
        if (position_quantity > 0 and quantity > 0) or (position_quantity < 0 and quantity < 0):
            remaining_quantity = position_quantity - quantity
        else:
            remaining_quantity = position_quantity + quantity

        if remaining_quantity == 0:
            # Fully close the position
            del self.positions[ticker]

            position_type = 'option' if instrument.is_option else 'stock'
            self.logger.position_close(
                timestamp=timestamp,
                ticker=ticker,
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=self.cash
            )
        else:
            # Partially close the position
            # Update remaining entry transaction costs
            remaining_ratio = abs(remaining_quantity / position_quantity)
            remaining_entry_costs = position.entry_transaction_costs * remaining_ratio

            position.quantity = remaining_quantity
            position.entry_transaction_costs = remaining_entry_costs  # Update remaining entry costs
            self.positions[ticker] = position

            position_type = 'option' if instrument.is_option else 'stock'
            self.logger.get_logger().info(
                f"Partially closed {quantity} of {ticker}. Remaining quantity: {remaining_quantity}"
            )
            self.logger.position_close(
                timestamp=timestamp,
                ticker=ticker,
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=self.cash
            )

        # Log the closure
        self.logger.trade_complete(timestamp, closing_trade)
        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(),
            self.get_total_value()
        )

        self.log_position_status()

        return True

    def update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Update equity curve with current market prices."""
        current_equity = self.get_total_value(market_prices)
        self.equity_curve.append((timestamp, current_equity))

    def log_position_status(self):
        """Add this to Portfolio class to help debug position states"""
        self.logger.get_logger().info("Current Portfolio Positions:")
        for ticker, position in self.positions.items():
            position_type = 'option' if position.instrument.is_option else 'stock'
            self.logger.get_logger().info(f"{ticker}: {position.quantity} units @ {position.entry_price} ({position_type})")

    def handle_assignment(
        self,
        option: Instrument,
        strike_price: float,
        timestamp: datetime,
        is_call: bool,
        broker: Broker
    ) -> bool:
        """
        Handle the assignment of a short option position.
        """
        ticker = option.format_option_ticker()
        underlying_ticker = option.underlying_ticker
        contract_quantity = self.positions[ticker].quantity
        share_quantity = contract_quantity * 100  # Standard option contract size

        if is_call:
            # Short Call Assignment: Sell underlying stock at strike price
            self.logger.get_logger().info(f"Assigning short Call Option for {ticker}: Selling {share_quantity} shares at ${strike_price:.2f}")

            # Close the corresponding covered position
            success = self.close_position(
                instrument=Stock(underlying_ticker),
                price=strike_price,
                timestamp=timestamp,
                quantity=-share_quantity,
                transaction_costs=broker.calculate_commission(
                    strike_price,
                    share_quantity,
                    'stock'
                ),
                is_assignment=True,
                realized_pnl=0
            )
        else:
            # Short Put Assignment: Buy underlying stock at strike price
            self.logger.get_logger().info(f"Assigning short Put Option for {ticker}: Buying {share_quantity} shares at ${strike_price:.2f}")
            instrument = Stock(underlying_ticker)
            # Open a new position for the underlying stock
            success = self.open_position(
                instrument=instrument,
                quantity=share_quantity,
                price=strike_price,
                timestamp=timestamp,
                transaction_costs=broker.calculate_commission(
                    strike_price,
                    share_quantity,
                    'stock'
                ),
                is_assignment=True
            )

        # After assignment, close the option position
        if self.close_position(
            instrument=option,
            price=0.0,  # Option is assigned/exercised; no residual value
            timestamp=timestamp,
            transaction_costs=broker.calculate_commission(
                0.0,
                contract_quantity,
                'option'
            ),
            is_assignment=True
        ):
            self.logger.get_logger().info(f"Option {ticker} position closed due to assignment.")
            if ticker in self.positions:
                del self.positions[ticker]

            return True
        else:
            self.logger.get_logger().error(f"Failed to close option position {ticker} after assignment.")
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
        share_quantity = contract_quantity * 100  # Standard option contract size

        if is_call:
            # Long Call Exercise: Buy underlying stock at strike price
            self.logger.get_logger().info(f"Exercising long Call Option for {ticker}: Buying {share_quantity} shares at ${strike_price:.2f}")
            instrument = Stock(underlying_ticker)
            # Open a new position for the underlying stock
            success = self.open_position(
                instrument=instrument,
                quantity=share_quantity,
                price=strike_price,
                timestamp=timestamp,
                transaction_costs=broker.calculate_commission(
                    strike_price,
                    share_quantity,
                    'stock'
                ),
                is_exercise=True,
            )
        else:
            # Long Put Exercise: Sell underlying stock at strike price
            self.logger.get_logger().info(f"Exercising long Put Option for {ticker}: Selling {share_quantity} shares at ${strike_price:.2f}")
            instrument = Stock(underlying_ticker)
            # Close the corresponding stock position if exists
            success = self.close_position(
                instrument=instrument,
                price=strike_price,
                timestamp=timestamp,
                quantity=share_quantity,
                transaction_costs=broker.calculate_commission(
                    strike_price,
                    share_quantity,
                    'stock'
                ),
                is_exercise=True,
            )

        # After exercise, close the option position
        if self.close_position(
            instrument=option,
            price=intrinsic_value or 0.0,
            timestamp=timestamp,
            transaction_costs=broker.calculate_commission(
                intrinsic_value or 0.0,
                contract_quantity,
                'option'
            ),
            is_exercise=True
        ):
            self.logger.get_logger().info(f"Option {ticker} position closed due to exercise.")
            if ticker in self.positions:
                del self.positions[ticker]

            return True
        else:
            self.logger.get_logger().error(f"Failed to close option position {ticker} after exercise.")
            return False
