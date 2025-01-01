# ancilla/backtesting/portfolio.py
from datetime import datetime
from typing import Dict, Optional, List
from ancilla.backtesting.instruments import Instrument, InstrumentType
from ancilla.models import Trade, Position
from ancilla.utils.logging import BookLogger

class Portfolio:
    """Portfolio class"""

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

        # First calculate stock values
        for ticker, position in self.positions.items():
            if not position.instrument.is_option:
                if market_prices and ticker in market_prices:
                    price = market_prices[ticker]
                else:
                    price = position.entry_price

                if price is not None:
                    value = position.quantity * price
                    total_value += value

        # Then calculate option values
        for ticker, position in self.positions.items():
            if position.instrument.is_option:
                if market_prices and ticker in market_prices:
                    price = market_prices[ticker]
                else:
                    price = position.entry_price

                if price is not None:
                    multiplier = position.instrument.get_multiplier()
                    # For short positions, liability is positive price * multiplier
                    # For long positions, asset is positive price * multiplier
                    value = -position.quantity * price * multiplier  # Note the negative sign!
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
        transaction_costs: float = 0.0
    ) -> bool:
        """Open a new position with accurate cash flow tracking."""
        multiplier = instrument.get_multiplier()
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        cash_impact = 0.0  # Initialize cash impact

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
        self.opening_cash_flows.append(cash_impact)

        # Create the position
        self.positions[ticker] = Position(
            instrument=instrument,
            quantity=quantity,
            entry_price=price,
            entry_date=timestamp
        )

        # Log the transaction
        self.logger.position_open(
            timestamp=timestamp,
            ticker=instrument.ticker,
            quantity=quantity,
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
        transaction_costs: float = 0.0,
        realized_pnl: Optional[float] = None
    ) -> bool:
        """Close a position with accurate PnL tracking."""
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        if ticker not in self.positions:
            self.logger.get_logger().warning(f"No open position found for {ticker}")
            return False

        position = self.positions[ticker]
        quantity = position.quantity
        entry_price = position.entry_price

        # Calculate P&L
        multiplier = instrument.get_multiplier()
        if instrument.is_option:
            # For options, P&L depends on whether it's a call or put and if it's long or short
            if quantity < 0:
                # Short option: P&L = (entry_price - exit_price) * quantity * multiplier - transaction_costs
                pnl = (entry_price - price) * quantity * multiplier - transaction_costs
            else:
                # Long option: P&L = (price - entry_price) * quantity * multiplier - transaction_costs
                pnl = (price - entry_price) * quantity * multiplier - transaction_costs
        else:
            # For stocks, P&L = (price - entry_price) * quantity - transaction_costs
            pnl = (price - entry_price) * quantity - transaction_costs

        # Update cash
        if instrument.is_option:
            # For options, we want to pay when closing shorts and receive when closing longs
            if quantity < 0:  # short position
                self.cash -= (price * abs(quantity) * multiplier) - transaction_costs
            else:  # long position
                self.cash += (price * quantity * multiplier) - transaction_costs
        else:
            # For stocks, simple multiplication works because quantity already has the right sign
            self.cash += (price * quantity) - transaction_costs

        # Create Trade object
        closing_trade = Trade(
            instrument=instrument,
            entry_time=position.entry_date,
            exit_time=timestamp,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            transaction_costs=transaction_costs,
            realized_pnl=pnl
        )
        self.trades.append(closing_trade)

        # Remove position
        del self.positions[ticker]

        # Log the closure
        self.logger.position_close(
            timestamp=timestamp,
            ticker=ticker,
            quantity=quantity,
            price=price,
            position_type='option' if instrument.is_option else 'stock',
            capital=self.cash
        )
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

    def handle_option_assignment(
        self,
        option: Instrument,
        timestamp: datetime,
        transaction_costs: float = 0.0
    ) -> bool:
        """Handle assignment of an option position."""
        if not option.is_option:
            self.logger.warning(f"Cannot handle assignment for non-option instrument {option.ticker}")
            return False

        option_ticker = option.format_option_ticker()
        if option_ticker not in self.positions:
            self.logger.warning(f"No position found for {option_ticker}")
            return False

        option_position = self.positions[option_ticker]

        # For covered calls:
        if (option.instrument_type == InstrumentType.CALL_OPTION
            and option_position.quantity < 0
            and option.underlying_ticker in self.positions):

            # Get the stock position
            stock_position = self.positions[option.underlying_ticker]
            assignment_quantity = abs(option_position.quantity) * 100

            if stock_position.quantity < assignment_quantity:
                self.logger.warning(f"Insufficient shares for covered call assignment")
                return False

            # Calculate P&L including both stock and option components
            stock_proceeds = option.strike * assignment_quantity
            stock_cost_basis = stock_position.entry_price * assignment_quantity
            option_premium = abs(option_position.quantity) * option_position.entry_price * option.get_multiplier()

            total_pnl = (stock_proceeds - stock_cost_basis) + option_premium - transaction_costs

            # Close both positions
            self.close_position(
                instrument=stock_position.instrument,
                price=option.strike,  # Stock gets called away at strike
                timestamp=timestamp,
                transaction_costs=transaction_costs,
                realized_pnl=total_pnl
            )

            # Close the option position
            self.close_position(
                instrument=option,
                price=0,  # Option expires worthless due to assignment
                timestamp=timestamp,
                transaction_costs=0,  # Already included above
                realized_pnl=0  # P&L handled in stock position
            )

            return True

        return False

    def log_position_status(self):
        """Add this to Portfolio class to help debug position states"""
        self.logger.get_logger().info("Current Portfolio Positions:")
        for ticker, position in self.positions.items():
            position_type = 'option' if position.instrument.is_option else 'stock'
            self.logger.get_logger().info(f"{ticker}: {position.quantity} units @ {position.entry_price} ({position_type})")
