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
        transaction_costs: float = 0.0,
        allow_naked_calls: bool = False
    ) -> bool:
        """Open a new position with accurate cash flow tracking."""
        if (instrument.is_option
            and instrument.instrument_type == InstrumentType.CALL_OPTION
            and quantity < 0  # Short call
            and not instrument.naked):  # Only check if it's not a naked call

            required_shares = abs(quantity) * instrument.get_multiplier()
            if (instrument.underlying_ticker not in self.positions
                or self.positions[instrument.underlying_ticker].quantity < required_shares):
                self.logger.get_logger().warning(
                    f"Insufficient shares for covered call: need {required_shares}, "
                    f"have {self.positions.get(instrument.underlying_ticker, Position(None, 0, 0, timestamp)).quantity}"
                )
                return False

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
            entry_date=timestamp,
            entry_transaction_costs=transaction_costs
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
        if instrument.is_option:
            if quantity < 0:
                # Closing a short option position (buying back)
                pnl = (position.entry_price - price) * abs(quantity) * instrument.get_multiplier()
            else:
                # Closing a long option position (selling)
                pnl = (price - position.entry_price) * quantity * instrument.get_multiplier()
        else:
            # For stocks, P&L = (price - entry_price) * quantity
            pnl = (price - entry_price) * quantity

        # Update cash
        if instrument.is_option:
            if quantity < 0:
                # Closing a short option: pay to buy back
                cash_impact = -(price * abs(quantity) * instrument.get_multiplier()) - transaction_costs
            else:
                # Closing a long option: receive from selling
                cash_impact = (price * quantity * instrument.get_multiplier()) - transaction_costs
        else:
            # For stocks
            cash_impact = (price * quantity) - transaction_costs

        self.cash += cash_impact

        total_transaction_costs = position.entry_transaction_costs + transaction_costs

        # Calculate realized PnL
        realized_pnl = pnl - total_transaction_costs

        # Create Trade object
        closing_trade = Trade(
            instrument=instrument,
            entry_time=position.entry_date,
            exit_time=timestamp,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            transaction_costs=total_transaction_costs,
            realized_pnl=realized_pnl
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

    def log_position_status(self):
        """Add this to Portfolio class to help debug position states"""
        self.logger.get_logger().info("Current Portfolio Positions:")
        for ticker, position in self.positions.items():
            position_type = 'option' if position.instrument.is_option else 'stock'
            self.logger.get_logger().info(f"{ticker}: {position.quantity} units @ {position.entry_price} ({position_type})")
