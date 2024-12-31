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
            if market_prices and ticker in market_prices:
                price = market_prices[ticker]
            else:
                price = position.entry_price

            if price is None:
                continue

            if position.instrument.is_option:
                multiplier = position.instrument.get_multiplier()
                value = position.quantity * price * multiplier
                total_value += value
            else:
                value = position.quantity * price
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
        """Open a new position with logging."""
        # Calculate position impact
        multiplier = instrument.get_multiplier()
        position_value = abs(quantity) * price * multiplier

        # Check if this is part of a covered call
        if instrument.is_option:
            is_covered_call = (
                instrument.instrument_type == InstrumentType.CALL_OPTION
                and quantity < 0
                and instrument.underlying_ticker in self.positions
                and self.positions[instrument.underlying_ticker].quantity > 0
            )

            if is_covered_call:
                # For covered calls, we only want to account for the premium in cash
                cash_impact = position_value - transaction_costs
            else:
                # For other options, normal buy/sell logic
                if quantity > 0:
                    cash_impact = -(position_value + transaction_costs)
                else:
                    cash_impact = position_value - transaction_costs

        # For buying (quantity > 0), we need sufficient cash for position + costs
        # For selling (quantity < 0), we receive premium but need cash for costs
        if quantity > 0:
            required_cash = position_value + transaction_costs
            if required_cash > self.cash:
                self.logger.get_logger().warning(
                    f"Insufficient cash for {instrument.ticker}: "
                    f"need ${required_cash:,.2f}, have ${self.cash:,.2f}"
                )
                return False
            cash_impact = -required_cash
        else:
            # For selling, just need enough for transaction costs
            if transaction_costs > self.cash:
                self.logger.get_logger().warning(
                    f"Insufficient cash for {instrument.ticker} transaction costs: "
                    f"need ${transaction_costs:,.2f}, have ${self.cash:,.2f}"
                )
                return False
            cash_impact = position_value - transaction_costs  # Receive premium, pay costs

        # Create the position
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        self.positions[ticker] = Position(
            instrument=instrument,
            quantity=quantity,
            entry_price=price,
            entry_date=timestamp
        )

        # Update cash
        self.cash += cash_impact

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

        return True

    def close_position(
        self,
        instrument: Instrument,
        price: float,
        timestamp: datetime,
        transaction_costs: float = 0.0,
        realized_pnl: Optional[float] = None
    ) -> bool:
        """Close a position with logging."""
        ticker = instrument.ticker
        if instrument.is_option:
            ticker = instrument.format_option_ticker()

        if ticker not in self.positions:
            self.logger.get_logger().warning(f"No position found for {ticker}")
            return False

        position = self.positions[ticker]

        # Create trade record
        trade = Trade(
            instrument=instrument,
            entry_time=position.entry_date,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            transaction_costs=transaction_costs,
            realized_pnl=realized_pnl
        )

        self.trades.append(trade)

        # If realized P&L is provided, use it directly
        if realized_pnl is not None:
            self.cash += realized_pnl
        else:
            # Calculate proceeds and P&L
            gross_proceeds = position.get_market_value(price)
            net_proceeds = gross_proceeds - transaction_costs
            self.cash += net_proceeds

        del self.positions[ticker]

        # Log the close
        self.logger.position_close(
            timestamp=timestamp,
            ticker=ticker,
            quantity=position.quantity,
            price=price,
            position_type='option' if instrument.is_option else 'stock',
            capital=self.cash
        )
        self.logger.trade_complete(timestamp, trade)
        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(),
            self.get_total_value()
        )

        return True

    def update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Update equity curve with current market prices."""
        current_equity = self.get_total_value(market_prices)
        self.equity_curve.append((timestamp, current_equity))
