# ancilla/backtesting/portfolio.py
from datetime import datetime
from typing import Dict, Optional, List
from ancilla.models import OptionData, Trade, Position
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
        """
        Get the total value of all open positions.
        If market_prices provided, use those, otherwise use entry prices.
        """
        total_value = 0
        for ticker, position in self.positions.items():
            price = (market_prices.get(ticker) if market_prices
                    else position.entry_price)
            multiplier = 100 if position.position_type == 'option' else 1
            if price is None:
                continue
            total_value += position.quantity * price * multiplier
        return total_value

    def get_total_value(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Get the total value of the portfolio."""
        return self.cash + self.get_position_value(market_prices)

    def open_position(
        self,
        ticker: str,
        quantity: int,
        price: float,
        timestamp: datetime,
        position_type: str = 'stock',
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> bool:
        """Open a new position with logging."""
        # Check if position already exists
        if ticker in self.positions:
            self.logger.get_logger().warning(f"Position already exists for {ticker}")
            return False

        # Calculate total transaction cost including commission and slippage
        multiplier = 100 if position_type == 'option' else 1
        position_cost = quantity * price * multiplier
        total_cost = position_cost + commission + slippage

        if total_cost > self.cash:
            self.logger.get_logger().warning(
                f"Insufficient cash for {ticker}: "
                f"need ${total_cost:,.2f} (position: ${position_cost:,.2f}, "
                f"commission: ${commission:,.2f}, slippage: ${slippage:,.2f}), "
                f"have ${self.cash:,.2f}"
            )
            return False

        # Create the position
        self.positions[ticker] = Position(
            ticker=ticker,
            quantity=quantity,
            entry_price=price,
            entry_date=timestamp,
            position_type=position_type,
            commission=commission,
            slippage=slippage
        )

        # Deduct total cost from cash
        self.cash -= total_cost

        # Log the transaction with full cost breakdown
        self.logger.position_open(
            timestamp=timestamp,
            ticker=ticker,
            quantity=quantity,
            price=price,
            position_type=position_type,
            capital=self.cash
        )

        # Update capital state
        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(),
            self.get_total_value()
        )

        return True

    def close_position(
        self,
        ticker: str,
        price: float,
        timestamp: datetime,
        commission: float = 0.0,
        slippage: float = 0.0
    ) -> Optional[Trade]:
        """Close a position with logging."""
        if ticker not in self.positions:
            self.logger.get_logger().warning(f"No position found for {ticker}")
            return None

        position = self.positions[ticker]
        multiplier = 100 if position.position_type == 'option' else 1

        # Calculate gross proceeds and costs
        gross_proceeds = position.quantity * price * multiplier
        total_costs = commission + slippage
        net_proceeds = gross_proceeds - total_costs

        # Calculate complete P&L including entry and exit costs
        entry_costs = position.commission + position.slippage
        exit_costs = commission + slippage
        total_costs = entry_costs + exit_costs

        gross_pnl = gross_proceeds - (position.quantity * position.entry_price * multiplier)
        net_pnl = gross_pnl - total_costs

        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            pnl=net_pnl,
            position_type=position.position_type,
            metadata={
                'entry_commission': position.commission,
                'entry_slippage': position.slippage,
                'exit_commission': commission,
                'exit_slippage': slippage,
                'total_costs': total_costs,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl
            }
        )

        self.trades.append(trade)
        self.cash += net_proceeds
        del self.positions[ticker]

        # Log the close with full cost breakdown
        self.logger.position_close(
            timestamp=timestamp,
            ticker=ticker,
            quantity=position.quantity,
            price=price,
            position_type=position.position_type,
            capital=self.cash
        )

        self.logger.trade_complete(timestamp, trade)

        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(),
            self.get_total_value()
        )

        return trade

    def update_equity(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Update equity curve with current market prices."""
        current_equity = self.get_total_value(market_prices)
        self.equity_curve.append((timestamp, current_equity))

        # Log daily capital update
        self.logger.capital_update(
            timestamp,
            self.cash,
            self.get_position_value(market_prices),
            current_equity
        )
