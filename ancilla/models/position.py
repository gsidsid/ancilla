from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
from ancilla.models import OptionData

@dataclass
class Position:
    """Represents an open position with transaction costs."""
    ticker: str
    quantity: int
    entry_price: float
    entry_date: datetime
    position_type: str  # 'option' or 'stock'
    commission: float = 0.0
    slippage: float = 0.0
    margin_requirement: float = 0.0  # For future margin tracking

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis including transaction costs."""
        multiplier = 100 if self.position_type == 'option' else 1
        position_cost = abs(self.quantity * self.entry_price * multiplier)
        return position_cost + self.commission + self.slippage

    @property
    def notional_value(self) -> float:
        """Calculate notional value of the position."""
        multiplier = 100 if self.position_type == 'option' else 1
        return abs(self.quantity * self.entry_price * multiplier)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    def get_market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        multiplier = 100 if self.position_type == 'option' else 1
        return self.quantity * current_price * multiplier

    def get_unrealized_pnl(self, current_price: float) -> Dict[str, float]:
        """Calculate unrealized P&L with cost breakdown."""
        multiplier = 100 if self.position_type == 'option' else 1
        current_value = self.quantity * current_price * multiplier
        position_cost = self.quantity * self.entry_price * multiplier

        gross_pnl = current_value - position_cost
        transaction_costs = self.commission + self.slippage
        net_pnl = gross_pnl - transaction_costs

        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'transaction_costs': transaction_costs
        }
