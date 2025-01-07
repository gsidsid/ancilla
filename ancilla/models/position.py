from dataclasses import dataclass
from datetime import datetime
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ancilla.models import Instrument


@dataclass
class Position:
    """Represents an open position."""

    instrument: "Instrument"
    quantity: int
    entry_price: float
    entry_date: datetime
    entry_transaction_costs: float = 0.0
    assignment: bool = False
    exercised: bool = False

    @property
    def cost_basis(self) -> float:
        """Calculate total cost basis."""
        multiplier = 100 if self.instrument.is_option else 1
        return abs(self.quantity * self.entry_price * multiplier)

    @property
    def notional_value(self) -> float:
        """Calculate notional value of the position."""
        return self.cost_basis

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    def get_market_value(self, price: float) -> float:
        """Calculate current market value of the position."""
        if self.instrument.is_option:
            # For options, the value is negative for short positions (representing liability)
            # and positive for long positions (representing asset)
            multiplier = self.instrument.get_multiplier()
            contract_value = price * abs(self.quantity) * multiplier
            return -contract_value if self.quantity < 0 else contract_value
        else:
            # For stocks, simple multiplication
            return self.quantity * price

    def get_unrealized_pnl(self, current_price: float) -> Dict[str, float]:
        """Calculate unrealized P&L."""
        multiplier = 100 if self.instrument.is_option else 1
        current_value = self.quantity * current_price * multiplier
        position_cost = self.quantity * self.entry_price * multiplier
        gross_pnl = current_value - position_cost
        return {
            "gross_pnl": gross_pnl,
            "net_pnl": gross_pnl,  # No transaction costs on unrealized P&L
        }
