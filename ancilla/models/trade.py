from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

from ancilla.models import Instrument


@dataclass
class Trade:
    """Represents a completed trade."""

    instrument: "Instrument"
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    transaction_costs: float = 0.0  # Combined commission and slippage
    assignment: bool = False
    exercised: bool = False
    realized_pnl: Optional[float] = None

    @property
    def duration_hours(self) -> float:
        """Calculate trade duration in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600

    @property
    def pnl(self) -> float:
        """Calculate net P&L including transaction costs."""
        if self.realized_pnl is not None:
            return self.realized_pnl
        multiplier = 100 if self.instrument.is_option else 1
        if self.quantity < 0 and self.instrument.is_option:
            # Short option
            gross_pnl = (
                (self.entry_price - self.exit_price) * abs(self.quantity) * multiplier
            )
        else:
            # Long option or stock
            gross_pnl = (
                (self.exit_price - self.entry_price) * self.quantity * multiplier
            )
        return gross_pnl - self.transaction_costs

    @property
    def return_pct(self) -> float:
        """Calculate percentage return on trade."""
        multiplier = 100 if self.instrument.is_option else 1
        cost_basis = abs(self.quantity * self.entry_price * multiplier)
        return (self.pnl / cost_basis * 100) if cost_basis > 0 else 0

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trade metrics."""
        return {
            "ticker": self.instrument.ticker,
            "type": "option" if self.instrument.is_option else "stock",
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "duration_hours": self.duration_hours,
            "transaction_costs": self.transaction_costs,
            "option_type": (
                self.instrument.instrument_type if self.instrument.is_option else None
            ),
            "option_ticker": (
                self.instrument.format_option_ticker()
                if self.instrument.is_option
                else None
            ),
            "strike": self.instrument.strike if self.instrument.is_option else None,
            "expiration": (
                self.instrument.expiration if self.instrument.is_option else None
            ),
            "assignment": self.assignment,
            "exercised": self.exercised,
        }
