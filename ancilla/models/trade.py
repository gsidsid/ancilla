from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any

@dataclass
class Trade:
    """Represents a completed trade with full cost analysis."""
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    position_type: str
    pnl: float  # Net P&L after all costs
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> timedelta:
        """Calculate trade duration."""
        return self.exit_date - self.entry_date

    @property
    def gross_pnl(self) -> float:
        """Get gross P&L before costs."""
        return self.metadata.get('gross_pnl', self.pnl)

    @property
    def transaction_costs(self) -> float:
        """Get total transaction costs."""
        return self.metadata.get('total_costs', 0.0)

    @property
    def return_pct(self) -> float:
        """Calculate percentage return including costs."""
        multiplier = 100 if self.position_type == 'option' else 1
        initial_value = abs(self.quantity * self.entry_price * multiplier)
        if initial_value == 0:
            return 0.0
        return self.pnl / initial_value

    @property
    def cost_analysis(self) -> Dict[str, float]:
        """Get detailed cost breakdown."""
        return {
            'entry_commission': self.metadata.get('entry_commission', 0.0),
            'entry_slippage': self.metadata.get('entry_slippage', 0.0),
            'exit_commission': self.metadata.get('exit_commission', 0.0),
            'exit_slippage': self.metadata.get('exit_slippage', 0.0),
            'total_costs': self.metadata.get('total_costs', 0.0)
        }

    def summary(self) -> Dict[str, Any]:
        """Generate comprehensive trade summary."""
        return {
            'ticker': self.ticker,
            'position_type': self.position_type,
            'quantity': self.quantity,
            'duration_days': self.duration.days,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.pnl,
            'return_pct': self.return_pct,
            'costs': self.cost_analysis,
            'metadata': self.metadata
        }
