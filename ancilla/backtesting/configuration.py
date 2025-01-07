# ancilla/backtesting/simulation.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import copy


@dataclass
class CommissionConfig:
    """Configuration for commission models."""

    min_commission: float = 1.0  # Minimum commission per trade
    per_share: float = 0.005  # Per share commission
    per_contract: float = 0.65  # Per option contract
    percentage: float = 0.0  # Percentage of trade value


@dataclass
class SlippageConfig:
    """Configuration for slippage models."""

    base_points: float = 1.0  # Base slippage in basis points
    vol_impact: float = 0.1  # Volume impact factor
    spread_factor: float = 0.5  # Fraction of spread to cross
    market_impact: float = 0.1  # Price impact per 1% of ADV


@dataclass
class ExecutionDetails:
    """Container for execution-related calculations to avoid redundant computation."""

    execution_price: float
    slippage: float
    commission: float
    price_impact: float
    fill_probability: float
    participation_rate: float
    total_transaction_costs: float
    adjusted_quantity: int


class Broker:
    """Handles realistic broker simulation including slippage and commissions."""

    def __init__(
        self,
        commission_config: Optional[CommissionConfig] = None,
        slippage_config: Optional[SlippageConfig] = None,
        deterministic_fill: bool = False,
    ):
        self.commission_config = commission_config or CommissionConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.deterministic_fill = deterministic_fill
        self._market_state = {}

    def calculate_execution_details(
        self, ticker, base_price, quantity, market_data, asset_type="stock"
    ):
        # Use 'vwap' for a more realistic base price if available
        base_price = market_data.get("vwap", market_data.get("close", 0))
        high = market_data.get("high", base_price)
        low = market_data.get("low", base_price)
        volume = market_data.get("volume", 1)
        direction = 1 if quantity > 0 else -1

        # Participation rate
        participation_rate = abs(quantity) / max(volume, 1)

        # Slippage adjustments
        relative_spread = (high - low) / base_price if high > low else 0.001
        base_slippage = self.slippage_config.base_points / 10000
        spread_slippage = relative_spread * self.slippage_config.spread_factor
        scaled_market_impact = self._scale_market_impact(quantity, volume)
        volume_impact = (participation_rate**0.5) * scaled_market_impact

        total_slippage = base_slippage + spread_slippage + volume_impact
        # Cap total slippage to prevent unrealistic jumps
        MAX_SLIPPAGE = 0.01  # 1%
        total_slippage = min(total_slippage, MAX_SLIPPAGE)
        execution_price = base_price * (1 + direction * total_slippage)
        slippage = execution_price - base_price

        # Commission
        commission = self.calculate_commission(execution_price, quantity, asset_type)

        # Fill Probability
        if self.deterministic_fill:
            fill_probability = 1.0
        else:
            if participation_rate < 0.01:
                fill_probability = 0.99
            elif participation_rate < 0.05:
                fill_probability = 0.95
            else:
                fill_probability = 0.90

        # Return details
        return ExecutionDetails(
            execution_price=execution_price,
            slippage=slippage,
            commission=commission,
            price_impact=volume_impact * direction,
            fill_probability=fill_probability,
            participation_rate=participation_rate,
            total_transaction_costs=commission + abs(slippage),
            adjusted_quantity=quantity,
        )

    def _scale_market_impact(self, quantity, avg_volume):
        ratio = abs(quantity) / max(avg_volume, 1)
        # Continuous scaling factor based on square root law
        scale = 1 / (1 + (ratio**0.5))
        # Clamp scale factor
        scale = max(0.1, min(scale, 1.0))
        return self.slippage_config.market_impact * scale

    def calculate_commission(
        self, price: float, quantity: int, asset_type: str = "stock"
    ) -> float:
        """Calculate trading commission."""
        if asset_type == "stock":
            commission = max(
                self.commission_config.min_commission,
                abs(quantity) * self.commission_config.per_share,
            )
        else:  # option
            commission = max(
                self.commission_config.min_commission,
                abs(quantity) * self.commission_config.per_contract,
            )

        # Add percentage-based commission
        if self.commission_config.percentage > 0:
            commission += abs(price * quantity) * self.commission_config.percentage

        return commission

    def estimate_market_hours_fill_probability(
        self,
        price: float,
        quantity: int,
        market_data: Dict[str, Any],
        volume: int,
        asset_type: str = "stock",
    ) -> float:
        """Estimate probability of fill during market hours."""
        if self.deterministic_fill:
            return 1.0
        if asset_type == "stock":
            # Use price relative to day's range
            high = market_data.get("high", price)
            low = market_data.get("low", price)
            if high == low:
                return 1.0

            # Normalize price within [0, 1]
            normalized_price = (price - low) / (high - low)
            normalized_price = max(
                0.0, min(1.0, normalized_price)
            )  # Clamp between 0 and 1

            # Calculate volume impact on probability
            volume_factor = min(
                1.0, volume / abs(quantity) if quantity != 0 and volume != 0 else 1.0
            )

            if quantity > 0:  # Buy order
                # Higher probability for lower prices and higher volume
                prob = (0.5 + 0.5 * (1 - normalized_price)) * volume_factor
            else:  # Sell order
                # Higher probability for higher prices and higher volume
                prob = (0.5 + 0.5 * normalized_price) * volume_factor

            return prob
        else:
            # Options are generally harder to fill
            return (
                0.85 if abs(quantity) < 10 else 0.70
            )  # Lower probability for larger option orders
