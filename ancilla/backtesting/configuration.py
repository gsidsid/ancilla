# ancilla/backtesting/simulation.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CommissionConfig:
    """Configuration for commission models."""
    min_commission: float = 1.0  # Minimum commission per trade
    per_share: float = 0.005     # Per share commission
    per_contract: float = 0.65   # Per option contract
    percentage: float = 0.0      # Percentage of trade value

@dataclass
class SlippageConfig:
    """Configuration for slippage models."""
    base_points: float = 1.0     # Base slippage in basis points
    vol_impact: float = 0.1      # Volume impact factor
    spread_factor: float = 0.5   # Fraction of spread to cross
    market_impact: float = 0.1   # Price impact per 1% of ADV

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
        slippage_config: Optional[SlippageConfig] = None
    ):
        self.commission_config = commission_config or CommissionConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self._market_state = {}

    def calculate_execution_details(
        self,
        ticker: str,
        base_price: float,
        quantity: int,
        market_data: Dict[str, Any],
        asset_type: str = 'stock'
    ) -> Optional[ExecutionDetails]:
        """
        Consolidates all execution-related calculations into a single method.
        Returns all execution details to avoid recalculating values.
        """
        print("CALC EXECUTION DETAILS", market_data, ticker)
        # Extract market data once
        volume = market_data.get('volume', 0)
        high = market_data.get('high', base_price)
        low = market_data.get('low', base_price)

        # Calculate direction once
        direction = 1 if quantity > 0 else -1

        # Calculate participation rate once
        participation_rate = abs(quantity) / volume if volume > 0 else 1

        # Calculate adjusted quantity
        if participation_rate > 0.1:  # Limit to 10% of daily volume
            adjusted_quantity = int(0.1 * volume) * direction
        else:
            adjusted_quantity = quantity

        if adjusted_quantity == 0:
            return None

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(market_data)

        # Calculate base impact components once
        spread = (high - low) / base_price if high > low else 0.001
        volume_impact = (participation_rate ** 0.5) * self.slippage_config.market_impact
        spread_slippage = spread * self.slippage_config.spread_factor

        # Calculate price impact
        liquidity_adjustment = (1.5 - liquidity_score)
        price_impact = (
            volume_impact * liquidity_adjustment +
            self.slippage_config.base_points / 10000
        ) * direction

        # Calculate total slippage
        total_slippage = (
            self.slippage_config.base_points / 10000 +
            volume_impact +
            spread_slippage
        )

        # Calculate execution price
        execution_price = base_price * (1 + direction * total_slippage)

        # For options, ensure price respects minimum tick size
        if asset_type == 'option':
            tick_size = 0.05 if base_price >= 3.0 else 0.01
            execution_price = round(execution_price / tick_size) * tick_size

        slippage = execution_price - base_price

        # Calculate commission
        commission = self.calculate_commission(
            price=execution_price,
            quantity=adjusted_quantity,
            asset_type=asset_type
        )

        # Calculate fill probability
        fill_probability = self.estimate_market_hours_fill_probability(
            price=execution_price,
            quantity=adjusted_quantity,
            market_data=market_data,
            volume=volume,
            asset_type=asset_type
        )

        # Calculate total costs
        total_transaction_costs = commission + abs(slippage)

        return ExecutionDetails(
            execution_price=execution_price,
            slippage=slippage,
            commission=commission,
            price_impact=price_impact,
            fill_probability=fill_probability,
            participation_rate=participation_rate,
            total_transaction_costs=total_transaction_costs,
            adjusted_quantity=adjusted_quantity
        )


    def _calculate_liquidity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity score based on market data."""
        volume = market_data.get('volume', 0)
        avg_volume = market_data.get('avg_volume', volume)
        spread = market_data.get('spread', 0.01)

        volume_score = min(1.0, volume / avg_volume if avg_volume > 0 else 0)
        spread_score = min(1.0, 1 - (spread * 10))  # Penalize wide spreads

        return (volume_score + spread_score) / 2

    def calculate_commission(
        self,
        price: float,
        quantity: int,
        asset_type: str = 'stock'
    ) -> float:
        """Calculate trading commission."""
        if asset_type == 'stock':
            commission = max(
                self.commission_config.min_commission,
                abs(quantity) * self.commission_config.per_share
            )
        else:  # option
            commission = max(
                self.commission_config.min_commission,
                abs(quantity) * self.commission_config.per_contract
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
        asset_type: str = 'stock'
    ) -> float:
        """Estimate probability of fill during market hours."""
        if asset_type == 'stock':
            # Use price relative to day's range
            high = market_data.get('high', price)
            low = market_data.get('low', price)
            if high == low:
                return 1.0

            # Normalize price within [0, 1]
            normalized_price = (price - low) / (high - low)
            normalized_price = max(0.0, min(1.0, normalized_price))  # Clamp between 0 and 1

            # Calculate volume impact on probability
            volume_factor = min(1.0, volume / abs(quantity) if quantity != 0 and volume != 0 else 1.0)

            if quantity > 0:  # Buy order
                # Higher probability for lower prices and higher volume
                prob = (0.5 + 0.5 * (1 - normalized_price)) * volume_factor
            else:  # Sell order
                # Higher probability for higher prices and higher volume
                prob = (0.5 + 0.5 * normalized_price) * volume_factor

            return prob
        else:
            # Options are generally harder to fill
            return 0.85 if abs(quantity) < 10 else 0.70  # Lower probability for larger option orders
