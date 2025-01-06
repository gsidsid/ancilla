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
        slippage_config: Optional[SlippageConfig] = None,
        deterministic_fill: bool = False
    ):
        self.commission_config = commission_config or CommissionConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self.deterministic_fill = deterministic_fill
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
        Calculates execution details using only OHLCV data.
        Adjusts calculations for options based on price and spread characteristics.
        """
        # Extract market data
        volume = market_data.get('volume', 0)
        high = market_data.get('high', base_price)
        low = market_data.get('low', base_price)

        # For options, adjust volume interpretation
        if asset_type == 'option':
            # Options trade in much lower volume, so scale up the interpretation
            # of what constitutes "good" volume
            volume = max(volume * 50, 100)  # Assume minimum baseline liquidity

            # Estimate option "tier" based on price to adjust expectations
            if base_price <= 0.10:  # Deep OTM options
                volume *= 0.2  # These are typically less liquid
            elif base_price <= 0.50:
                volume *= 0.5
            elif base_price <= 1.0:
                volume *= 0.8
            elif base_price >= 10.0:  # ITM options
                volume *= 0.7  # These are typically less liquid too

        if volume == 0:
            # Use price-based minimum volume assumption
            min_volume = 1000 if asset_type == 'stock' else 50
            volume = min_volume
            # self.logger.warning(f"No volume data for {ticker}, using minimum volume of {min_volume}")

        # Calculate direction
        direction = 1 if quantity > 0 else -1

        # Calculate participation rate with option adjustments
        base_participation = abs(quantity) / volume if volume > 0 else 1
        if asset_type == 'option':
            # Options can typically handle higher participation rates
            participation_rate = base_participation * 0.5  # Scale down impact
        else:
            participation_rate = base_participation

        # Calculate adjusted quantity
        max_participation = 0.15 if asset_type == 'option' else 0.1
        if participation_rate > max_participation:
            adjusted_quantity = int(max_participation * volume) * direction
        else:
            adjusted_quantity = quantity

        if adjusted_quantity == 0:
            return None

        # Calculate spread-based liquidity score
        relative_spread = (high - low) / base_price if high > low else 0.001
        if asset_type == 'option':
            # Options naturally have wider spreads
            relative_spread = max(0.01, min(relative_spread * 0.5, 0.15))
        liquidity_score = 1.0 - min(1.0, relative_spread * 10)

        # Calculate volume-based liquidity component
        if asset_type == 'option':
            min_good_volume = 100  # Baseline for "good" option volume
            volume_score = min(1.0, volume / min_good_volume)
        else:
            min_good_volume = 10000
            volume_score = min(1.0, volume / min_good_volume)

        # Combine liquidity scores
        liquidity_score = (liquidity_score * 0.7 + volume_score * 0.3)

        # Calculate impact components
        spread = relative_spread
        if asset_type == 'option':
            # Options typically have higher spreads and impact
            volume_impact = (participation_rate ** 0.3) * self.slippage_config.market_impact * 1.5
            spread *= 1.2  # Wider spreads for options
        else:
            volume_impact = (participation_rate ** 0.5) * self.slippage_config.market_impact

        spread_slippage = spread * self.slippage_config.spread_factor

        # Calculate price impact
        liquidity_adjustment = (1.5 - liquidity_score)
        if asset_type == 'option':
            # Price moves are typically more pronounced in options
            liquidity_adjustment *= 1.3

        price_impact = (
            volume_impact * liquidity_adjustment +
            self.slippage_config.base_points / 10000
        ) * direction

        # Calculate total slippage
        base_slippage = self.slippage_config.base_points / 10000
        if asset_type == 'option':
            # Options typically have higher base slippage
            base_slippage *= 1.5

        total_slippage = (
            base_slippage +
            volume_impact +
            spread_slippage
        )

        # Calculate execution price
        execution_price = base_price * (1 + direction * total_slippage)

        # Apply tick size rounding for options
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
        if asset_type == 'option':
            # Base probability on price and volume characteristics
            base_prob = min(0.95, liquidity_score)

            # Adjust based on price level
            if base_price <= 0.10:
                price_adj = 0.6  # Deep OTM options are harder to fill
            elif base_price <= 0.50:
                price_adj = 0.8
            elif base_price <= 1.0:
                price_adj = 0.9
            else:
                price_adj = 1.0

            fill_probability = base_prob * price_adj
        else:
            fill_probability = self.estimate_market_hours_fill_probability(
                price=execution_price,
                quantity=adjusted_quantity,
                market_data=market_data,
                volume=volume,
                asset_type=asset_type
            )

        if self.deterministic_fill:
            fill_probability = 1.0

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
        if self.deterministic_fill:
            return 1.0
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
