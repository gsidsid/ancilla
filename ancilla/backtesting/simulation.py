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

class MarketSimulator:
    """Handles realistic market simulation including slippage and commissions."""

    def __init__(
        self,
        commission_config: Optional[CommissionConfig] = None,
        slippage_config: Optional[SlippageConfig] = None
    ):
        self.commission_config = commission_config or CommissionConfig()
        self.slippage_config = slippage_config or SlippageConfig()
        self._market_state = {}  # Cache for market state data

    def calculate_execution_price(
        self,
        ticker: str,
        price: float,
        quantity: int,
        market_data: Dict[str, Any],
        direction: int,  # 1 for buy, -1 for sell
        asset_type: str = 'stock'
    ) -> float:
        """Calculate realistic execution price with slippage."""
        # Get market data
        volume = market_data.get('volume', 0)
        high = market_data.get('high', price)
        low = market_data.get('low', price)
        # vwap = market_data.get('vwap', price)

        # Estimate spread
        spread = (high - low) / price if high > low else 0.001

        # Calculate volume-based slippage
        participation_rate = abs(quantity) / volume if volume > 0 else 0
        vol_slippage = self.slippage_config.vol_impact * participation_rate

        # Calculate spread-based slippage
        spread_slippage = spread * self.slippage_config.spread_factor

        # Calculate market impact
        market_impact = (
            self.slippage_config.market_impact *
            (abs(quantity) / volume) if volume > 0 else 0
        )

        # Combine slippage components
        total_slippage = (
            self.slippage_config.base_points / 10000 +  # Convert bps to decimal
            vol_slippage +
            spread_slippage +
            market_impact
        )

        # Apply direction-specific slippage
        execution_price = price * (1 + direction * total_slippage)

        # For options, ensure price respects minimum tick size
        if asset_type == 'option':
            tick_size = 0.05 if price >= 3.0 else 0.01
            execution_price = round(execution_price / tick_size) * tick_size

        return execution_price

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

    def adjust_for_liquidity(
        self,
        quantity: int,
        market_data: Dict[str, Any],
        asset_type: str = 'stock'
    ) -> int:
        """Adjust order size based on liquidity."""
        volume = market_data.get('volume', 0)

        if asset_type == 'stock':
            # Limit to 10% of daily volume by default
            max_quantity = int(volume * 0.10)
            return min(abs(quantity), max_quantity) * (1 if quantity > 0 else -1)
        else:
            # Options are typically less liquid
            open_interest = market_data.get('open_interest', 0)
            max_quantity = int(open_interest * 0.05)  # 5% of open interest
            return min(abs(quantity), max_quantity) * (1 if quantity > 0 else -1)

    def calculate_overnight_gap_risk(
        self,
        current_price: float,
        avg_true_range: float
    ) -> float:
        """Estimate potential overnight gap risk."""
        # Use 2x ATR as estimate for potential gap
        return current_price * (avg_true_range * 2)

    def calculate_price_impact(
        self,
        base_price: float,
        quantity: int,
        daily_volume: float,
        liquidity_score: float
    ) -> float:
        """
        Calculate price impact of order based on size and liquidity.
        Returns impact as a percentage of price.

        Args:
            base_price: Current market price
            quantity: Order quantity (positive for buy, negative for sell)
            daily_volume: Daily trading volume
            liquidity_score: Market liquidity score (0-1)

        Returns:
            Price impact as a decimal (e.g., 0.001 for 0.1% impact)
        """
        # Calculate participation rate
        participation_rate = abs(quantity) / daily_volume if daily_volume > 0 else 1

        # Base impact from order size
        volume_impact = (participation_rate ** 0.5) * self.slippage_config.market_impact

        # Adjust for liquidity - less liquid markets have higher impact
        liquidity_adjustment = (1.5 - liquidity_score)  # ranges from 0.5 to 1.5

        # Apply direction
        direction = 1 if quantity > 0 else -1

        # Combine components
        total_impact = (
            volume_impact * liquidity_adjustment +
            self.slippage_config.base_points / 10000  # Base impact in decimal
        ) * direction

        return total_impact
