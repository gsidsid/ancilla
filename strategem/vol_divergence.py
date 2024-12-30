# strategies/vol_divergence.py
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from ancilla.models import OptionData
from ancilla.backtesting.engine import Strategy, Portfolio

class VolatilityDivergenceStrategy(Strategy):
    """
    Strategy that trades options based on divergence between implied and historical volatility.

    Logic:
    - When IV is significantly higher than HV: Sell options (capture premium)
    - When IV is significantly lower than HV: Buy options (potentially underpriced)
    - Uses put/call ratio and term structure for additional signals
    """

    def __init__(self, data_provider, lookback_days=30, vol_threshold=0.15,
                 max_positions=5, position_size=0.1, min_days_to_expiry=15,
                 max_days_to_expiry=45):
        """
        Args:
            lookback_days: Days to calculate historical volatility
            vol_threshold: Minimum IV-HV difference to trigger trades
            max_positions: Maximum number of concurrent positions
            position_size: Size of each position as fraction of portfolio
            min_days_to_expiry: Minimum days to expiration for new positions
            max_days_to_expiry: Maximum days to expiration for new positions
        """
        super().__init__(data_provider)
        self.lookback_days = lookback_days
        self.vol_threshold = vol_threshold
        self.max_positions = max_positions
        self.position_size = position_size
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry

        # Cache for historical data
        self.hist_vol_cache = {}

    def initialize(self, portfolio: Portfolio):
        """Initialize strategy and historical data."""
        super().initialize(portfolio)
        self.positions_by_underlying = {}

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """
        Process daily market data updates.
        - Update historical volatility calculations
        - Check existing positions
        - Close positions if necessary
        """
        for ticker, data in market_data.items():
            # Update historical volatility cache
            if ticker not in self.hist_vol_cache:
                self.hist_vol_cache[ticker] = []
            self.hist_vol_cache[ticker].append({
                'timestamp': timestamp,
                'close': data['close'],
                'realized_vol': data.get('realized_vol')
            })

            # Trim cache to lookback period
            cutoff = timestamp - timedelta(days=self.lookback_days)
            self.hist_vol_cache[ticker] = [
                d for d in self.hist_vol_cache[ticker]
                if d['timestamp'] > cutoff
            ]

        # Check existing positions for exit conditions
        self._check_existing_positions(timestamp, market_data)

    def on_option_data(self, timestamp: datetime, options_data: Dict[str, List[OptionData]]) -> None:
        """
        Process options data and execute trades based on volatility divergence.

        Args:
            timestamp: Current timestamp
            options_data: Dictionary of options chains by underlying
        """
        for ticker, chain in options_data.items():
            # Skip if we don't have enough historical data
            if ticker not in self.hist_vol_cache or \
               len(self.hist_vol_cache[ticker]) < self.lookback_days:
                continue

            # Get current historical volatility
            hist_vol = self._calculate_historical_vol(ticker)
            if hist_vol is None:
                continue

            # Get chain-wide IV and other metrics
            chain_metrics = self._analyze_options_chain(chain)
            if not chain_metrics:
                continue

            # Check if we have a trading opportunity
            signal = self._generate_trading_signal(
                hist_vol,
                chain_metrics['avg_iv'],
                chain_metrics
            )

            if signal:
                self._execute_trade(
                    timestamp,
                    ticker,
                    chain,
                    signal,
                    chain_metrics,
                    hist_vol
                )

    def _check_existing_positions(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """Check and manage existing positions."""
        for ticker in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[ticker]

            # Skip if not an options position
            if position.position_type != 'option' or not position.option_data:
                continue

            option_data = position.option_data
            days_to_expiry = (option_data.expiration - timestamp).days

            # Close if near expiration
            if days_to_expiry <= 5:
                # Get current option price (simplified)
                # In practice, you'd want to use an options pricing model here
                if ticker in market_data:
                    self.portfolio.close_position(ticker, market_data[ticker]['close'], timestamp)

    def _calculate_historical_vol(self, ticker: str) -> Optional[float]:
        """Calculate historical volatility from cached data."""
        if len(self.hist_vol_cache[ticker]) < self.lookback_days:
            return None

        prices = pd.DataFrame(self.hist_vol_cache[ticker])
        if 'realized_vol' in prices.columns and not prices['realized_vol'].isna().all():
            return prices['realized_vol'].iloc[-1]

        # Calculate if not provided
        returns = np.log(prices['close'] / prices['close'].shift(1))
        return np.sqrt(252) * returns.std()

    def _analyze_options_chain(self, chain: List[OptionData]) -> Optional[Dict[str, Any]]:
        """Analyze options chain for trading signals."""
        if not chain:
            return None

        # Calculate average IV and other metrics
        calls = [opt for opt in chain if opt.contract_type == 'call']
        puts = [opt for opt in chain if opt.contract_type == 'put']

        metrics = {
            'avg_iv': np.mean([opt.implied_volatility for opt in chain]),
            'call_iv': np.mean([opt.implied_volatility for opt in calls]) if calls else None,
            'put_iv': np.mean([opt.implied_volatility for opt in puts]) if puts else None,
            'put_call_ratio': len(puts) / len(calls) if calls else None,
            'skew': None
        }

        # Calculate volatility skew
        atm_calls = [opt for opt in calls if 0.45 <= abs(opt.delta) <= 0.55]
        atm_puts = [opt for opt in puts if 0.45 <= abs(opt.delta) <= 0.55]
        if atm_calls and atm_puts:
            metrics['skew'] = np.mean([opt.implied_volatility for opt in atm_puts]) / \
                            np.mean([opt.implied_volatility for opt in atm_calls])

        return metrics

    def _generate_trading_signal(self, hist_vol: float, implied_vol: float,
                               chain_metrics: Dict[str, Any]) -> Optional[str]:
        """Generate trading signal based on volatility divergence."""
        # Calculate vol spread
        vol_spread = implied_vol - hist_vol

        # No signal if spread is too small
        if abs(vol_spread) < self.vol_threshold:
            return None

        # Check for extreme skew
        skew = chain_metrics.get('skew')
        if skew:
            if skew > 1.1:  # High put premium
                return 'sell_puts' if vol_spread > 0 else None
            elif skew < 0.9:  # High call premium
                return 'sell_calls' if vol_spread > 0 else None

        # Default signals based on vol spread
        if vol_spread > self.vol_threshold:
            return 'sell'  # IV significantly higher than HV
        elif vol_spread < -self.vol_threshold:
            return 'buy'   # IV significantly lower than HV

        return None

    def _execute_trade(self, timestamp: datetime, ticker: str,
                      chain: List[OptionData], signal: str,
                      chain_metrics: Dict[str, Any], hist_vol: float) -> None:
        """Execute trade based on signal."""
        # Check position limits
        if len(self.portfolio.positions) >= self.max_positions:
            return

        # Filter options by days to expiry
        valid_options = [
            opt for opt in chain
            if self.min_days_to_expiry <= (opt.expiration - timestamp).days <= self.max_days_to_expiry
        ]

        if not valid_options:
            return

        # Select options based on signal
        if signal in ['sell', 'sell_puts']:
            candidates = [opt for opt in valid_options
                        if opt.contract_type == 'put' and 0.3 <= abs(opt.delta) <= 0.4]
        elif signal in ['sell_calls']:
            candidates = [opt for opt in valid_options
                        if opt.contract_type == 'call' and 0.3 <= abs(opt.delta) <= 0.4]
        elif signal == 'buy':
            candidates = [opt for opt in valid_options
                        if 0.45 <= abs(opt.delta) <= 0.55]
        else:
            return

        if not candidates:
            return

        # Select the option with the highest volume
        selected_option = max(candidates, key=lambda x: x.volume if x.volume else 0)

        # Calculate position size
        portfolio_value = self.portfolio.get_total_value()
        max_position_value = portfolio_value * self.position_size

        # For selling options, consider margin requirements (simplified)
        contracts = int(max_position_value / (selected_option.strike * 100))
        if contracts < 1:
            return

        # Execute trade
        if signal.startswith('sell'):
            # Short options position
            self.portfolio.open_position(
                selected_option.ticker,
                -contracts,  # Negative for short
                selected_option.strike,
                timestamp,
                'option',
                selected_option
            )
        else:
            # Long options position
            self.portfolio.open_position(
                selected_option.ticker,
                contracts,
                selected_option.strike,
                timestamp,
                'option',
                selected_option
            )
