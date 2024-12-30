# ancilla/backtesting/strategy.py
from datetime import datetime
from typing import Dict, Any

from ancilla.providers import PolygonDataProvider
from ancilla.backtesting.portfolio import Portfolio
from ancilla.utils.logging import StrategyLogger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ancilla.backtesting.engine import BacktestEngine

class Strategy:
    """Base class for implementing trading strategies."""

    def __init__(self, data_provider: PolygonDataProvider, name: str = "Untitled Strategy"):
        self.name = name
        self.data_provider = data_provider
        self.portfolio: Portfolio;
        self.engine: BacktestEngine;
        self.logger = StrategyLogger(name).get_logger()
        self.logger.info("Starting strategy: %s", name)

    def initialize(self, portfolio: Portfolio, engine: "BacktestEngine") -> None:
        """Initialize the strategy with a portfolio."""
        self.portfolio = portfolio
        self.engine = engine

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """
        Process market data and execute strategy logic.

        Args:
            timestamp: Current timestamp
            market_data: Dictionary of market data by ticker

        Strategy implementations should override this method to:
        1. Fetch any additional data needed (options chains, etc.)
        2. Process market data
        3. Execute trades via self.portfolio
        """
        raise NotImplementedError("Implement on_data in subclass")
