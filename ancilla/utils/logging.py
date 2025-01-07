# ancilla/utils/logging.py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from ancilla.models import OptionData


class BaseLogger:
    """Base logger class with common functionality"""

    def __init__(self, name: str, log_dir: str):
        self.name = name
        self.log_dir = log_dir
        self.logger: Optional[logging.Logger] = None
        self._setup_logger()
        self._previous_level = None

    def _setup_logger(self) -> None:
        """Set up logging configuration"""
        # Create logger
        logger_name = f"ancilla.{self.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = Path(f"logs/{self.log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create handlers
        self._add_file_handler()
        self._add_console_handler()

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _add_file_handler(self) -> None:
        """Add file handler with daily rotation"""
        today = datetime.now().strftime("%Y%m%d")
        file_path = f"logs/{self.log_dir}/{self.name}_{today}.log"

        # Create file handler
        file_handler = logging.FileHandler(file_path, mode="a")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        if isinstance(self.logger, logging.Logger):
            self.logger.addHandler(file_handler)

    def _add_console_handler(self) -> None:
        """Add console handler for important messages"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        # Add handler to logger
        if isinstance(self.logger, logging.Logger):
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """Get configured logger"""
        if self.logger is None:
            raise ValueError("Logger not configured")
        return self.logger

    @contextmanager
    def mute(self):
        """Temporarily disable logging for performance-critical sections"""
        if self.logger:
            self._previous_level = self.logger.level
            self.logger.setLevel(logging.CRITICAL + 1)
        try:
            yield
        finally:
            if self.logger and self._previous_level is not None:
                self.logger.setLevel(self._previous_level)
                self._previous_level = None


class MarketDataLogger(BaseLogger):
    """Logger configuration for market data providers."""

    def __init__(self, provider_name: str):
        super().__init__(provider_name, "providers")


class BacktesterLogger(BaseLogger):
    """Logger configuration for trading strategies."""

    def __init__(self):
        super().__init__("backtest", "strategies")


class StrategyLogger(BaseLogger):
    """Logger configuration for trading strategies."""

    def __init__(self, strategy_name: str):
        super().__init__(strategy_name, "strategies")


class VisualizerLogger(BaseLogger):
    """Logger configuration for data visualizers."""

    def __init__(self, visualizer_name: str):
        super().__init__(visualizer_name, "visualizers")


class BookLogger(BaseLogger):
    """Logger for tracking portfolio changes and trades."""

    def __init__(self, name: str = "BookLogger"):
        super().__init__(name, "books")
        self.logger = self.get_logger()

        self.position_fmt = (
            "POSITION | {timestamp} | {ticker:^10} | {action:^6} | "
            "Qty: {quantity:>6} @ {price:<8.2f} | Type: {position_type} | "
            "Capital: ${capital:,.2f}"
        )
        self.trade_fmt = (
            "TRADE    | {timestamp} | {ticker:^10} | {action:^6} | "
            "Qty: {quantity:>6} | Entry: {entry_price:<8.2f} | Exit: {exit_price:<8.2f} | "
            "P&L: ${pnl:,.2f}"
        )
        self.capital_fmt = (
            "CAPITAL  | {timestamp} | Cash: ${cash:,.2f} | "
            "Position Value: ${position_value:,.2f} | "
            "Total Value: ${total_value:,.2f}"
        )
        self.option_fmt = (
            "OPTION   | {timestamp} | {ticker:^10} | Strike: {strike:<8.2f} | "
            "Type: {contract_type:^4} | Exp: {expiration} | "
            "Delta: {delta:>6.3f} | IV: {iv:>6.2%}"
        )

    def position_open(
        self,
        timestamp: datetime,
        ticker: str,
        quantity: int,
        price: float,
        position_type: str,
        capital: float,
    ):
        """Log opening a new position."""
        if self.logger is None:
            raise ValueError("Logger not initialized")
        self.logger.debug(
            self.position_fmt.format(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=ticker,
                action="OPEN",
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=capital,
            )
        )

    def position_close(
        self,
        timestamp: datetime,
        ticker: str,
        quantity: int,
        price: float,
        position_type: str,
        capital: float,
    ):
        """Log closing a position."""
        if self.logger is None:
            raise ValueError("Logger not initialized")

        self.logger.debug(
            self.position_fmt.format(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=ticker,
                action="CLOSE",
                quantity=quantity,
                price=price,
                position_type=position_type,
                capital=capital,
            )
        )

    def trade_complete(self, timestamp: datetime, trade: Any):
        """Log completed trade details."""
        if self.logger is None:
            raise ValueError("Logger not initialized")
        self.logger.debug(
            self.trade_fmt.format(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=trade.instrument.ticker,
                action="COMPLETE",
                quantity=trade.quantity,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
            )
        )

    def capital_update(
        self,
        timestamp: datetime,
        cash: float,
        position_value: float,
        total_value: float,
    ):
        """Log capital changes."""
        if self.logger is None:
            raise ValueError("Logger not initialized")
        self.logger.debug(
            self.capital_fmt.format(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                cash=cash,
                position_value=position_value,
                total_value=total_value,
            )
        )

    def option_data(self, timestamp: datetime, option: "OptionData"):
        """Log option contract details."""
        if self.logger is None:
            raise ValueError("Logger not initialized")
        self.logger.debug(
            self.option_fmt.format(
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                ticker=option.instrument.ticker,  # type: ignore
                strike=option.strike,
                contract_type=option.contract_type,
                expiration=option.expiration.strftime("%Y-%m-%d"),
                delta=option.delta if option.delta else 0,
                iv=option.implied_volatility if option.implied_volatility else 0,
            )
        )
