# ancilla/utils/logging.py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

class MarketDataLogger:
    """
    Logger configuration for market data providers.

    Features:
    - Separate log files for different providers
    - Daily log rotation
    - Different log levels for file and console
    - Structured logging format
    - Temporary muting for performance optimization
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.logger: Optional[logging.Logger] = None
        self._setup_logger()
        self._previous_level = None

    def _setup_logger(self) -> None:
        """Set up logging configuration"""
        # Create logger
        logger_name = f"ancilla.{self.provider_name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs/providers")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create handlers
        self._add_file_handler()
        self._add_console_handler()

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _add_file_handler(self) -> None:
        """Add file handler with daily rotation"""
        today = datetime.now().strftime("%Y%m%d")
        file_path = f"logs/providers/{self.provider_name}_{today}.log"

        # Create file handler
        # Replace existing log file if it exists
        file_handler = logging.FileHandler(file_path, mode="w+")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
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


class StrategyLogger:
    """
    Logger configuration for trading strategies.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.logger: Optional[logging.Logger] = None
        self._setup_logger()
        self._previous_level = None

    def _setup_logger(self) -> None:
        """Set up logging configuration"""
        # Create logger
        logger_name = f"ancilla.{self.strategy_name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        log_dir = Path("logs/strategies")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create handlers
        self._add_file_handler()
        self._add_console_handler()

        # Prevent propagation to root logger
        self.logger.propagate = False

    def _add_file_handler(self) -> None:
        """Add file handler with daily rotation"""
        today = datetime.now().strftime("%Y%m%d")
        file_path = f"logs/strategies/{self.strategy_name}_{today}.log"

        # Create file handler
        file_handler = logging.FileHandler(file_path, mode="w+")
        file_handler.setLevel(logging.DEBUG)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
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
