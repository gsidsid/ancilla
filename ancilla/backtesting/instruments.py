# ancilla/backtesting/instruments.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum

class InstrumentType(Enum):
    """Types of tradeable instruments."""
    STOCK = "stock"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"

@dataclass
class Instrument:
    """Base class for all tradeable instruments."""
    ticker: str
    instrument_type: InstrumentType

    def get_multiplier(self) -> float:
        """Get contract multiplier."""
        return 100.0 if self.is_option else 1.0

    @property
    def underlying_ticker(self) -> str:
        """Get the underlying ticker symbol."""
        return self.ticker  # For stocks, just return the ticker

    @property
    def is_option(self) -> bool:
        """Check if instrument is an option."""
        return self.instrument_type in [InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION]

@dataclass
class Stock(Instrument):
    """Stock instrument."""
    def __init__(self, ticker: str):
        super().__init__(ticker=ticker, instrument_type=InstrumentType.STOCK)

@dataclass
class Option(Instrument):
    """Option instrument."""
    strike: float
    expiration: datetime

    def __init__(self, ticker: str, strike: float, expiration: datetime, **kwargs):
        if kwargs.get('instrument_type') is not None:
            super().__init__(ticker=ticker, instrument_type=kwargs['instrument_type'])
        elif kwargs.get('option_type') is not None:
            option_type = kwargs['option_type']
            super().__init__(ticker=ticker,
                            instrument_type=(InstrumentType.CALL_OPTION
                                        if option_type.lower() == 'call'
                                        else InstrumentType.PUT_OPTION))
        self.strike = strike
        self.expiration = expiration

    @property
    def underlying_ticker(self) -> str:
        """Get the underlying ticker symbol."""
        return self.ticker

    def format_option_ticker(self) -> str:
        """Format option ticker for data provider."""
        exp_str = self.expiration.strftime('%y%m%d')
        strike_int = int(self.strike * 1000)  # Convert strike to integer points
        strike_str = f"{strike_int:08d}"      # Zero-pad to 8 digits
        opt_type = 'C' if self.instrument_type == InstrumentType.CALL_OPTION else 'P'
        return f"O:{self.ticker}{exp_str}{opt_type}{strike_str}"

    @classmethod
    def from_option_ticker(cls, option_ticker: str) -> 'Option':
        """Create Option instance from formatted option ticker."""
        # Parse O:TSLA230113C00015000 format
        parts = option_ticker.split(':')[1]  # Remove O: prefix
        ticker = ''.join(c for c in parts if c.isalpha())  # Extract ticker
        date_str = parts[len(ticker):len(ticker)+6]  # Extract YYMMDD
        option_type = parts[len(ticker)+6]  # Extract C/P
        strike_str = parts[len(ticker)+7:]  # Extract strike

        expiration = datetime.strptime(f"20{date_str}", "%Y%m%d")
        strike = float(strike_str) / 1000.0  # Convert strike to decimal

        return cls(
            ticker=ticker,
            strike=strike,
            expiration=expiration,
            option_type='call' if option_type == 'C' else 'put'
        )

    @property
    def is_option(self) -> bool:
        """Check if instrument is an option."""
        return True
