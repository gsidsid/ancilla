# ancilla/backtesting/instruments.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum


class InstrumentType(Enum):
    """Enumeration of tradeable instrument types."""

    STOCK = "stock"
    CALL_OPTION = "call_option"
    PUT_OPTION = "put_option"

    def __str__(self):
        return self.value


@dataclass
class Instrument:
    """Base class for all tradeable instruments."""

    ticker: str
    instrument_type: InstrumentType

    def get_multiplier(self) -> float:
        """Get the contract multiplier based on instrument type."""
        return 100.0 if self.is_option else 1.0

    def format_option_ticker(self) -> str:
        """Format the option ticker for data providers."""
        raise NotImplementedError

    @property
    def underlying_ticker(self) -> str:
        """Get the underlying ticker symbol."""
        return self.ticker  # For stocks, just return the ticker

    @property
    def is_option(self) -> bool:
        """Determine if the instrument is an option."""
        return isinstance(self, Option)


@dataclass
class Stock(Instrument):
    """Represents a stock instrument."""

    def __init__(self, ticker: str):
        super().__init__(ticker=ticker, instrument_type=InstrumentType.STOCK)


@dataclass
class Option(Instrument):
    """Represents an option instrument."""

    strike: float
    expiration: datetime
    naked: bool = False

    def __init__(
        self,
        ticker: str,
        strike: float,
        expiration: datetime,
        option_type: Optional[str] = None,
        instrument_type: Optional[InstrumentType] = InstrumentType.CALL_OPTION,
        naked: bool = False,
    ):
        if option_type:
            instrument_type = self._parse_option_type(option_type)
        super().__init__(
            ticker=ticker, instrument_type=instrument_type or InstrumentType.CALL_OPTION
        )
        self.strike = strike
        self.expiration = expiration
        self.naked = naked

    @staticmethod
    def _parse_option_type(option_type: str) -> InstrumentType:
        """Parse the option type string to InstrumentType enum."""
        option_type = option_type.lower()
        if option_type == "call":
            return InstrumentType.CALL_OPTION
        elif option_type == "put":
            return InstrumentType.PUT_OPTION
        else:
            raise ValueError(f"Invalid option type: {option_type}")

    def format_option_ticker(self) -> str:
        """
        Format the option ticker for data providers.

        Example format: O:TSLA230113C00015000
        """
        exp_str = self.expiration.strftime("%y%m%d")
        strike_int = int(self.strike * 1000)  # Convert strike to integer points
        strike_str = f"{strike_int:08d}"  # Zero-pad to 8 digits
        opt_type = "C" if self.instrument_type == InstrumentType.CALL_OPTION else "P"
        return f"O:{self.ticker}{exp_str}{opt_type}{strike_str}"

    @classmethod
    def from_option_ticker(cls, option_ticker: str) -> "Option":
        """
        Create an Option instance from a formatted option ticker.

        Expected format: O:TSLA230113C00015000
        """
        try:
            prefix, details = option_ticker.split(":")
            if prefix != "O":
                raise ValueError("Option ticker must start with 'O:' prefix.")

            # Extract ticker by filtering out non-alphabetic characters
            ticker = ""
            i = 0
            while i < len(details) and details[i].isalpha():
                ticker += details[i]
                i += 1

            # Extract expiration date (next 6 characters after ticker)
            date_str = details[i : i + 6]
            expiration = datetime.strptime(f"20{date_str}", "%Y%m%d")

            # Extract option type (1 character)
            option_type_char = details[i + 6]
            option_type = (
                InstrumentType.CALL_OPTION
                if option_type_char == "C"
                else InstrumentType.PUT_OPTION
            )

            # Extract strike price (remaining characters)
            strike_str = details[i + 7 :]
            strike = float(strike_str) / 1000.0

            return cls(
                ticker=ticker,
                strike=strike,
                expiration=expiration,
                instrument_type=option_type,
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid option ticker format: {option_ticker}") from e

    @property
    def is_option(self) -> bool:
        """Override to confirm that this instrument is an option."""
        return True
