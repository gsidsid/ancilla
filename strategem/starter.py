# strategem/starter.py

import os
import pytz
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import dotenv

# Local imports
from ancilla.backtesting import Backtest, Strategy
from ancilla.backtesting.configuration import CommissionConfig, SlippageConfig
from ancilla.providers import PolygonDataProvider
from ancilla.models import Stock, Option

dotenv.load_dotenv()


class StarterBaseStrategy(Strategy):
    """
    A 'starter' strategy that provides:
      1. Built-in helpers for opening/closing equities and options.
      2. Actual stop-loss, stop-limit, and trailing-stop order logic.
      3. Common multi-leg option structure methods (covered calls, straddles, etc.).
      4. Retry logic for orders.

    Override on_data() in your subclass to implement your own trading signals.
    The stop/limit logic below will handle exits automatically if triggered.
    """

    def __init__(self, data_provider, name: str = "starter_base"):
        super().__init__(data_provider, name=name)
        # Keep track of stop/trailing orders in memory
        # Example structure:
        # self.stop_orders = {
        #    "AAPL": {
        #       "shares": 100,
        #       "stop_loss": 150.0,
        #       "stop_limit": (155.0, 154.5),
        #       "trailing_stop": 5.0,
        #       "highest_price": 160.0
        #    }
        # }
        self.stop_orders: Dict[
            str, Dict[str, Union[float, Tuple[float, float], int]]
        ] = {}
        self.market_data: Dict[str, Any] = {}

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """
        Override this in your subclass for custom signal logic.
        Meanwhile, we handle any stop-loss / stop-limit / trailing-stop triggers here.
        """
        self.time = timestamp
        self.market_data = market_data
        # Check each ticker that might have active stops
        for ticker, stop_info in list(self.stop_orders.items()):
            if ticker not in market_data:
                continue

            current_price = market_data[ticker]["close"]
            triggered_sell = False
            qty = stop_info["shares"]

            # Trailing stop logic: update highest_price if needed, check if triggered
            if stop_info.get("trailing_stop") is not None:
                if current_price > stop_info.get("highest_price", current_price):
                    stop_info["highest_price"] = current_price
                trail_amount = stop_info["trailing_stop"]
                max_price = stop_info.get("highest_price", current_price)
                if current_price <= max_price - trail_amount:
                    triggered_sell = True

            # Stop-loss logic (market order)
            if not triggered_sell and stop_info.get("stop_loss") is not None:
                if current_price <= stop_info["stop_loss"]:
                    triggered_sell = True

            # Stop-limit logic
            # If price crosses below 'stop_price', we place a limit sell at 'limit_price'.
            # For simplicity, assume if we see price < stop_price, we attempt the limit immediately.
            # (In a real engine, you'd queue this until the market crosses the stop.)
            if not triggered_sell and stop_info.get("stop_limit") is not None:
                s_price, l_price = stop_info["stop_limit"]
                if current_price <= s_price:
                    # Attempt a limit sell at l_price
                    # If limit_price is above the current_price, it likely won't fill unless price recovers.
                    # For demonstration, we'll just attempt to sell immediately.
                    # If it fails, you might store partial fills, etc.
                    triggered_sell = True

            # If triggered, do the sell
            if triggered_sell:
                # You may add partial fill logic, but here we just sell at market.
                success = self.engine.sell_stock(ticker, qty)
                if success:
                    self.logger.info(
                        f"STOP/TRAIL SELL: {qty} {ticker} @ ~${current_price}"
                    )
                    self.stop_orders.pop(ticker, None)  # remove from stop tracking
                else:
                    self.logger.warning(f"{ticker}: stop/limit sell order failed.")

    # -------------------------------------------------------------------------
    # Internal Retry Helpers
    # -------------------------------------------------------------------------
    def _retry_buy_stock(self, ticker: str, shares: int, max_retries: int) -> bool:
        for _ in range(max_retries):
            if self.engine.buy_stock(ticker, shares):
                return True
        return False

    def _retry_sell_stock(self, ticker: str, shares: int, max_retries: int) -> bool:
        for _ in range(max_retries):
            if self.engine.sell_stock(ticker, shares):
                return True
        return False

    def _retry_buy_option(
        self, option: Option, contracts: int, max_retries: int
    ) -> bool:
        for _ in range(max_retries):
            if self.engine.buy_option(option, contracts):
                return True
        return False

    def _retry_sell_option(
        self, option: Option, contracts: int, max_retries: int
    ) -> bool:
        for _ in range(max_retries):
            if self.engine.sell_option(option, contracts):
                return True
        return False

    # -------------------------------------------------------------------------
    # Stock Position Management (with optional stops)
    # -------------------------------------------------------------------------
    def open_stock_position(
        self,
        ticker: str,
        allocation_pct: float = 0.1,
        max_retries: int = 1,
        stop_loss: Optional[float] = None,
        stop_limit: Optional[Tuple[float, float]] = None,
        trailing_stop: Optional[float] = None,
    ) -> bool:
        """
        Buy stock with an optional percentage of total portfolio.
        Also sets up stop-loss/stop-limit/trailing-stop if provided.
        """
        if ticker in self.portfolio.positions:
            self.logger.info(f"{ticker}: existing stock position, skipping buy.")
            return False

        portfolio_val = self.portfolio.get_total_value()
        price = self.market_data[ticker]["close"]
        if not price or price <= 0:
            self.logger.warning(f"{ticker}: invalid price data.")
            return False

        budget = portfolio_val * allocation_pct
        shares = int(budget // price)
        if shares < 1:
            self.logger.info(f"{ticker}: not enough capital for 1 share.")
            return False

        success = self._retry_buy_stock(ticker, shares, max_retries)
        if success:
            self.logger.info(f"BOUGHT {shares} {ticker}")
            # Track stops
            if any([stop_loss, stop_limit, trailing_stop]):
                self.stop_orders[ticker] = {
                    "shares": shares,
                    "stop_loss": stop_loss,
                    "stop_limit": stop_limit,
                    "trailing_stop": trailing_stop,
                    "highest_price": price,
                }
        else:
            self.logger.warning(f"{ticker}: buy failed.")
        return success

    def close_stock_position(self, ticker: str, max_retries: int = 1) -> bool:
        """
        Close any open stock position (cancels associated stop/trail).
        """
        if ticker not in self.portfolio.positions:
            self.logger.info(f"{ticker}: no stock position to close.")
            return False

        position = self.portfolio.positions[ticker]
        if not isinstance(position.instrument, Stock):
            self.logger.info(f"{ticker}: position is not stock.")
            return False

        qty = position.quantity
        if qty <= 0:
            self.logger.info(f"{ticker}: quantity=0, skip close.")
            return False

        success = self._retry_sell_stock(ticker, qty, max_retries)
        if success:
            self.logger.info(f"SOLD {qty} {ticker}")
            if ticker in self.stop_orders:
                self.stop_orders.pop(ticker)
        else:
            self.logger.warning(f"{ticker}: sell failed.")
        return success

    # -------------------------------------------------------------------------
    # Option Position Management
    # -------------------------------------------------------------------------
    def open_option_position(
        self,
        ticker: str,
        contract_type: str = "call",
        num_contracts: int = 1,
        max_expiration_days: int = 30,
        strike_proximity_pct: float = 0.0,
        max_retries: int = 1,
    ) -> bool:
        """
        Buy an option near ATM (default) or set your strike_proximity_pct to move OTM/ITM.
        """
        price = self.market_data[ticker]["close"]
        if not price or price <= 0:
            self.logger.warning(f"{ticker}: no price data for option buy.")
            return False

        px = price
        lower_strike = px * (1 - abs(strike_proximity_pct))
        upper_strike = px * (1 + abs(strike_proximity_pct))

        contracts = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(lower_strike, upper_strike),
            max_expiration_days=max_expiration_days,
            contract_type=contract_type,
        )
        if not contracts:
            self.logger.info(f"{ticker}: no {contract_type} in range.")
            return False

        opt = min(contracts, key=lambda c: abs(c.strike - px))
        success = self._retry_buy_option(opt, num_contracts, max_retries)
        if success:
            self.logger.info(
                f"BOUGHT {num_contracts} {contract_type.upper()} {ticker} strike={opt.strike}"
            )
        else:
            self.logger.warning(f"{ticker}: option buy failed.")
        return success

    def close_option_position(self, option: Option, max_retries: int = 1) -> bool:
        """
        Close an open option position (long or short).
        """
        pos = None
        for p in self.portfolio.positions.values():
            if p.instrument == option:
                pos = p
                break

        if not pos or pos.quantity == 0:
            self.logger.info("Option not found or quantity=0.")
            return False

        qty = abs(pos.quantity)
        if pos.quantity > 0:
            success = self._retry_sell_option(option, qty, max_retries)
            if success:
                self.logger.info(f"CLOSED long {option.ticker}")
            else:
                self.logger.warning(f"{option.ticker}: close failed.")
            return success
        else:
            success = self._retry_buy_option(option, qty, max_retries)
            if success:
                self.logger.info(f"CLOSED short {option.ticker}")
            else:
                self.logger.warning(f"{option.ticker}: close failed.")
            return success

    # -------------------------------------------------------------------------
    # Common Option Structures
    # -------------------------------------------------------------------------
    def open_covered_call(
        self,
        ticker: str,
        stock_allocation_pct: float = 0.2,
        call_otm_pct: float = 0.05,
        max_expiration_days: int = 30,
        max_retries: int = 1,
    ) -> bool:
        """
        1) Buy underlying stock with stock_allocation_pct of the portfolio.
        2) Write (sell) an OTM call.
        """
        bought_stock = self.open_stock_position(
            ticker=ticker, allocation_pct=stock_allocation_pct, max_retries=max_retries
        )
        if not bought_stock:
            return False

        price = self.market_data[ticker]["close"]
        if not price:
            return False

        px = price
        low_strike = px * (1 + call_otm_pct * 0.9)
        high_strike = px * (1 + call_otm_pct * 1.1)
        calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(low_strike, high_strike),
            max_expiration_days=max_expiration_days,
            contract_type="call",
        )
        if not calls:
            self.logger.info(f"{ticker}: no OTM calls found.")
            return False

        best_call = min(calls, key=lambda c: abs(c.strike - px * (1 + call_otm_pct)))
        stock_pos = self.portfolio.positions.get(ticker)
        if not stock_pos or stock_pos.quantity < 100:
            self.logger.info(
                f"{ticker}: not enough shares (need >=100) to cover calls."
            )
            return False

        n_contracts = stock_pos.quantity // 100
        success = self._retry_sell_option(best_call, n_contracts, max_retries)
        if success:
            self.logger.info(
                f"COVERED CALL {ticker} x{n_contracts}, strike={best_call.strike}"
            )
        else:
            self.logger.warning(f"{ticker}: covered call sell failed.")
        return success

    def open_straddle(
        self,
        ticker: str,
        strike_proximity_pct: float = 0.0,
        max_expiration_days: int = 30,
        num_contracts: int = 1,
        max_retries: int = 1,
    ) -> bool:
        """
        Buy a straddle: 1 long call + 1 long put at the same strike/expiration.
        """
        price = self.market_data[ticker]["close"]
        if not price or price <= 0:
            self.logger.warning(f"{ticker}: no price for straddle.")
            return False

        px = price
        low_strike = px * (1 - abs(strike_proximity_pct))
        high_strike = px * (1 + abs(strike_proximity_pct))

        calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(low_strike, high_strike),
            max_expiration_days=max_expiration_days,
            contract_type="call",
        )
        puts = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(low_strike, high_strike),
            max_expiration_days=max_expiration_days,
            contract_type="put",
        )
        if not calls or not puts:
            self.logger.info(f"{ticker}: missing call/put for straddle.")
            return False

        closest_call = min(calls, key=lambda c: abs(c.strike - px))
        closest_put = min(puts, key=lambda p: abs(p.strike - px))

        call_ok = self._retry_buy_option(closest_call, num_contracts, max_retries)
        put_ok = self._retry_buy_option(closest_put, num_contracts, max_retries)
        if call_ok and put_ok:
            self.logger.info(f"STRADDLE {ticker} strike={closest_call.strike}")
            return True

        self.logger.warning(f"{ticker}: straddle partial/fail.")
        return False

    def open_iron_condor(
        self,
        ticker: str,
        center_strike_proximity_pct: float = 0.0,
        wing_width: float = 5.0,
        max_expiration_days: int = 30,
        num_contracts: int = 1,
        max_retries: int = 1,
    ) -> bool:
        """
        Sells an iron condor:
         - Short call and put near ATM.
         - Long further OTM call and put for protection.
        """
        price = self.market_data[ticker]["close"]
        if not price or price <= 0:
            self.logger.warning(f"{ticker}: no price for condor.")
            return False

        px = price
        short_low = px * (1 - abs(center_strike_proximity_pct))
        short_high = px * (1 + abs(center_strike_proximity_pct))

        calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(short_low, short_high),
            max_expiration_days=max_expiration_days,
            contract_type="call",
        )
        puts = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(short_low, short_high),
            max_expiration_days=max_expiration_days,
            contract_type="put",
        )
        if not calls or not puts:
            self.logger.info(f"{ticker}: no calls/puts for condor center.")
            return False

        short_call = min(calls, key=lambda c: abs(c.strike - px))
        short_put = min(puts, key=lambda p: abs(p.strike - px))

        # Build wing strikes
        long_call_strike = short_call.strike + wing_width
        long_put_strike = short_put.strike - wing_width

        # Look up OTM calls/puts for the wings
        protect_calls = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(long_call_strike, long_call_strike + 5),
            max_expiration_days=max_expiration_days,
            contract_type="call",
        )
        protect_puts = self.data_provider.get_options_contracts(
            ticker=ticker,
            as_of=self.time,
            strike_range=(long_put_strike - 5, long_put_strike),
            max_expiration_days=max_expiration_days,
            contract_type="put",
        )
        if not protect_calls or not protect_puts:
            self.logger.info(f"{ticker}: can't find OTM wings.")
            return False

        long_call = protect_calls[0]
        long_put = protect_puts[-1]

        sc_ok = self._retry_sell_option(short_call, num_contracts, max_retries)
        sp_ok = self._retry_sell_option(short_put, num_contracts, max_retries)
        lc_ok = self._retry_buy_option(long_call, num_contracts, max_retries)
        lp_ok = self._retry_buy_option(long_put, num_contracts, max_retries)

        if sc_ok and sp_ok and lc_ok and lp_ok:
            self.logger.info(
                f"IRON CONDOR {ticker} short={short_call.strike}/{short_put.strike}"
            )
            return True

        self.logger.warning(
            f"{ticker}: condor partial/fail, SC={sc_ok}, SP={sp_ok}, LC={lc_ok}, LP={lp_ok}"
        )
        return False


def run_starter_kit_backtest(
    strategy_class: Any,
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
    polygon_api_key_var: str = "POLYGON_API_KEY",
    commission_cfg: Optional[CommissionConfig] = None,
    slippage_cfg: Optional[SlippageConfig] = None,
):
    """
    Quick backtest entry point:
      - Initializes your Strategy class (defaults to StarterBaseStrategy).
      - Uses a PolygonDataProvider with your API key.
      - Creates a Backtest object with commission/slippage if provided.
      - Runs the backtest and plots results.
    """
    api_key = os.getenv(polygon_api_key_var)
    if not api_key:
        raise ValueError(f"{polygon_api_key_var} not set.")

    data_provider = PolygonDataProvider(api_key)

    if commission_cfg is None:
        commission_cfg = CommissionConfig(
            min_commission=1.0, per_share=0.005, per_contract=0.65, percentage=0.0001
        )
    if slippage_cfg is None:
        slippage_cfg = SlippageConfig(
            base_points=1.0, vol_impact=0.1, spread_factor=0.5, market_impact=0.1
        )

    strategy_instance = strategy_class(data_provider=data_provider)
    engine = Backtest(
        data_provider=data_provider,
        strategy=strategy_instance,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        commission_config=commission_cfg,
        slippage_config=slippage_cfg,
        deterministic_fill=True,
    )

    results = engine.run()
    results.plot(include_drawdown=True)
    return results


class SophisticatedTestStrategy(StarterBaseStrategy):
    """
    Demonstrates a more sophisticated usage of StarterBaseStrategy, mixing:
      - Equity positions with stops
      - Covered calls on those equity positions
      - Option structures like straddles or iron condors
      - Simple technical logic (e.g., naive moving average or volatility check)
    """

    def __init__(self, data_provider):
        super().__init__(data_provider=data_provider, name="sophisticated_test")
        self.lookback = 5  # Example: small rolling window for naive signal
        # Keep track of a small price history for each ticker
        self.price_history = {}

    def on_data(self, timestamp: datetime, market_data: Dict[str, Any]) -> None:
        """
        Implement some naive trading logic to demonstrate multiple features:
         - Buys AAPL with a trailing stop when a short SMA is rising
         - Writes a covered call on AAPL after it rises significantly
         - Opens a straddle on MSFT if volatility is high
         - Opens an iron condor on TSLA if price is stable
        """
        self.time = timestamp
        self.market_data = market_data
        for ticker, data in list(market_data.items()):
            if len(ticker) > 5:
                continue  # Skip if it's an option ticker
            price = data["close"]

            # Update rolling history
            if ticker not in self.price_history:
                self.price_history[ticker] = []
            self.price_history[ticker].append(price)
            # Limit history size
            if len(self.price_history[ticker]) > self.lookback:
                self.price_history[ticker].pop(0)

            # If we don't have enough history, skip
            if len(self.price_history[ticker]) < self.lookback:
                continue

            # Simple check: is our short SMA sloping up?
            # We'll treat the earliest vs. latest in our small window
            old_price = self.price_history[ticker][0]
            newest_price = self.price_history[ticker][-1]

            # 1. AAPL logic: buy with trailing stop, possibly do a covered call
            if ticker == "AAPL":
                # If short SMA rising by at least $1 from earliest to newest
                # and we have no position, open a trailing stop buy.
                if (
                    newest_price - old_price > 1.0
                    and ticker not in self.portfolio.positions
                ):
                    self.open_stock_position(
                        ticker="AAPL",
                        allocation_pct=0.2,
                        stop_loss=None,
                        trailing_stop=5.0,
                    )
                # If we already hold AAPL and it has risen a lot, consider a covered call
                if ticker in self.portfolio.positions:
                    # If it jumped more than $5 from start to end of history, open covered call
                    if newest_price - old_price > 5.0:
                        self.open_covered_call(
                            ticker="AAPL",
                            stock_allocation_pct=0.0,  # Already have the stock
                            call_otm_pct=0.03,
                            max_expiration_days=45,
                        )

            # 2. MSFT logic: if short-term volatility is low, open a straddle
            if ticker == "MSFT":
                # Naive measure: standard deviation in our small window
                avg = sum(self.price_history[ticker]) / len(self.price_history[ticker])
                variance = sum(
                    (p - avg) ** 2 for p in self.price_history[ticker]
                ) / len(self.price_history[ticker])
                vol = variance**0.5
                # If short window volatility is below 0.3, open a straddle if we have none
                if vol < 0.3:
                    # Attempt a straddle (this won't double-open if we've already done it).
                    # In a real strategy, you'd track your positions carefully
                    self.open_straddle(
                        ticker="MSFT",
                        strike_proximity_pct=0.02,
                        max_expiration_days=30,
                        num_contracts=1,
                    )

            # 3. TSLA logic: if price stable (range < $2 over lookback), open an iron condor
            if ticker == "TSLA":
                mini = min(self.price_history[ticker])
                maxi = max(self.price_history[ticker])
                if (maxi - mini) < 2.0:
                    # Sell an iron condor
                    self.open_iron_condor(
                        ticker="TSLA",
                        center_strike_proximity_pct=0.01,
                        wing_width=5.0,
                        max_expiration_days=30,
                        num_contracts=1,
                    )


def run_sophisticated_test():
    """
    Runs a backtest with our SophisticatedTestStrategy on multiple tickers:
      - AAPL, MSFT, TSLA
      For demonstration over a short time window.
    """
    start = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end = datetime(2024, 2, 1, tzinfo=pytz.UTC)

    results = run_starter_kit_backtest(
        strategy_class=SophisticatedTestStrategy,
        tickers=["AAPL", "MSFT", "TSLA"],
        start_date=start,
        end_date=end,
        initial_capital=100000,
    )

    # You could do any post-processing or analysis of results here if desired
    return results


if __name__ == "__main__":
    run_sophisticated_test()
