import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    filename='logs/ivarb.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

class SimulatedOption:
    def __init__(self, strike, expiry, is_call, position_size, underlying_price, volatility, rate=0.02):
        self.strike = float(strike)
        self.expiry = expiry
        self.is_call = is_call
        self.position_size = float(position_size)
        self.volatility = min(max(float(volatility), 0.01), 5.0)
        self.rate = rate
        self.entry_price = self.black_scholes(float(underlying_price))
        self.entry_date = None

    def black_scholes(self, S, t=30/365):
        try:
            S = max(min(float(S), 1e6), 0.01)
            K = max(min(self.strike, 1e6), 0.01)
            r = max(min(float(self.rate), 0.1), -0.1)
            t = max(min(float(t), 2.0), 1/365)
            v = self.volatility

            d1 = (np.log(S/K) + (r + v**2/2)*t) / (v*np.sqrt(t))
            d2 = d1 - v*np.sqrt(t)

            if self.is_call:
                price = S * norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
            else:
                price = K*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)

            return max(price, 0.01)
        except Exception as e:
            logger.error(f"Error in Black-Scholes: {str(e)}")
            return 0.0

    def calculate_greeks(self, S, t=30/365):
        try:
            K = self.strike
            r = self.rate
            v = self.volatility
            d1 = (np.log(S/K) + (r + v**2/2)*t) / (v*np.sqrt(t))
            d2 = d1 - v*np.sqrt(t)

            if self.is_call:
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)

            gamma = norm.pdf(d1)/(S*v*np.sqrt(t))
            vega = S*np.sqrt(t)*norm.pdf(d1)
            theta = -(S*v*norm.pdf(d1))/(2*np.sqrt(t))

            return {
                'delta': delta * self.position_size,
                'gamma': gamma * self.position_size,
                'vega': vega * self.position_size,
                'theta': theta * self.position_size
            }
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}

class IVArbBacktester:
    def __init__(self, tickers, sector_etf, transaction_cost=1.0, slippage=0.005):
        self.tickers = tickers
        self.sector_etf = sector_etf
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.data = {}
        self.positions = {}
        self.prev_volscore = {ticker: None for ticker in tickers}
        self.last_trade_date = {ticker: None for ticker in tickers}  # Track last trade per ticker

        self.risk_limits = {
            'position_stop_loss': -0.05,
            'min_profit_taking': 0.05,
            'max_position_size': 0.05,
            'max_portfolio_vol': 0.10,
            'max_positions': 5,
            'min_entry_vol': 0.15,
            'max_entry_vol': 0.75,
            'min_holding_days': 3,
            'max_holding_days': 21,
            'delta_limit': 0.10,
            'rebalance_threshold': 0.25,
            'vol_score_threshold': 0.20,     # Lowered from 0.20
            'min_trade_spacing': 1
        }

    def fetch_data(self, start_date, end_date):
        """Fetch data with proper moving averages"""
        print("\nFetching data...")

        # Get sector ETF data first
        try:
            sector = yf.Ticker(self.sector_etf)
            sector_hist = sector.history(start=start_date, end=end_date)

            if len(sector_hist) == 0:
                raise ValueError(f"No data for sector ETF {self.sector_etf}")

            # Ensure unique dates
            sector_hist = sector_hist[~sector_hist.index.duplicated(keep='last')]

            sector_returns = np.log(sector_hist['Close'] / sector_hist['Close'].shift(1))
            sector_hist['rv_20'] = sector_returns.rolling(window=20).std() * np.sqrt(252)
            sector_hist['rv_60'] = sector_returns.rolling(window=60).std() * np.sqrt(252)
            sector_hist['price_ma20'] = sector_hist['Close'].rolling(window=20).mean()
            sector_hist['price_ma60'] = sector_hist['Close'].rolling(window=60).mean()

            # Simulate sector IV
            base_vol = sector_hist['rv_20'].rolling(window=10).mean()
            hist_premium = 1.1  # Fixed premium
            sector_hist['iv'] = base_vol * hist_premium

            self.data[self.sector_etf] = sector_hist
            print(f"Processed {self.sector_etf}: {len(sector_hist)} days")

        except Exception as e:
            logger.error(f"Error processing sector ETF: {str(e)}")
            return

        # Process individual stocks
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)

                if len(hist) == 0:
                    logger.warning(f"No data for {ticker}")
                    continue

                # Ensure unique dates
                hist = hist[~hist.index.duplicated(keep='last')]

                # Calculate returns and volatility metrics
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                hist['rv_20'] = returns.rolling(window=20).std() * np.sqrt(252)
                hist['rv_60'] = returns.rolling(window=60).std() * np.sqrt(252)

                # Add price moving averages
                hist['price_ma20'] = hist['Close'].rolling(window=20).mean()
                hist['price_ma60'] = hist['Close'].rolling(window=60).mean()

                # Better IV estimation
                base_vol = hist['rv_20'].rolling(window=10).mean()
                vol_premium = 1.1 + (hist['rv_20'] - hist['rv_60']).clip(-0.1, 0.1)
                hist['iv'] = base_vol * vol_premium

                # Calculate correlation with sector
                if self.sector_etf in self.data:
                    sector_rets = sector_returns[hist.index]
                    hist['sector_corr'] = returns.rolling(window=60).corr(sector_rets)

                # Add trend indicators
                hist['price_trend'] = (hist['Close'] > hist['price_ma20']).astype(int)
                hist['vol_trend'] = (hist['rv_20'] > hist['rv_60']).astype(int)

                hist.dropna(inplace=True)
                self.data[ticker] = hist
                print(f"Processed {ticker}: {len(hist)} days")

                # Print some debug info
                print(f"\n{ticker} debug info:")
                print(f"Moving averages present: {all(x in hist.columns for x in ['price_ma20', 'price_ma60'])}")
                print(f"Latest price: {hist['Close'].iloc[-1]:.2f}")
                print(f"Latest MA20: {hist['price_ma20'].iloc[-1]:.2f}")
                print(f"In uptrend: {hist['Close'].iloc[-1] > hist['price_ma20'].iloc[-1]}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")

        print("\nData fetch complete. Summary:")
        for ticker in [self.sector_etf] + self.tickers:
            if ticker in self.data:
                data = self.data[ticker]
                print(f"{ticker}: {len(data)} days, columns: {list(data.columns)}")


    def calculate_volscore(self, ticker, date):
        """Recalibrated volscore calculation"""
        try:
            if ticker not in self.data or date not in self.data[ticker].index:
                return None
            if self.sector_etf not in self.data or date not in self.data[self.sector_etf].index:
                return None

            s_data = self.data[ticker].loc[date]
            e_data = self.data[self.sector_etf].loc[date]

            # More generous convergence threshold
            convergence_threshold = 0.08  # 8% instead of 5%

            # Calculate basic metrics
            iv_rv_ratio = ((s_data['iv'] / s_data['rv_20']) - 1).clip(-1, 1)
            sector_spread = ((s_data['iv'] - e_data['iv']) / e_data['iv']).clip(-1, 1)

            # Add momentum component
            if date - pd.Timedelta(days=5) in self.data[ticker].index:
                prev_iv = self.data[ticker].loc[date - pd.Timedelta(days=5), 'iv']
                iv_momentum = ((s_data['iv'] / prev_iv) - 1).clip(-0.5, 0.5)
            else:
                iv_momentum = 0

            # Get sector correlation
            if 'sector_corr' in s_data and not pd.isna(s_data['sector_corr']):
                sector_correlation = abs(s_data['sector_corr'])
            else:
                sector_correlation = 0.5

            # Check convergence with look-ahead bias prevention
            is_converged = abs(iv_rv_ratio) < convergence_threshold

            # Enhanced scoring with momentum
            weights = {
                'iv_rv': 0.5,      # Reduced weight
                'sector': 0.3,     # Reduced weight
                'momentum': 0.2    # Added momentum component
            }

            base_score = (weights['iv_rv'] * iv_rv_ratio +
                         weights['sector'] * sector_spread +
                         weights['momentum'] * iv_momentum)

            # Print debug info
            print(f"\nVolscore details for {ticker}:")
            print(f"Stock IV: {s_data['iv']:.3f}")
            print(f"Stock RV20: {s_data['rv_20']:.3f}")
            print(f"Sector IV: {e_data['iv']:.3f}")
            print(f"IV/RV Ratio: {iv_rv_ratio:.3f}")
            print(f"Sector Spread: {sector_spread:.3f}")
            print(f"IV Momentum: {iv_momentum:.3f}")
            print(f"Sector Correlation: {sector_correlation:.3f}")
            print(f"Is Converged: {is_converged}")
            print(f"Base Score: {base_score:.3f}")

            return base_score, is_converged, sector_correlation

        except Exception as e:
            logger.error(f"Error in volscore calculation: {str(e)}")
            return None, False, 0.5


    def calculate_portfolio_delta(self, date):
        """Calculate total portfolio delta"""
        total_delta = 0.0
        try:
            for ticker, positions in self.positions.items():
                if ticker not in self.data or date not in self.data[ticker].index:
                    continue
                spot = float(self.data[ticker].loc[date, 'Close'])
                for pos in positions:
                    dte = (pos.expiry - date).days
                    if dte > 0:
                        greeks = pos.calculate_greeks(spot, dte/365)
                        total_delta += greeks['delta']
        except Exception as e:
            logger.error(f"Error calculating portfolio delta: {str(e)}")
        return total_delta

    def calculate_pnl(self, position, entry_date, current_date, ticker):
        """Calculate position PnL"""
        try:
            if ticker not in self.data or current_date not in self.data[ticker].index:
                return 0.0

            current_spot = float(self.data[ticker].loc[current_date, 'Close'])
            days_to_exp = (position.expiry - current_date).days

            if days_to_exp <= 0:
                if position.is_call:
                    current_price = max(current_spot - position.strike, 0)
                else:
                    current_price = max(position.strike - current_spot, 0)
            else:
                current_price = position.black_scholes(current_spot, days_to_exp/365)

            # Apply slippage
            if current_price > position.entry_price:
                current_price *= (1 - self.slippage)
            else:
                current_price *= (1 + self.slippage)

            pnl = (current_price - position.entry_price) * position.position_size * 100
            pnl -= self.transaction_cost * abs(position.position_size)

            return pnl

        except Exception as e:
            logger.error(f"Error calculating PnL: {str(e)}")
            return 0.0


    def process_existing_positions(self, date, pv, pnl_daily):
        """Enhanced position management with adaptive exits"""
        try:
            for ticker in list(self.positions.keys()):
                if ticker not in self.data or date not in self.data[ticker].index:
                    continue

                positions = self.positions[ticker]

                # Track sector exposure
                sector_positions = sum(1 for t, p in self.positions.items()
                                    if 'sector_corr' in self.data[t].loc[date]
                                    and self.data[t].loc[date]['sector_corr'] > 0.5)

                for pos in positions[:]:
                    days_held = (date - pos.entry_date).days
                    if days_held < self.risk_limits['min_holding_days']:
                        continue

                    pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)
                    pct = pnl / pv if pv > 0 else 0

                    # Adaptive stop loss - tighter for correlated positions
                    if 'sector_corr' in self.data[ticker].loc[date]:
                        correlation = self.data[ticker].loc[date]['sector_corr']
                        stop_loss = self.risk_limits['position_stop_loss'] * (1.0 - correlation * 0.3)
                    else:
                        stop_loss = self.risk_limits['position_stop_loss']

                    # Early profit taking if significant gain in short time
                    if days_held < 10 and pct >= self.risk_limits['min_profit_taking'] * 1.5:
                        print(f"Taking early profit on {ticker} ({pct:.1%} in {days_held} days)")
                        pnl_daily += pnl
                        positions.remove(pos)
                        continue

                    # Regular stop loss
                    if pct <= stop_loss:
                        print(f"Stop loss triggered for {ticker}")
                        pnl_daily += pnl
                        positions.remove(pos)
                        continue

                    # Adaptive profit target based on volatility
                    vol = self.data[ticker].loc[date]['iv']
                    profit_target = self.risk_limits['min_profit_taking'] * (1 + (vol - 0.2))
                    if pct >= profit_target:
                        print(f"Profit target hit for {ticker}")
                        pnl_daily += pnl
                        positions.remove(pos)
                        continue

                    # Max holding period only for losing trades
                    if days_held > self.risk_limits['max_holding_days']:
                        if pct < 0 or sector_positions > 3:  # Force close if too many sector positions
                            print(f"Closing losing position at max hold time")
                            pnl_daily += pnl
                            positions.remove(pos)
                        else:
                            # For winning trades, use trailing stop
                            trailing_stop = max(pct * 0.5, stop_loss)
                            if pct <= trailing_stop:
                                print(f"Trailing stop hit")
                                pnl_daily += pnl
                                positions.remove(pos)

                if not positions:
                    del self.positions[ticker]

            return pnl_daily

        except Exception as e:
            logger.error(f"Error processing positions: {str(e)}")
            return pnl_daily


    def look_for_new_positions(self, date, pv):
        """Modified trade entry with explicit checks"""
        try:
            for ticker in self.tickers:
                # Debug print
                print(f"\nChecking {ticker} for trade entry:")

                # Check minimum trade spacing
                if (self.last_trade_date[ticker] is not None and
                    (date - self.last_trade_date[ticker]).days < self.risk_limits['min_trade_spacing']):
                    print(f"Too soon since last trade for {ticker}")
                    continue

                if ticker in self.positions:
                    print(f"Already have position in {ticker}")
                    continue

                if ticker not in self.data or date not in self.data[ticker].index:
                    print(f"No data for {ticker} on {date}")
                    continue

                s_data = self.data[ticker].loc[date]
                spot = float(s_data['Close'])
                vol = float(s_data['iv'])

                # Check if we have moving averages
                if 'price_ma20' not in s_data:
                    print(f"Missing price_ma20 for {ticker}")
                    continue

                is_uptrend = spot > s_data['price_ma20']
                print(f"Price trend check - Current: {spot:.2f}, MA20: {s_data['price_ma20']:.2f}, Uptrend: {is_uptrend}")

                vscore_result = self.calculate_volscore(ticker, date)
                if vscore_result is None:
                    print(f"No volscore for {ticker}")
                    continue

                vscore, is_converged, sector_correlation = vscore_result
                print(f"Volscore: {vscore:.3f}, Converged: {is_converged}, Correlation: {sector_correlation:.3f}")

                # Base position size
                size = self.calculate_position_size(pv, spot, vol, date, sector_correlation)
                expiry_date = date + pd.Timedelta(days=30)

                # Short vol trade
                if vscore > self.risk_limits['vol_score_threshold']:
                    print(f"Short vol signal detected (score: {vscore:.3f})")
                    if vol > self.risk_limits['min_entry_vol']:
                        call = SimulatedOption(spot, expiry_date, True, -size, spot, vol)
                        put = SimulatedOption(spot, expiry_date, False, -size, spot, vol)

                        if call.entry_price > 0.01 and put.entry_price > 0.01:
                            call.entry_date = put.entry_date = date
                            self.positions[ticker] = [call, put]
                            self.last_trade_date[ticker] = date
                            print(f"Opened short straddle on {ticker}, size: {size}")
                    else:
                        print(f"Vol too low for short trade: {vol:.3f} vs {self.risk_limits['min_entry_vol']:.3f}")

                # Convergence trade
                elif is_converged:
                    print("Convergence signal detected")
                    if is_uptrend:
                        atm_call = SimulatedOption(spot, expiry_date, True, size*1.5, spot, vol)
                        otm_call = SimulatedOption(spot * 1.03, expiry_date, True, -size*1.5, spot, vol)

                        if atm_call.entry_price > 0.01 and otm_call.entry_price > 0.01:
                            atm_call.entry_date = otm_call.entry_date = date
                            self.positions[ticker] = [atm_call, otm_call]
                            self.last_trade_date[ticker] = date
                            print(f"Opened long call spread on {ticker}, size: {size*1.5}")
                    else:
                        print("Not in uptrend for convergence trade")

        except Exception as e:
            print(f"Error looking for new positions: {str(e)}")
            import traceback
            print(traceback.format_exc())


    def calculate_position_size(self, portfolio_value, spot_price, vol, date, sector_correlation=0.5, delta=1.0):
        """Position sizing with correlation and volatility scaling"""
        try:
            # Base size as percentage of current portfolio
            max_notional = portfolio_value * self.risk_limits['max_position_size']

            # Scale down size based on sector correlation
            correlation_scalar = 1.0 - (sector_correlation * 0.5)  # 0.5 to 1.0

            # Scale by volatility - less aggressive in high vol
            vol_scalar = 1.0 / (vol * np.sqrt(63))

            # Additional volatility adjustments
            if vol > 0.6:
                vol_scalar *= 0.5  # Much smaller size in very high vol
            elif vol > 0.4:
                vol_scalar *= 0.7  # Smaller size in high vol

            # Calculate sector exposure
            sector_exposure = sum(
                abs(sum(pos.position_size for pos in positions))
                for ticker, positions in self.positions.items()
                if 'sector_corr' in self.data[ticker].loc[date]
                and self.data[ticker].loc[date]['sector_corr'] > 0.5
            )

            # Scale down if sector exposure is high
            if sector_exposure > 200:  # Arbitrary threshold
                sector_scalar = 0.5
            else:
                sector_scalar = 1.0

            # Final position size calculation
            position_size = (max_notional * vol_scalar * correlation_scalar * sector_scalar) / spot_price

            # More conservative limits
            min_contracts = 2
            max_contracts = min(50, int(portfolio_value * 0.001))  # Dynamic max based on portfolio size

            return max(min_contracts, min(int(position_size), max_contracts))

        except Exception as e:
            logger.error(f"Error in position sizing: {str(e)}")
            return 2  # Minimum size on error

    def rebalance_delta(self, date, portfolio_value):
        """Much less frequent delta hedging"""
        try:
            portfolio_delta = self.calculate_portfolio_delta(date)

            # Only hedge if delta is extremely high
            if abs(portfolio_delta) <= self.risk_limits['delta_limit']:
                return

            # When we do hedge, do it less aggressively
            for ticker, positions in self.positions.items():
                if ticker not in self.data or date not in self.data[ticker].index:
                    continue

                spot = float(self.data[ticker].loc[date, 'Close'])
                vol = float(self.data[ticker].loc[date, 'iv'])

                position_delta = 0
                for pos in positions:
                    dte = (pos.expiry - date).days
                    if dte <= 0:
                        continue
                    greeks = pos.calculate_greeks(spot, dte/365)
                    position_delta += greeks['delta']

                # Only hedge extremely large deltas
                if abs(position_delta) > self.risk_limits['rebalance_threshold']:
                    hedge_size = int(self.calculate_position_size(portfolio_value, spot, vol, date) * 0.5)  # Smaller hedge
                    hedge_size = int(hedge_size * np.sign(-position_delta))

                    if position_delta > 0:
                        hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                True, hedge_size, spot, vol)
                    else:
                        hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                False, hedge_size, spot, vol)

                    hedge.entry_date = date
                    positions.append(hedge)

        except Exception as e:
            logger.error(f"Error in delta rebalancing: {str(e)}")

    def strategy(self, initial_capital=1000000):
        print("\nStarting strategy execution...")

        # Initialize portfolio
        pv = float(initial_capital)
        ph = [pv]
        vs_hist = {t: [] for t in self.tickers}

        # Track trades
        trades_taken = 0
        short_vol_trades = 0
        convergence_trades = 0

        # Track closed trade stats
        self.total_closed_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_win_pnl = 0.0
        self.total_loss_pnl = 0.0

        # Get valid dates
        valid_dates = None
        for ticker in self.tickers + [self.sector_etf]:
            if ticker in self.data and not self.data[ticker].empty:
                dates = self.data[ticker].index
                valid_dates = dates if valid_dates is None else valid_dates.intersection(dates)
        if len(valid_dates) == 0:
            logger.error("No valid trading dates found")
            return pd.Series([initial_capital]), vs_hist

        valid_dates = sorted(valid_dates)
        print(f"Trading period: {valid_dates[0]} to {valid_dates[-1]}")

        portfolio_values = pd.Series(index=valid_dates, dtype=float)
        portfolio_values.iloc[0] = pv

        for i, date in enumerate(valid_dates[:-1]):
            try:
                if i % 20 == 0:
                    print(f"\nProcessing date {date}")
                    print(f"Portfolio Value: ${pv:,.2f}")
                    print(f"Total Trades: {trades_taken} (Short Vol: {short_vol_trades}, Convergence: {convergence_trades})")

                pnl_daily = 0.0

                # Process existing positions
                for ticker in list(self.positions.keys()):
                    if ticker not in self.data or date not in self.data[ticker].index:
                        continue

                    positions = self.positions[ticker]
                    for pos in positions[:]:
                        days_held = (date - pos.entry_date).days
                        if days_held < self.risk_limits['min_holding_days']:
                            continue

                        pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)
                        pct = pnl / pv if pv > 0 else 0

                        # Check exit conditions
                        if pct <= self.risk_limits['position_stop_loss']:
                            pnl_daily += pnl
                            positions.remove(pos)
                            self.total_closed_trades += 1
                            if pnl > 0:
                                self.winning_trades += 1
                                self.total_win_pnl += pnl
                            else:
                                self.losing_trades += 1
                                self.total_loss_pnl += pnl
                        elif pct >= self.risk_limits['min_profit_taking']:
                            pnl_daily += pnl
                            positions.remove(pos)
                            self.total_closed_trades += 1
                            if pnl > 0:
                                self.winning_trades += 1
                                self.total_win_pnl += pnl
                            else:
                                self.losing_trades += 1
                                self.total_loss_pnl += pnl
                        elif days_held > self.risk_limits['max_holding_days']:
                            pnl_daily += pnl
                            positions.remove(pos)
                            self.total_closed_trades += 1
                            if pnl > 0:
                                self.winning_trades += 1
                                self.total_win_pnl += pnl
                            else:
                                self.losing_trades += 1
                                self.total_loss_pnl += pnl

                    if not positions:
                        del self.positions[ticker]

                # Look for new positions if under max positions
                if len(self.positions) < self.risk_limits['max_positions']:
                    for ticker in self.tickers:
                        if ticker in self.positions:
                            continue
                        if ticker not in self.data or date not in self.data[ticker].index:
                            continue

                        spot = float(self.data[ticker].loc[date, 'Close'])
                        vol = float(self.data[ticker].loc[date, 'iv'])

                        vscore_result = self.calculate_volscore(ticker, date)
                        if vscore_result is None:
                            continue
                        vscore, is_converged, sector_corr = vscore_result
                        vs_hist[ticker].append((date, vscore))

                        if abs(vscore) > self.risk_limits['vol_score_threshold'] or is_converged:
                            size = self.calculate_position_size(pv, spot, vol, date)
                            expiry_date = date + pd.Timedelta(days=30)

                            # Short vol
                            if vscore > self.risk_limits['vol_score_threshold']:
                                call = SimulatedOption(spot, expiry_date, True, -size, spot, vol)
                                put = SimulatedOption(spot, expiry_date, False, -size, spot, vol)
                                if call.entry_price > 0.01 and put.entry_price > 0.01:
                                    call.entry_date = date
                                    put.entry_date = date
                                    self.positions[ticker] = [call, put]
                                    trades_taken += 1
                                    short_vol_trades += 1
                                    print(f"Opened short straddle on {ticker}, size: {size}")

                            # Convergence trade
                            elif is_converged:
                                atm_call = SimulatedOption(spot, expiry_date, True, size*1.5, spot, vol)
                                otm_call = SimulatedOption(spot*1.03, expiry_date, True, -size*1.5, spot, vol)
                                if atm_call.entry_price > 0.01 and otm_call.entry_price > 0.01:
                                    atm_call.entry_date = date
                                    otm_call.entry_date = date
                                    self.positions[ticker] = [atm_call, otm_call]
                                    trades_taken += 1
                                    convergence_trades += 1
                                    print(f"Opened long call spread on {ticker} (convergence), size: {size*1.5}")

                # Update portfolio value
                old_pv = pv
                pv = max(pv + pnl_daily, 0)
                ph.append(pv)
                portfolio_values.loc[date] = pv

            except Exception as e:
                logger.error(f"Error processing date {date}: {str(e)}")
                continue

        last_date = valid_dates[-1]
        for ticker, positions in list(self.positions.items()):
            for pos in positions:
                # Realistically get the final mark
                final_pnl = self.calculate_pnl(pos, pos.entry_date, last_date, ticker)
                # Book it, update counters, etc.
                self.total_closed_trades += 1
                if final_pnl > 0:
                    self.winning_trades += 1
                    self.total_win_pnl += final_pnl
                else:
                    self.losing_trades += 1
                    self.total_loss_pnl += final_pnl
            del self.positions[ticker]

        # Final stats (trade-based)
        print("\nTrading Statistics:")
        print(f"Total Trades Taken: {trades_taken}")
        print(f"  - Short Vol Trades: {short_vol_trades}")
        print(f"  - Convergence Trades: {convergence_trades}")
        print(f"Trades Closed: {self.total_closed_trades}")

        # Compute trade-based win rate
        if self.total_closed_trades > 0:
            closed_win_rate = self.winning_trades / self.total_closed_trades
        else:
            closed_win_rate = 0.0
        print(f"Trade Win Rate: {closed_win_rate*100:.1f}%")

        # Compute total closed PnL
        closed_pnl = self.total_win_pnl + self.total_loss_pnl
        print(f"Total Closed PnL: ${closed_pnl:,.2f}")

        return portfolio_values.fillna(method='ffill'), vs_hist

    def plot_results(self, portfolio_series, vs_hist):
        """Plot with aligned axes and external legend"""
        try:
            fig = plt.figure(figsize=(15, 12))

            # Create a gridspec for the plots only (no legend)
            gs_plots = plt.GridSpec(2, 1, height_ratios=[1, 1])

            # Portfolio value plot
            ax1 = fig.add_subplot(gs_plots[0])
            portfolio_series.plot(ax=ax1)
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_xlabel('')  # Remove x-label since it's not the bottom plot
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.grid(True)

            # IV/RV plot
            ax2 = fig.add_subplot(gs_plots[1])

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            lines = []
            labels = []
            lined = {}

            for i, ticker in enumerate(self.tickers):
                if ticker in self.data:
                    data = self.data[ticker]
                    color = colors[i % len(colors)]

                    iv_line = ax2.plot(data.index, data['iv'] * 100,
                                     color=color, linestyle='-',
                                     label=f'{ticker} IV', alpha=0.7)[0]
                    rv_line = ax2.plot(data.index, data['rv_20'] * 100,
                                     color=color, linestyle='--',
                                     label=f'{ticker} RV', alpha=0.7)[0]

                    lines.extend([iv_line, rv_line])
                    labels.extend([f'{ticker} IV', f'{ticker} RV'])

            ax2.set_title('Implied vs Realized Volatility')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Volatility (%)')
            ax2.grid(True)

            # Adjust the subplot parameters to get the plots aligned
            plt.subplots_adjust(right=0.85)  # Make room for legend on the right

            # Create legend outside the plots
            leg = ax2.legend(lines, labels,
                            loc='center left',
                            bbox_to_anchor=(1.05, 0.5),
                            fontsize='small',
                            labelspacing=0.2)

            leg.get_frame().set_alpha(0.8)

            # Map legend lines to actual plot lines
            for legline, origline in zip(leg.get_lines(), lines):
                legline.set_picker(5)
                lined[legline] = origline

            def on_pick(event):
                try:
                    legline = event.artist
                    origline = lined[legline]
                    vis = not origline.get_visible()
                    origline.set_visible(vis)
                    legline.set_alpha(1.0 if vis else 0.2)
                    fig.canvas.draw()
                except Exception as e:
                    print(f"Error in click handler: {str(e)}")

            fig.canvas.mpl_connect('pick_event', on_pick)

            # Make sure the plots are perfectly aligned
            fig.align_xlabels([ax1, ax2])

            plt.tight_layout()
            # Adjust after tight_layout to fine-tune legend position
            plt.subplots_adjust(right=0.85)

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise e

    def calculate_performance_metrics(self, portfolio_series, risk_free_rate=0.02):
        if len(portfolio_series) < 2:
            return None

        # 1) Day-based stats: total/annual returns, Sharpe, Sortino, max drawdown
        returns = portfolio_series.pct_change().dropna()
        if len(returns) == 0:
            return None

        start_val = portfolio_series.iloc[0]
        end_val = portfolio_series.iloc[-1]
        total_return = (end_val / start_val) - 1  # overall growth
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) else 0

        ann_vol = returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        excess_returns = returns - risk_free_rate / 252
        sharpe = (np.sqrt(252) * excess_returns.mean() / returns.std()) if returns.std() else 0

        neg_excess = excess_returns[excess_returns < 0]
        if len(neg_excess) > 0 and neg_excess.std() != 0:
            sortino = np.sqrt(252) * (excess_returns.mean() / neg_excess.std())
        else:
            sortino = 0

        rolling_max = portfolio_series.expanding().max()
        drawdowns = portfolio_series / rolling_max - 1
        max_drawdown = drawdowns.min()

        # 2) Trade-based stats: use counters from your code that track closed trades
        trade_count = getattr(self, 'total_closed_trades', 0)
        trade_win_count = getattr(self, 'winning_trades', 0)
        trade_loss_count = getattr(self, 'losing_trades', 0)
        total_win_pnl = getattr(self, 'total_win_pnl', 0)
        total_loss_pnl = getattr(self, 'total_loss_pnl', 0)

        if trade_count > 0:
            win_rate = trade_win_count / trade_count
            avg_win = total_win_pnl / trade_win_count if trade_win_count > 0 else 0
            avg_loss = total_loss_pnl / trade_loss_count if trade_loss_count > 0 else 0
            if trade_loss_count > 0 and total_loss_pnl < 0:
                profit_factor = abs(total_win_pnl / total_loss_pnl)
            else:
                profit_factor = float('inf') if total_win_pnl > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': trade_count,
            'trading_days': len(returns),
        }


def run_backtest():
    """Run backtest with debug info"""
    tickers = [
        'AAPL', 'NVDA', 'MSFT', 'AVGO', 'CRM', 'ORCL', 'CSCO', 'NOW', 'ACN', 'IBM',
        'AMD', 'ADBE', 'INTU', 'QCOM', 'TXN', 'PLTR', 'AMAT', 'PANW', 'ANET', 'ADI',
        'MU', 'LRCX', 'INTC', 'KLAC', 'APH', 'CRWD', 'CDNS', 'MSI', 'SNPS', 'ADSK',
        'FTNT', 'WDAY', 'ROP', 'NXPI', 'FICO', 'TEL', 'CTSH', 'IT', 'GLW', 'DELL',
        'HPQ', 'MCHP', 'MPWR', 'ANSS', 'GDDY', 'HPE', 'KEYS', 'ON', 'TYL', 'NTAP',
        'CDW', 'PTC', 'TDY', 'WDC', 'TER', 'ZBRA', 'FSLR', 'STX', 'TRMB', 'SMCI',
        'VRSN', 'JBL', 'GEN', 'FFIV', 'AKAM', 'SWKS', 'EPAM', 'JNPR', 'ENPH'
    ]

    sector_etf = 'XLK'
    start_date = '2019-01-01'
    end_date = '2024-01-01'

    print(f"\nInitializing backtest:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Sector ETF: {sector_etf}")

    vb = IVArbBacktester(
        tickers=tickers,
        sector_etf=sector_etf,
        transaction_cost=0.5,
        slippage=0.002
    )

    vb.risk_limits = {
        'position_stop_loss': -0.05,
        'min_profit_taking': 0.08,
        'max_position_size': 0.05,
        'max_portfolio_vol': 0.10,
        'max_positions': 3,
        'min_entry_vol': 0.15,
        'max_entry_vol': 0.75,
        'min_holding_days': 3,
        'max_holding_days': 21,
        'delta_limit': 0.10,
        'rebalance_threshold': 0.15,
        'vol_score_threshold': 0.10,
        'min_trade_spacing': 1
    }

    print("\nRisk Parameters:")
    for param, value in vb.risk_limits.items():
        print(f"{param}: {value}")

    print("\nFetching data...")
    vb.fetch_data(start_date=start_date, end_date=end_date)

    print("\nChecking data quality:")
    for ticker in [sector_etf] + tickers:
        if ticker in vb.data:
            data = vb.data[ticker]
            print(f"\n{ticker}:")
            print(f"Days of data: {len(data)}")
            print(f"IV range: {data['iv'].min():.2%} to {data['iv'].max():.2%}")
            print(f"RV range: {data['rv_20'].min():.2%} to {data['rv_20'].max():.2%}")

    print("\nRunning strategy...")
    portfolio_series, vs_hist = vb.strategy(initial_capital=10000)

    print("\nChecking strategy output:")
    print(f"Portfolio series length: {len(portfolio_series)}")
    print(f"Number of unique values: {len(portfolio_series.unique())}")
    print(f"Start value: ${portfolio_series.iloc[0]:,.2f}")
    print(f"End value: ${portfolio_series.iloc[-1]:,.2f}")

    print("\nCalculating metrics...")
    metrics = vb.calculate_performance_metrics(portfolio_series)

    if metrics is not None:
        print("\nFinal Performance Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in metric:
                    print(f"{metric}: {value:.2f}")
                elif 'return' in metric or 'rate' in metric or 'drawdown' in metric:
                    print(f"{metric}: {value:.2%}")
                else:
                    print(f"{metric}: {value:.2f}")
            else:
                print(f"{metric}: {value}")

    print("\nPlotting results...")
    vb.plot_results(portfolio_series, vs_hist)

if __name__ == "__main__":
    run_backtest()
