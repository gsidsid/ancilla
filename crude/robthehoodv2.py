import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    filename='logs/strategy_diagnostics.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Set to DEBUG for detailed logs
)
logger = logging.getLogger(__name__)

class SimulatedOption:
    def __init__(self, strike, expiry, is_call, position_size, underlying_price, volatility, rate=0.02):
        self.strike = float(strike)
        self.expiry = expiry
        self.is_call = is_call
        self.position_size = float(position_size)
        self.volatility = min(max(float(volatility), 0.05), 1.0)
        self.rate = rate
        self.entry_price = self.black_scholes(float(underlying_price))
        self.entry_date = None

    def black_scholes(self, S, t=30/365):
        try:
            # Input validation
            S = max(min(float(S), 1e6), 0.01)
            K = max(min(self.strike, 1e6), 0.01)
            r = max(min(float(self.rate), 0.1), -0.1)
            t = max(min(float(t), 2.0), 1/365)
            v = self.volatility

            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + v**2/2)*t) / (v*np.sqrt(t))
            d2 = d1 - v*np.sqrt(t)

            # Calculate price with bounds
            if self.is_call:
                price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
                price = min(price, S)
            else:
                price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)  # Corrected to use norm.cdf
                price = min(price, K)

            # Ensure price >= intrinsic value
            intrinsic = max(S - K, 0) if self.is_call else max(K - S, 0)
            price = max(price, intrinsic)

            return max(price, 0.01)

        except Exception as e:
            logger.error(f"Error in Black-Scholes: {str(e)}")
            return 0.0

    def calculate_greeks(self, S, t=30/365):
        try:
            K = self.strike
            r = self.rate
            v = self.volatility

            d1 = (np.log(S/K) + (r + v**2 / 2) * t) / (v * np.sqrt(t))
            d2 = d1 - v * np.sqrt(t)

            # Delta
            if self.is_call:
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)

            # Gamma
            gamma = norm.pdf(d1)/(S*v*np.sqrt(t))

            # Vega
            vega = S*np.sqrt(t)*norm.pdf(d1)

            # Scale by position size
            return {
                'delta': delta * self.position_size,
                'gamma': gamma * self.position_size,
                'vega': vega * self.position_size
            }

        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {'delta': 0, 'gamma': 0, 'vega': 0}

class VolatilityArbitrageBacktester:
    def __init__(self, tickers, sector_etf, transaction_cost=1.0, slippage=0.005):
        self.tickers = tickers
        self.sector_etf = sector_etf
        self.data = {}
        self.positions = {}

        self.transaction_cost = transaction_cost  # Flat fee per contract
        self.slippage = slippage  # Percentage slippage per trade

        self.prev_volscore = {ticker: None for ticker in self.tickers}  # Initialize previous volscore

        self.risk_limits = {
            'position_stop_loss': -0.05,      # 5% stop loss
            'min_profit_taking': 0.08,        # 8% profit target
            'max_position_size': 0.05,        # 5% position size
            'max_portfolio_vol': 0.10,        # 10% target vol
            'max_positions': 3,                # Maximum positions
            'min_entry_vol': 0.15,             # Minimum IV for entry
            'max_entry_vol': 0.75,             # Maximum IV for entry
            'min_holding_days': 1,             # Minimum holding period
            'max_holding_days': 21,            # Maximum holding period
            'delta_limit': 0.10,               # Maximum portfolio delta
            'rebalance_threshold': 0.10        # Rebalance threshold
        }

    def fetch_data(self, start_date, end_date):
        print("\nFetching data...")

        for ticker in self.tickers + [self.sector_etf]:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if len(hist) == 0:
                raise ValueError(f"No data found for {ticker}")

            returns = np.log(hist['Close'] / hist['Close'].shift(1)).clip(-0.5, 0.5)
            hist['rv_20'] = returns.rolling(window=20).std().clip(0.01, 1.0) * np.sqrt(252)
            hist['rv_60'] = returns.rolling(window=60).std().clip(0.01, 1.0) * np.sqrt(252)

            # Simulate implied volatility with some randomness
            hist['iv'] = hist['rv_20'] * np.random.uniform(0.9, 1.3, len(hist))  # Increased upper bound for more variability
            hist['iv'] = hist['iv'].clip(0.05, 1.0)

            # Drop rows with NaN values to ensure complete data
            hist.dropna(inplace=True)

            # Log IV statistics
            iv_mean = hist['iv'].mean()
            iv_std = hist['iv'].std()
            logger.debug(f"{ticker} IV - Mean: {iv_mean:.2f}, Std: {iv_std:.2f}")

            self.data[ticker] = hist
            print(f"Processed {ticker}: {len(hist)} days")

    def calculate_volscore(self, ticker, date):
        try:
            stock_data = self.data[ticker].loc[date]
            sector_data = self.data[self.sector_etf].loc[date]

            iv_rv_ratio = ((stock_data['iv'] / stock_data['rv_20']) - 1).clip(-1, 1)
            sector_spread = ((stock_data['iv'] - sector_data['iv']) / sector_data['iv']).clip(-1, 1)
            vol_trend = ((stock_data['rv_20'] / stock_data['rv_60']) - 1).clip(-1, 1)

            weights = {'iv_rv': 0.4, 'sector': 0.4, 'trend': 0.2}
            volscore = (weights['iv_rv'] * iv_rv_ratio +
                      weights['sector'] * sector_spread +
                      weights['trend'] * vol_trend)

            volscore = max(min(volscore, 1.0), -1.0)

            # Log the volscore
            logger.debug(f"Volscore for {ticker} on {date}: {volscore:.2f}")

            return volscore

        except Exception as e:
            logger.error(f"Error calculating volscore: {str(e)}")
            return None

    def calculate_pnl(self, position, entry_date, current_date, ticker):
        try:
            current_spot = float(self.data[ticker]['Close'].loc[current_date])
            current_vol = float(self.data[ticker]['iv'].loc[current_date])
            entry_spot = float(self.data[ticker]['Close'].loc[entry_date])

            days_held = (current_date - entry_date).days
            days_to_expiry = (position.expiry - current_date).days

            time_factor = max(days_to_expiry / 30, 0)
            adjusted_vol = current_vol * (0.8 + 0.2 * time_factor)

            if days_to_expiry <= 0:
                if position.is_call:
                    current_price = max(current_spot - position.strike, 0)
                else:
                    current_price = max(position.strike - current_spot, 0)
            else:
                current_price = position.black_scholes(current_spot, days_to_expiry/365)
                max_price_move = position.entry_price * (1 + adjusted_vol * np.sqrt(days_held/252))
                min_price_move = position.entry_price * (1 - adjusted_vol * np.sqrt(days_held/252))
                current_price = min(max(current_price, min_price_move), max_price_move)

            # Apply slippage (0.5% instead of 5%)
            if current_price > position.entry_price:
                current_price -= self.slippage * position.entry_price  # Slippage against the move
            else:
                current_price += self.slippage * position.entry_price  # Slippage against the move

            pnl = (current_price - position.entry_price) * position.position_size * 100  # 100 multiplier for options

            # Deduct transaction costs (per contract)
            pnl -= self.transaction_cost * abs(position.position_size)

            # Logging detailed PnL information
            logger.debug(f"P&L Calculation for {ticker} on {current_date}:")
            logger.debug(f"  Entry Price: {position.entry_price:.2f}")
            logger.debug(f"  Current Price: {current_price:.2f}")
            logger.debug(f"  Position Size: {position.position_size}")
            logger.debug(f"  PnL: {pnl:.2f}")

            # Assertion to ensure PnL sign aligns with position direction
            if position.position_size > 0 and pnl < 0:
                logger.warning(f"Negative PnL for long position on {ticker} on {current_date}.")
            elif position.position_size < 0 and pnl > 0:
                logger.warning(f"Positive PnL for short position on {ticker} on {current_date}.")

            return pnl

        except Exception as e:
            logger.error(f"Error in P&L calculation: {str(e)}")
            return 0.0

    def calculate_position_size(self, portfolio_value, spot_price, vol, delta=1.0):
        """
        Calculates the number of contracts to trade based on current risk limits.

        Parameters:
        - portfolio_value (float): Current value of the portfolio.
        - spot_price (float): Current price of the underlying asset.
        - vol (float): Current implied volatility of the asset.
        - delta (float): Current portfolio delta.

        Returns:
        - int: Number of contracts to trade.
        """
        try:
            # Calculate the maximum allowable notional based on max_position_size
            max_notional = portfolio_value * self.risk_limits['max_position_size']

            # Adjust for portfolio delta to manage directional exposure
            if delta != 0:
                # Scale position size inversely with current portfolio delta
                position_size = max_notional * (1 - abs(delta)) / spot_price
            else:
                position_size = max_notional / spot_price

            # Ensure position size is within reasonable bounds
            # Set a more conservative maximum number of contracts
            min_contracts = 1
            max_contracts = 50  # Further reduced for controlled backtest

            num_contracts = int(position_size)
            num_contracts = max(min_contracts, min(num_contracts, max_contracts))

            logger.debug(f"Calculated Position Size: {num_contracts} contracts for {spot_price} spot price.")

            return num_contracts

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            # In case of any error, default to 1 contract
            return 1

    def calculate_portfolio_delta(self, date):
        total_delta = 0.0

        for ticker, positions in self.positions.items():
            spot = float(self.data[ticker]['Close'].loc[date])
            for pos in positions:
                days_to_expiry = (pos.expiry - date).days
                if days_to_expiry > 0:
                    greeks = pos.calculate_greeks(spot)
                    total_delta += greeks['delta']

        return total_delta

    def rebalance_delta(self, date, portfolio_value):
        try:
            portfolio_delta = self.calculate_portfolio_delta(date)

            if abs(portfolio_delta) > self.risk_limits['delta_limit']:
                logger.info(f"Rebalancing delta: {portfolio_delta:.3f}")

                for ticker, positions in self.positions.items():
                    spot = float(self.data[ticker]['Close'].loc[date])
                    vol = float(self.data[ticker]['iv'].loc[date])

                    position_deltas = []
                    for pos in positions:
                        days_to_expiry = (pos.expiry - date).days
                        if days_to_expiry > 0:
                            greeks = pos.calculate_greeks(spot)
                            delta = greeks['delta']
                            position_deltas.append((pos, delta))

                    if position_deltas:
                        # Select the position with the highest absolute delta to hedge
                        max_delta_pos = max(position_deltas, key=lambda x: abs(x[1]))
                        if abs(max_delta_pos[1]) > self.risk_limits['rebalance_threshold']:
                            hedge_size = self.calculate_position_size(portfolio_value, spot, vol, -max_delta_pos[1])

                            if max_delta_pos[0].is_call:
                                hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                      True, -hedge_size, spot, vol)
                            else:
                                hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                      False, -hedge_size, spot, vol)

                            hedge.entry_date = date
                            positions.append(hedge)
                            logger.info(f"Added delta hedge to {ticker}: {hedge_size} contracts")

        except Exception as e:
            logger.error(f"Error in delta rebalancing: {str(e)}")

    def adjust_risk_limits_growth(self, portfolio_value, initial_capital):
        """
        Adjusts risk limits based on portfolio growth to increase aggressiveness.

        Parameters:
        - portfolio_value (float): Current value of the portfolio.
        - initial_capital (float): Starting capital of the portfolio.
        """
        growth_factor = portfolio_value / initial_capital

        # Define growth milestones (e.g., every 20% growth)
        milestone = 0.20  # 20%

        # Calculate the number of milestones achieved
        milestones_achieved = int(growth_factor // milestone)

        # Increase max_position_size by 10% for every milestone, cap at 1.00 (100%)
        additional_size = 0.10 * milestones_achieved
        self.risk_limits['max_position_size'] = min(0.25 + additional_size, 1.00)

        # Increase max_portfolio_vol by 5% for every milestone, cap at 0.60 (60%)
        additional_vol = 0.05 * milestones_achieved
        self.risk_limits['max_portfolio_vol'] = min(0.30 + additional_vol, 0.60)

        # Adjust 'delta_limit' by 5% per milestone, cap at 0.60
        self.risk_limits['delta_limit'] = min(0.30 + 0.05 * milestones_achieved, 0.60)

        logger.debug(f"Adjusted risk limits based on growth: {growth_factor:.2f}x")

    def adjust_risk_limits_volatility(self, current_vol):
        """
        Adjusts risk limits based on current market volatility to enhance aggressiveness.

        Parameters:
        - current_vol (float): Current market volatility indicator (e.g., portfolio volatility).
        """
        # Define volatility thresholds
        high_vol_threshold = 0.30  # 30%
        low_vol_threshold = 0.15   # 15%

        if current_vol > high_vol_threshold:
            # In high volatility, slightly reduce position sizes to manage risk without being overly restrictive
            self.risk_limits['max_position_size'] = max(self.risk_limits['max_position_size'] * 0.95, 0.15)  # Reduce by 5%, floor at 15%

            # Minimal adjustment to stop loss to avoid premature exits
            self.risk_limits['position_stop_loss'] = min(self.risk_limits['position_stop_loss'] - 0.005, -0.20)  # Tighten by 0.5%

            # Slight increase in delta limit to maintain aggressiveness
            self.risk_limits['delta_limit'] = min(self.risk_limits['delta_limit'] + 0.02, 0.70)  # Increase by 2%

            logger.debug(f"High volatility detected: {current_vol:.2f}. Adjusted risk limits.")

        elif current_vol < low_vol_threshold:
            # In low volatility, increase position sizes to capitalize on stable conditions
            self.risk_limits['max_position_size'] = min(self.risk_limits['max_position_size'] * 1.25, 2.50)  # Increase by 25%, cap at 250%

            # Relax stop loss to allow positions to breathe
            self.risk_limits['position_stop_loss'] = max(self.risk_limits['position_stop_loss'] + 0.03, -0.03)  # Relax by 3%

            # Increase delta limit to enhance directional bets
            self.risk_limits['delta_limit'] = min(self.risk_limits['delta_limit'] + 0.05, 0.80)  # Increase by 5%

            logger.debug(f"Low volatility detected: {current_vol:.2f}. Adjusted risk limits.")

        else:
            logger.debug(f"Moderate volatility detected: {current_vol:.2f}. No adjustment.")

    def adjust_risk_limits_performance(self, portfolio_history):
        """
        Adjusts risk limits based on portfolio performance to scale aggressiveness.

        Parameters:
        - portfolio_history (list of floats): Historical portfolio values.
        """
        cumulative_return = (portfolio_history[-1] / portfolio_history[0]) - 1

        # Define performance thresholds
        aggressive_threshold = 1.00    # 100% return
        drawdown_threshold = -0.20     # 20% drawdown

        if cumulative_return > aggressive_threshold:
            # Upon reaching aggressive return milestones, further increase risk limits
            self.risk_limits['position_stop_loss'] = max(self.risk_limits['position_stop_loss'] - 0.03, -0.30)
            self.risk_limits['max_position_size'] = min(self.risk_limits['max_position_size'] * 1.10, 1.50)
            self.risk_limits['max_portfolio_vol'] = min(self.risk_limits['max_portfolio_vol'] * 1.10, 0.80)
            self.risk_limits['delta_limit'] = min(self.risk_limits['delta_limit'] * 1.10, 0.80)

            logger.debug(f"Cumulative return {cumulative_return:.2%} exceeded threshold. Adjusted risk limits.")

        elif cumulative_return < drawdown_threshold:
            # After a significant drawdown, tighten risk limits to protect capital
            self.risk_limits['position_stop_loss'] = max(self.risk_limits['position_stop_loss'] + 0.05, -0.15)
            self.risk_limits['max_position_size'] = max(self.risk_limits['max_position_size'] * 0.70, 0.05)
            self.risk_limits['max_portfolio_vol'] = max(self.risk_limits['max_portfolio_vol'] * 0.80, 0.20)
            self.risk_limits['delta_limit'] = max(self.risk_limits['delta_limit'] * 0.80, 0.10)

            logger.debug(f"Cumulative return {cumulative_return:.2%} below drawdown threshold. Adjusted risk limits.")

        else:
            logger.debug(f"Cumulative return {cumulative_return:.2%} within acceptable range. No adjustment.")

    def calculate_portfolio_volatility(self, date):
        """
        Calculates portfolio volatility based on current positions and their vegas.
        """
        total_vega = 0.0
        for ticker, positions in self.positions.items():
            spot = float(self.data[ticker]['Close'].loc[date])
            for pos in positions:
                greeks = pos.calculate_greeks(spot)
                total_vega += greeks['vega']

        # Assume portfolio volatility is proportional to total vega
        # This is a simplification and may need refinement
        portfolio_vol = total_vega * 0.01  # Scaling factor
        return portfolio_vol

    def backtest_strategy(self, initial_capital=1000000):
        portfolio_value = float(initial_capital)
        portfolio_history = [portfolio_value]
        trades_taken = 0

        # To collect volscore for plotting
        volscore_history = {ticker: [] for ticker in self.tickers}

        # Align all dataframes to have the same dates
        common_dates = self.data[self.tickers[0]].index
        for ticker in self.tickers + [self.sector_etf]:
            common_dates = common_dates.intersection(self.data[ticker].index)
        dates = common_dates.sort_values()

        min_portfolio_value = 1000  # Define a sensible minimum, e.g., $1,000

        for i, date in enumerate(dates[:-1]):
            daily_pnl = 0.0

            # Calculate current portfolio volatility
            current_vol = self.calculate_portfolio_volatility(date)

            # Adjust risk limits based on multiple factors
            self.adjust_risk_limits_growth(portfolio_value, initial_capital)
            self.adjust_risk_limits_volatility(current_vol)
            self.adjust_risk_limits_performance(portfolio_history)

            # Process existing positions
            for ticker in list(self.positions.keys()):
                try:
                    positions = self.positions[ticker]
                    for pos in positions[:]:  # Iterate over a copy to allow removal
                        days_held = (date - pos.entry_date).days
                        if days_held < self.risk_limits['min_holding_days']:
                            continue  # Do not calculate PnL yet

                        pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)

                        if portfolio_value > 0:
                            pnl_pct = pnl / portfolio_value
                        else:
                            pnl_pct = 0
                            logger.warning(f"Portfolio value is zero on {date}. Skipping P&L percentage calculation.")

                        if portfolio_value > 0:
                            if pnl_pct <= self.risk_limits['position_stop_loss']:
                                # Stop loss triggered
                                daily_pnl += pnl
                                positions.remove(pos)
                                logger.debug(f"Stop loss triggered for {ticker} on {date}. PnL: {pnl:.2f}")
                            elif pnl_pct >= self.risk_limits['min_profit_taking']:
                                # Profit target hit
                                daily_pnl += pnl
                                positions.remove(pos)
                                logger.debug(f"Profit target hit for {ticker} on {date}. PnL: {pnl:.2f}")

                        # Exit positions based on holding period
                        if days_held > self.risk_limits['max_holding_days']:
                            pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)
                            daily_pnl += pnl
                            positions.remove(pos)
                            logger.debug(f"Holding period exceeded for {ticker} on {date}. PnL: {pnl:.2f}")

                    if not positions:
                        del self.positions[ticker]

                except Exception as e:
                    logger.error(f"Error processing positions for {ticker}: {str(e)}")
                    continue

            # Rebalance delta
            self.rebalance_delta(date, portfolio_value)

            # Look for new positions
            if len(self.positions) < self.risk_limits['max_positions']:
                portfolio_delta = self.calculate_portfolio_delta(date)

                for ticker in self.tickers:
                    if ticker in self.positions:
                        continue

                    try:
                        spot = float(self.data[ticker]['Close'].loc[date])
                        vol = float(self.data[ticker]['iv'].loc[date])
                        volscore = self.calculate_volscore(ticker, date)

                        # Record volscore
                        volscore_history[ticker].append((date, volscore))

                        # Retrieve previous volscore
                        prev_volscore = self.prev_volscore[ticker]

                        # Update current volscore
                        self.prev_volscore[ticker] = volscore

                        # Trade entry condition: relaxed thresholds (single day for testing)
                        if (volscore is not None and
                            ((volscore > 0.10) or
                             (volscore < -0.10))):

                            position_size = self.calculate_position_size(portfolio_value, spot, vol, -portfolio_delta)

                            if volscore > 0.10:  # Short vol
                                call = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                     True, -position_size, spot, vol)
                                put = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                    False, -position_size, spot, vol)

                                if call.entry_price > 0 and put.entry_price > 0:
                                    call.entry_date = put.entry_date = date
                                    self.positions[ticker] = [call, put]
                                    trades_taken += 1
                                    logger.debug(f"Opened short vol positions for {ticker}: {position_size} contracts each.")

                            elif volscore < -0.10:  # Long vol
                                # Using ATM options for balanced exposure
                                long_call = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                          True, position_size, spot, vol)
                                short_call = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                           True, -position_size, spot, vol)

                                if long_call.entry_price > 0 and short_call.entry_price > 0:
                                    long_call.entry_date = short_call.entry_date = date
                                    self.positions[ticker] = [long_call, short_call]
                                    trades_taken += 1
                                    logger.debug(f"Opened long vol positions for {ticker}: {position_size} contracts long and short.")

                    except Exception as e:
                        logger.error(f"Error entering position for {ticker}: {str(e)}")
                        continue

            # Update portfolio value
            portfolio_value_before_pnl = portfolio_value
            portfolio_value = max(portfolio_value + daily_pnl, 0)
            portfolio_history.append(portfolio_value)

            # Check for maximum daily loss
            daily_return = (portfolio_value - portfolio_value_before_pnl) / portfolio_value_before_pnl
            max_daily_loss = -0.30  # 30% loss per day

            if daily_return <= max_daily_loss:
                logger.warning(f"Maximum daily loss of {max_daily_loss*100}% exceeded on {date}. Halting trading.")
                break  # Exit the backtest loop

            # Check for minimum portfolio value to prevent unnecessary calculations
            if portfolio_value < min_portfolio_value:
                logger.warning(f"Portfolio value {portfolio_value:.2f} dropped below minimum threshold on {date}. Halting trading.")
                break  # Exit the backtest loop

            # Optional: Minimal daily summaries can be printed if needed
            # Example:
            # if i % 100 == 0:
            #     print(f"Date: {date}, Portfolio Value: {portfolio_value}")

        # Backtest completion summary
        print(f"\nBacktest complete.")
        print(f"Total trades: {trades_taken}")

        # Performance metrics
        portfolio_series = pd.Series(portfolio_history, index=dates[:len(portfolio_history)])
        returns = portfolio_series.pct_change().dropna()
        cumulative_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        max_drawdown = (portfolio_series / portfolio_series.cummax() - 1).min()

        print("\nPerformance Summary:")
        print(f"Total Return: {cumulative_return:.2%}")
        print(f"Annual Volatility: {annual_volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")

        print("\nStrategy Performance:")
        print(f"Total Return: {cumulative_return:.2%}")
        print(f"Annual Sharpe: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Final Portfolio Value: {portfolio_value:.2f}")

        # Return volscore_history for plotting
        return portfolio_series, volscore_history

    def plot_portfolio(self, portfolio_series):
        plt.figure(figsize=(12,6))
        plt.plot(portfolio_series.index, portfolio_series.values, label='Portfolio Value')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_volscore(self, volscore_history):
        plt.figure(figsize=(12,6))
        for ticker, scores in volscore_history.items():
            if scores:
                dates, scores_values = zip(*scores)
                plt.plot(dates, scores_values, label=f"{ticker} Volscore")
        plt.axhline(0.10, color='green', linestyle='--', label='Long Vol Threshold (0.10)')
        plt.axhline(-0.10, color='red', linestyle='--', label='Short Vol Threshold (-0.10)')
        plt.title('Volatility Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Volscore')
        plt.legend()
        plt.grid(True)
        plt.show()

# Unit Test for PnL Calculation
def test_calculate_pnl():
    backtester = VolatilityArbitrageBacktester(tickers=['AAPL'], sector_etf='XLK', transaction_cost=1.0, slippage=0.005)
    backtester.data['AAPL'] = pd.DataFrame({
        'Close': {
            pd.Timestamp('2023-01-01'): 100,
            pd.Timestamp('2023-01-02'): 105,
        },
        'iv': {
            pd.Timestamp('2023-01-01'): 0.2,
            pd.Timestamp('2023-01-02'): 0.2,
        }
    })

    # Create a long call position
    position = SimulatedOption(strike=100, expiry=pd.Timestamp('2023-02-01'), is_call=True, position_size=10, underlying_price=100, volatility=0.2)
    position.entry_date = pd.Timestamp('2023-01-01')

    # Simulate a price increase
    pnl = backtester.calculate_pnl(position, pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), 'AAPL')
    print(f"P&L for long call position: {pnl:.2f}")

    assert pnl > 0, "PnL should be positive for a profitable long call position."

# Run the test
test_calculate_pnl()

# Main Execution
if __name__ == "__main__":
    # Initialize backtester with multiple tickers and an extended date range
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'INTC', 'AMD']
    sector_etf = 'XLK'
    backtester = VolatilityArbitrageBacktester(tickers, sector_etf, transaction_cost=1.0, slippage=0.005)

    # Modify risk limits for controlled backtest
    backtester.risk_limits = {
        'position_stop_loss': -0.05,      # 5% stop loss
        'min_profit_taking': 0.08,        # 8% profit target
        'max_position_size': 0.05,        # 5% position size
        'max_portfolio_vol': 0.10,        # 10% target vol
        'max_positions': 5,                # Increased maximum positions
        'min_entry_vol': 0.15,             # Minimum IV for entry
        'max_entry_vol': 0.75,             # Maximum IV for entry
        'min_holding_days': 1,             # Minimum holding period
        'max_holding_days': 21,            # Maximum holding period
        'delta_limit': 0.10,               # Maximum portfolio delta
        'rebalance_threshold': 0.10        # Rebalance threshold
    }

    # Fetch data for a longer period (e.g., January 1, 2018, to December 31, 2020)
    backtester.fetch_data(start_date='2018-01-01', end_date='2020-12-31')

    # Run the backtest
    portfolio_series, volscore_history = backtester.backtest_strategy(initial_capital=1000000)

    # Plot the portfolio value over time
    backtester.plot_portfolio(portfolio_series)

    # Plot the volatility scores
    backtester.plot_volscore(volscore_history)
