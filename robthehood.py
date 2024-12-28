"""
reetale
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
import matplotlib.pyplot as plt

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
            r = max(min(self.rate, 0.1), -0.1)
            t = max(min(float(t), 2.0), 1/365)
            v = self.volatility

            # Calculate d1 and d2
            d1 = (np.log(S/K) + (r + v**2/2)*t) / (v*np.sqrt(t))
            d2 = d1 - v*np.sqrt(t)

            # Calculate price with bounds
            if self.is_call:
                price = S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)
                price = min(price, S)
            else:
                price = K*np.exp(-r*t)*norm.cdf(-d2) - S*norm.cdf(-d1)
                price = min(price, K)

            # Ensure price >= intrinsic value
            intrinsic = max(S - K, 0) if self.is_call else max(K - S, 0)
            price = max(price, intrinsic)

            return max(price, 0.01)

        except Exception as e:
            print(f"Error in Black-Scholes: {str(e)}")
            return 0.0

    def calculate_greeks(self, S, t=30/365):
        try:
            K = self.strike
            r = self.rate
            v = self.volatility

            d1 = (np.log(S/K) + (r + v**2/2)*t) / (v*np.sqrt(t))
            d2 = d1 - v*np.sqrt(t)

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
            print(f"Error calculating Greeks: {str(e)}")
            return {'delta': 0, 'gamma': 0, 'vega': 0}

class VolatilityArbitrageBacktester:
    def __init__(self, tickers, sector_etf, transaction_cost=1.0, slippage=0.05):
        self.tickers = tickers
        self.sector_etf = sector_etf
        self.data = {}
        self.positions = {}

        self.transaction_cost = transaction_cost  # Flat fee per trade
        self.slippage = slippage  # Percentage slippage per trade

        self.prev_volscore = {ticker: None for ticker in self.tickers}  # Initialize previous volscore

        self.risk_limits = {
            'position_stop_loss': -0.05,      # 5% stop loss
            'min_profit_taking': 0.08,        # 8% profit target
            'max_position_size': 0.10,        # 10% position size
            'max_portfolio_vol': 0.15,        # 15% target vol
            'max_positions': 10,               # Maximum positions
            'min_entry_vol': 0.15,           # Minimum IV for entry
            'max_entry_vol': 0.75,           # Maximum IV for entry
            'min_holding_days': 3,           # Minimum holding period
            'max_holding_days': 21,          # Maximum holding period
            'delta_limit': 0.15,             # Maximum portfolio delta
            'rebalance_threshold': 0.05      # Rebalance threshold
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

            hist['iv'] = hist['rv_20'] * np.random.uniform(0.9, 1.1, len(hist))
            hist['iv'] = hist['iv'].clip(0.05, 1.0)

            # Drop rows with NaN values to ensure complete data
            hist.dropna(inplace=True)

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

            return max(min(volscore, 1.0), -1.0)

        except Exception as e:
            print(f"Error calculating volscore: {str(e)}")
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

            # Apply slippage
            current_price *= (1 + self.slippage * np.sign(current_price - position.entry_price))

            pnl = (current_price - position.entry_price) * position.position_size * 100

            # Deduct transaction costs
            pnl -= self.transaction_cost * position.position_size

            max_theoretical = current_spot if position.is_call else position.strike
            max_pnl = abs(max_theoretical * position.position_size * 100)
            pnl = max(min(pnl, max_pnl), -max_pnl)

            return pnl

        except Exception as e:
            print(f"Error in P&L calculation: {str(e)}")
            return 0.0

    def calculate_position_size(self, portfolio_value, spot_price, vol, delta=1.0):
        try:
            # Validate inputs
            if np.isnan(vol) or np.isnan(spot_price):
                print("Volatility or Spot Price is NaN. Skipping position sizing.")
                return 1  # Default to minimum position size or skip

            max_notional = portfolio_value * self.risk_limits['max_position_size']

            vol_scalar = 0.4 / max(vol, 0.1)
            delta_scalar = min(1.0, 1.5 - abs(delta))

            adjusted_notional = max_notional * vol_scalar * delta_scalar

            contract_value = spot_price * 100
            num_contracts = int(adjusted_notional / contract_value)

            num_contracts = max(min(num_contracts, 8), 1)

            total_exposure = num_contracts * contract_value
            if total_exposure > portfolio_value * 0.04:
                num_contracts = int((portfolio_value * 0.04) / contract_value)

            return max(num_contracts, 1)

        except Exception as e:
            print(f"Error in position sizing: {str(e)}")
            return 1

    def calculate_portfolio_delta(self, date):
        total_delta = 0.0

        for ticker, positions in self.positions.items():
            spot = float(self.data[ticker]['Close'].loc[date])
            for pos in positions:
                days_to_expiry = (pos.expiry - date).days
                if days_to_expiry > 0:
                    delta = pos.calculate_greeks(spot)['delta']
                    total_delta += delta

        return total_delta

    def rebalance_delta(self, date, portfolio_value):
        try:
            portfolio_delta = self.calculate_portfolio_delta(date)

            if abs(portfolio_delta) > self.risk_limits['delta_limit']:
                print(f"Rebalancing delta: {portfolio_delta:.3f}")

                for ticker, positions in self.positions.items():
                    spot = float(self.data[ticker]['Close'].loc[date])
                    vol = float(self.data[ticker]['iv'].loc[date])

                    position_deltas = []
                    for pos in positions:
                        days_to_expiry = (pos.expiry - date).days
                        if days_to_expiry > 0:
                            delta = pos.calculate_greeks(spot)['delta']
                            position_deltas.append((pos, delta))

                    if position_deltas:
                        max_delta_pos = max(position_deltas, key=lambda x: abs(x[1]))
                        if abs(max_delta_pos[1]) > self.risk_limits['rebalance_threshold']:
                            new_size = self.calculate_position_size(portfolio_value, spot, vol, -max_delta_pos[1])

                            if max_delta_pos[0].is_call:
                                hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                      True, -new_size, spot, vol)
                            else:
                                hedge = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                      False, -new_size, spot, vol)

                            hedge.entry_date = date
                            positions.append(hedge)
                            print(f"Added delta hedge to {ticker}")

        except Exception as e:
            print(f"Error in delta rebalancing: {str(e)}")

    def backtest_strategy(self, initial_capital=1000000):
        portfolio_value = float(initial_capital)
        portfolio_history = [portfolio_value]
        dates = self.data[self.tickers[0]].index
        trades_taken = 0

        print("\nStarting backtest...")

        # Align all dataframes to have the same dates
        common_dates = self.data[self.tickers[0]].index
        for ticker in self.tickers + [self.sector_etf]:
            common_dates = common_dates.intersection(self.data[ticker].index)
        dates = common_dates.sort_values()

        for i, date in enumerate(dates[:-1]):
            daily_pnl = 0.0

            if i % 20 == 0:
                print(f"\nDate: {date.date()}")
                print(f"Portfolio value: ${portfolio_value:,.2f}")
                print(f"Active positions: {list(self.positions.keys())}")

            # Process existing positions
            for ticker in list(self.positions.keys()):
                try:
                    positions = self.positions[ticker]
                    for pos in positions[:]:  # Iterate over a copy to allow removal
                        pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)
                        pnl_pct = pnl / portfolio_value
                        if pnl_pct <= self.risk_limits['position_stop_loss']:
                            print(f"Stop loss triggered for {ticker}: {pnl_pct:.2%}")
                            daily_pnl += pnl
                            positions.remove(pos)
                        elif pnl_pct >= self.risk_limits['min_profit_taking']:
                            print(f"Profit target hit for {ticker}: {pnl_pct:.2%}")
                            daily_pnl += pnl
                            positions.remove(pos)

                        # Exit positions based on holding period
                        days_held = (date - pos.entry_date).days
                        if days_held > self.risk_limits['max_holding_days']:
                            print(f"Exiting position in {ticker} after holding for {days_held} days.")
                            pnl = self.calculate_pnl(pos, pos.entry_date, date, ticker)
                            daily_pnl += pnl
                            positions.remove(pos)

                    if not positions:
                        del self.positions[ticker]

                except Exception as e:
                    print(f"Error processing position {ticker}: {str(e)}")
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

                        # Retrieve previous volscore
                        prev_volscore = self.prev_volscore[ticker]

                        # Update current volscore
                        self.prev_volscore[ticker] = volscore

                        # Check persistence: current and previous volscore exceed thresholds
                        if volscore is None or spot <= 10 or prev_volscore is None:
                            continue

                        if (volscore > 0.25 and prev_volscore > 0.25) or (volscore < -0.25 and prev_volscore < -0.25):
                            position_size = self.calculate_position_size(portfolio_value, spot, vol, -portfolio_delta)

                            if volscore > 0.25:  # Short vol
                                call = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                    True, -position_size, spot, vol)
                                put = SimulatedOption(spot, date + pd.Timedelta(days=30),
                                                    False, -position_size, spot, vol)

                                if call.entry_price > 0 and put.entry_price > 0:
                                    call.entry_date = put.entry_date = date
                                    self.positions[ticker] = [call, put]
                                    trades_taken += 1
                                    print(f"\nOpened straddle in {ticker}")
                                    print(f"Entry prices - Call: ${call.entry_price:.2f}, Put: ${put.entry_price:.2f}")
                                    print(f"Size: {position_size} contracts")
                                    print(f"Delta: {call.calculate_greeks(spot)['delta'] + put.calculate_greeks(spot)['delta']:.3f}")

                            elif volscore < -0.25:  # Long vol
                                long_call = SimulatedOption(spot*0.95, date + pd.Timedelta(days=30),
                                                          True, position_size, spot, vol)
                                short_call = SimulatedOption(spot*1.05, date + pd.Timedelta(days=30),
                                                          True, -position_size, spot, vol)

                                if long_call.entry_price > 0 and short_call.entry_price > 0:
                                    long_call.entry_date = short_call.entry_date = date
                                    self.positions[ticker] = [long_call, short_call]
                                    trades_taken += 1
                                    print(f"\nOpened call spread in {ticker}")
                                    print(f"Entry prices - Long: ${long_call.entry_price:.2f}, Short: ${short_call.entry_price:.2f}")
                                    print(f"Size: {position_size} contracts")
                                    print(f"Delta: {long_call.calculate_greeks(spot)['delta'] + short_call.calculate_greeks(spot)['delta']:.3f}")

                    except Exception as e:
                        print(f"Error processing {ticker}: {str(e)}")
                        continue

            # Update portfolio value
            portfolio_value = max(portfolio_value + daily_pnl, 0)
            portfolio_history.append(portfolio_value)

            # Print daily summary
            if abs(daily_pnl) > 0:
                print(f"\nDaily P&L: ${daily_pnl:,.2f}")
                print(f"Portfolio Value: ${portfolio_value:,.2f}")

                # Print risk metrics
                portfolio_delta = self.calculate_portfolio_delta(date)
                print(f"Portfolio Delta: {portfolio_delta:.3f}")

                # Check any positions near expiry
                """
                for ticker, positions in self.positions.items():
                    for pos in positions:
                        days_to_expiry = (pos.expiry - date).days
                        if days_to_expiry <= 5:
                            print(f"Warning: {ticker} position near expiry ({days_to_expiry} days)")
                """

        print(f"\nBacktest complete.")
        print(f"Total trades: {trades_taken}")

        # Performance metrics
        portfolio_series = pd.Series(portfolio_history, index=dates)
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

        return portfolio_series

# Example usage
if __name__ == "__main__":
    tickers = ['MSFT', 'AAPL', 'TSLA', 'AMZN', 'GOOGL']
    sector_etf = 'XLK'

    backtester = VolatilityArbitrageBacktester(tickers, sector_etf)
    backtester.fetch_data('2021-01-01', '2024-12-03')

    portfolio_values = backtester.backtest_strategy(initial_capital=10000)
    returns = portfolio_values.pct_change().dropna()

    print("\nStrategy Performance:")
    print(f"Total Return: {(portfolio_values.iloc[-1]/portfolio_values.iloc[0] - 1):.2%}")
    if returns.std() != 0:
        print(f"Annual Sharpe: {returns.mean()/returns.std() * np.sqrt(252):.2f}")
    print(f"Max Drawdown: {(portfolio_values/portfolio_values.cummax() - 1).min():.2%}")

    # Plot portfolio value over time
    plt.figure(figsize=(12,6))
    portfolio_values.plot()
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()
