"""
iv
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import pytz
import json
from pathlib import Path
from scipy.optimize import minimize

from scipy.interpolate import griddata, NearestNDInterpolator
from scipy.spatial import qhull
from scipy.ndimage import gaussian_filter
from typing import Optional, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ImpliedVolatilityAnalysis:
    """
    A class for analyzing and visualizing implied volatility surfaces.

    This class provides methods for collecting options data, calculating implied
    volatilities, storing historical data, and creating interactive visualizations
    of volatility surfaces.

    Attributes:
        ticker (str): The stock ticker symbol
        data_dir (Path): Directory for storing historical volatility surface data
        risk_free_rate (float): Risk-free rate used in calculations
    """

    def __init__(self, ticker: str, data_dir: str = 'vol_surface_history',
                 risk_free_rate: float = 0.05):
        """
        Initialize the ImpliedVolatilityAnalysis instance.

        Args:
            ticker: Stock ticker symbol
            data_dir: Directory for storing historical data
            risk_free_rate: Risk-free rate for calculations (default: 5%)
        """
        self.ticker = ticker.upper()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.risk_free_rate = risk_free_rate
        self.et_tz = pytz.timezone('US/Eastern')
        # Note: We ignore dividends/American style early exercise differences for simplicity.

    def _black_scholes_call_price(self, S, K, T, sigma):
        """
        Black-Scholes call option price.
        """
        try:
            d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return S*norm.cdf(d1) - K*np.exp(-self.risk_free_rate*T)*norm.cdf(d2)
        except:
            return np.nan

    def _black_scholes_put_price(self, S, K, T, sigma):
        """
        Black-Scholes put option price.
        """
        try:
            d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            return K*np.exp(-self.risk_free_rate*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        except:
            return np.nan

    def _vega(self, S, K, T, sigma):
        """
        Vega of a call or put in the Black-Scholes model (same for calls/puts).
        """
        try:
            d1 = (np.log(S/K) + (self.risk_free_rate + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            return S * np.sqrt(T) * norm.pdf(d1)
        except:
            return np.nan

    def _implied_vol_newton(self, price_func, S, K, T, market_price, tol=1e-4, max_iter=100):
        """
        Generic Newton iteration for implied vol based on a price function (call or put).
        """
        if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
            return np.nan

        sigma = 0.3  # initial guess
        for _ in range(max_iter):
            price = price_func(S, K, T, sigma)
            if np.isnan(price):
                return np.nan
            diff = market_price - price
            if abs(diff) < tol:
                return sigma
            v = self._vega(S, K, T, sigma)
            if abs(v) < 1e-10:
                return np.nan
            sigma_next = sigma + diff / v
            if sigma_next <= 0:
                return np.nan
            sigma = sigma_next
        return np.nan

    def _black_scholes_implied_vol(self, S: float, K: float, T: float,
                                  market_price: float, is_call: bool) -> float:
        """
        Calculate implied volatility for a single call/put using the Black-Scholes model.
        """
        price_func = self._black_scholes_call_price if is_call else self._black_scholes_put_price
        return self._implied_vol_newton(price_func, S, K, T, market_price)

    def _get_time_to_expiry(self, expiry_date: str) -> float:
        """
        Calculate time to expiry in years.
        """
        exp_date_naive = datetime.strptime(expiry_date, '%Y-%m-%d')
        exp_date_naive = exp_date_naive.replace(hour=16, minute=0)
        exp_date = self.et_tz.localize(exp_date_naive)
        now = datetime.now(self.et_tz)
        sec_to_expiry = (exp_date - now).total_seconds()
        return max(sec_to_expiry / (365.25 * 24 * 3600), 0)

    def calculate_surface(self) -> pd.DataFrame:
        """
        Calculate the current implied volatility surface for calls and puts.
        """
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period='1d')
        if hist.empty:
            return pd.DataFrame()

        current_price = hist['Close'].iloc[-1]
        data = []
        now = datetime.now(self.et_tz)

        # Check that 'stock.options' attribute exists
        if not hasattr(stock, 'options'):
            return pd.DataFrame()

        for exp_date in stock.options:
            T = self._get_time_to_expiry(exp_date)
            if T <= 0:
                continue

            try:
                option_data = stock.option_chain(exp_date)
            except Exception:
                continue

            # We'll iterate over calls and puts
            for opt_type, df_options in [('call', option_data.calls), ('put', option_data.puts)]:
                is_call = (opt_type == 'call')

                for _, row in df_options.iterrows():
                    strike = float(row['strike'])
                    last_price = float(row['lastPrice']) if row['lastPrice'] > 0 else None
                    bid = float(row['bid']) if row['bid'] > 0 else None
                    ask = float(row['ask']) if row['ask'] > 0 else None
                    mid_price = (bid + ask)/2 if (bid and ask) else None

                    use_price = mid_price or last_price
                    if use_price is None or use_price <= 0:
                        continue

                    iv = self._black_scholes_implied_vol(
                        S=current_price,
                        K=strike,
                        T=T,
                        market_price=use_price,
                        is_call=is_call
                    )

                    exp_date_naive = datetime.strptime(exp_date, '%Y-%m-%d').replace(hour=16, minute=0)
                    exp_date_local = self.et_tz.localize(exp_date_naive)

                    now_aligned = now.replace(hour=16, minute=0)
                    sec_to_expiry = (exp_date_local - now_aligned).total_seconds()

                    days_float = max(sec_to_expiry / (24.0 * 3600.0), 0)
                    days_int = int(days_float)

                    data.append({
                        'option_type': opt_type,
                        'strike': strike,
                        'days_float': days_float,
                        'days_int': days_int,
                        'impl_vol': iv if iv is not None else np.nan,
                        'last_price': last_price if last_price is not None else np.nan,
                        'bid': bid if bid is not None else np.nan,
                        'ask': ask if ask is not None else np.nan,
                        'mid_price': mid_price if mid_price is not None else np.nan,
                        'expiry': exp_date,
                        'stock_price': float(current_price),
                        'volume': int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None,
                        'open_interest': int(row['openInterest']) if 'openInterest' in row and pd.notna(row['openInterest']) else None,
                        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S')
                    })

        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def store_surface(self, df: pd.DataFrame) -> None:
        """
        Store the volatility surface data.
        """
        if df.empty:
            return

        now = datetime.now(self.et_tz)
        df_dict = df.copy()
        if 'timestamp' in df_dict.columns:
            df_dict['timestamp'] = df_dict['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        data = {
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'stock_price': float(df['stock_price'].iloc[0]),
            'surface_data': df_dict.to_dict(orient='records')
        }

        filename = self.data_dir / f"{self.ticker}_{now.strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_historical_data(self, days_back: int = 5) -> Optional[pd.DataFrame]:
        """
        Load historical volatility surface data.
        """
        start_date = datetime.now(self.et_tz) - timedelta(days=days_back)
        all_data = []

        for file in self.data_dir.glob(f"{self.ticker}_*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                file_date = datetime.strptime(data['timestamp'], '%Y-%m-%d %H:%M:%S')
                file_date = self.et_tz.localize(file_date)
                if file_date >= start_date:
                    df = pd.DataFrame(data['surface_data'])
                    df['timestamp'] = file_date
                    all_data.append(df)
            except Exception:
                continue

        return pd.concat(all_data, ignore_index=True) if all_data else None

    def prepare_visualization_data(self, df: pd.DataFrame,
                                   params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare data for visualization with optional filtering.
        """
        if params is None:
            params = {
                'min_volume': 0,
                'max_spread': float('inf'),
                'use_price': 'mid'
            }

        viz_df = df.copy()
        viz_df['moneyness'] = viz_df['strike'] / viz_df['stock_price']

        # Volume filter
        if params['min_volume'] > 0:
            viz_df = viz_df[viz_df['volume'] >= params['min_volume']]

        # Bid-ask spread filter
        if params['max_spread'] < float('inf'):
            spread_pct = (viz_df['ask'] - viz_df['bid']) / viz_df['bid']
            viz_df = viz_df[spread_pct <= params['max_spread']]

        # Drop rows with no implied vol
        viz_df = viz_df.dropna(subset=['impl_vol'])

        return viz_df

    def _safe_griddata(self, points, values, xi, method='cubic', fill_value=np.nan):
        """
        A safe wrapper around griddata that falls back if QhullError occurs.
        """
        try:
            return griddata(points, values, xi, method=method, fill_value=fill_value)
        except qhull.QhullError:
            # Fallback to 'linear' or even 'nearest'
            try:
                return griddata(points, values, xi, method='linear', fill_value=fill_value)
            except qhull.QhullError:
                return griddata(points, values, xi, method='nearest', fill_value=fill_value)

    def interpolate_surface(self,
                            points: np.ndarray,
                            values: np.ndarray,
                            grid_points: Tuple[np.ndarray, np.ndarray],
                            params: Optional[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        Interpolate surface data with directional smoothing, safely.
        """
        if params is None:
            params = {'strike_smooth': 2.0, 'time_smooth': 0.5}

        try:
            unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
            unique_values = values[unique_indices]
            if len(unique_points) < 4:
                return None

            # Safely attempt 'cubic', then 'linear', then 'nearest' if needed
            rough_surface = self._safe_griddata(
                unique_points,
                unique_values,
                grid_points,
                method='cubic',
                fill_value=np.nan
            )

            smooth_surface = gaussian_filter(
                rough_surface,
                sigma=[params['time_smooth'], params['strike_smooth']]
            )

            nan_mask = np.isnan(smooth_surface)
            if np.any(nan_mask):
                # Nearest fallback for remaining NaNs
                nearest_interp = self._safe_griddata(
                    unique_points,
                    unique_values,
                    grid_points,
                    method='nearest',
                    fill_value=np.nan
                )
                smooth_surface[nan_mask] = nearest_interp[nan_mask]

            return smooth_surface

        except:
            return None

    def _svi_total_variance(self, k, a, b, rho, m, sigma):
        """
        SVI total variance function: w(k) = a + b*(rho*(k-m) + sqrt((k-m)**2 + sigma**2))
        """
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def _calibrate_svi(self, k_data, iv_data, T):
        """
        Calibrate SVI parameters (a,b,rho,m,sigma) to the given log-moneyness (k_data)
        and implied vols (iv_data) at time to expiry T (in years).
        """
        w_data = (iv_data**2) * T

        def objective(params):
            a, b, rho, m, sigma = params
            w_model = self._svi_total_variance(k_data, a, b, rho, m, sigma)
            return np.mean((w_data - w_model)**2)

        initial_guess = [0.01, 0.1, 0.0, 0.0, 0.1]
        bounds = [(1e-9, None), (1e-9, None), (-0.999, 0.999), (None, None), (1e-9, None)]
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        return result.x  # [a, b, rho, m, sigma]

    def interpolate_surface_svi(self,
                                df: pd.DataFrame,
                                grid_points: Tuple[np.ndarray, np.ndarray],
                                smooth_params: Optional[Dict] = None):
        """
        Produce a smoother implied-vol surface via SVI on each day_bucket.
        Falls back to simpler interpolation if SVI fails or not enough data.
        """
        df = df.copy()
        df['day_bucket'] = df['days_float'].round()

        if smooth_params is None:
            smooth_params = {'z_smooth': 0.5}

        moneyness_mesh, days_mesh = grid_points
        output_surface = np.full_like(moneyness_mesh, np.nan, dtype=float)

        # Group by day_bucket
        for bucket_val, bucket_df in df.groupby('day_bucket'):
            row_mask = (np.round(days_mesh) == bucket_val)
            if not row_mask.any():
                continue

            T = bucket_df['days_float'].mean() / 365.25
            if T <= 0:
                continue

            k_data = np.log(bucket_df['moneyness'].values)
            iv_data = bucket_df['impl_vol'].values

            # Attempt SVI if at least 5 distinct log-moneyness points
            if len(np.unique(k_data)) >= 5:
                try:
                    a, b, rho, m, sigma = self._calibrate_svi(k_data, iv_data, T)
                    k_grid = np.log(moneyness_mesh[row_mask])
                    w_grid = self._svi_total_variance(k_grid, a, b, rho, m, sigma)
                    iv_grid = np.sqrt(np.maximum(w_grid / T, 1e-12))
                    output_surface[row_mask] = iv_grid
                    continue
                except:
                    pass  # fallback to simpler interpolation

            # If SVI not possible, do local interpolation for that bucket
            coords = np.column_stack((bucket_df['moneyness'], bucket_df['days_float']))
            vals = bucket_df['impl_vol']
            if len(np.unique(coords, axis=0)) < 3:
                # fallback directly to nearest if we can't do linear
                local_iv = self._safe_griddata(coords, vals,
                                               (moneyness_mesh[row_mask], days_mesh[row_mask]),
                                               method='nearest', fill_value=np.nan)
            else:
                local_iv = self._safe_griddata(coords, vals,
                                               (moneyness_mesh[row_mask], days_mesh[row_mask]),
                                               method='linear', fill_value=np.nan)
            output_surface[row_mask] = local_iv

        # Fill any leftover NaNs across the whole surface
        nan_mask = np.isnan(output_surface)
        if np.any(nan_mask):
            valid_idx = ~nan_mask
            if valid_idx.sum() > 4:
                coords = np.column_stack((moneyness_mesh[valid_idx], days_mesh[valid_idx]))
                vals = output_surface[valid_idx]
                # Attempt cubic fallback
                filled_vals = self._safe_griddata(coords, vals,
                                                  (moneyness_mesh, days_mesh),
                                                  method='cubic', fill_value=np.nan)
                output_surface[nan_mask] = filled_vals[nan_mask]

        # Final smoothing
        output_surface = gaussian_filter(output_surface,
                                         sigma=smooth_params.get('z_smooth', 0.5))

        if np.isnan(output_surface).all():
            return None

        return output_surface

    def visualize(self, days_back: int = 5, viz_params: Optional[Dict] = None) -> None:
        """
        Create interactive visualization of volatility surface evolution.
        """
        # 1) Calculate/store latest data
        current_df = self.calculate_surface()
        if not current_df.empty:
            self.store_surface(current_df)

        # 2) Load historical
        historical_df = self.load_historical_data(days_back)
        if historical_df is None or historical_df.empty:
            print("No historical data available. Starting to track changes now.")
            return

        # 3) Prep
        viz_df = self.prepare_visualization_data(historical_df, viz_params)
        timestamps = sorted(viz_df['timestamp'].unique())
        if len(timestamps) < 2:
            print("Not enough historical data points to animate.")
            return

        # 4) Ensure days_float
        if 'days_float' not in viz_df.columns or viz_df['days_float'].isna().all():
            if 'days_int' in viz_df.columns:
                viz_df['days_float'] = viz_df['days_int'].astype(float)
            else:
                print("No valid days to expiry found in data.")
                return

        # 5) Build interpolation grid
        moneyness_min, moneyness_max = viz_df['moneyness'].min(), viz_df['moneyness'].max()
        days_min, days_max = viz_df['days_float'].min(), viz_df['days_float'].max()
        days_range = np.linspace(days_min, days_max, 50)
        moneyness_range = np.linspace(moneyness_min, moneyness_max, 50)
        moneyness_mesh, days_mesh = np.meshgrid(moneyness_range, days_range)
        grid_points = (moneyness_mesh, days_mesh)

        # 6) Gather surfaces
        all_surfaces = {}
        global_zmin, global_zmax = np.inf, -np.inf

        for ts in timestamps:
            df_slice = viz_df[viz_df['timestamp'] == ts].copy()
            if df_slice.empty:
                continue

            surf = self.interpolate_surface_svi(df_slice, grid_points, viz_params)
            if surf is None or np.isnan(surf).all():
                continue

            all_surfaces[ts] = surf
            zmin_local = np.nanmin(surf)
            zmax_local = np.nanmax(surf)
            if zmin_local < global_zmin:
                global_zmin = zmin_local
            if zmax_local > global_zmax:
                global_zmax = zmax_local

        if not all_surfaces:
            print("Could not interpolate any surfaces for animation. Probably not enough data per timestamp.")
            return

        # 7) Plot
        last_ts = timestamps[-1]
        last_surface = all_surfaces[last_ts]

        fig = go.Figure()
        surface_trace = go.Surface(
            x=moneyness_mesh,
            y=days_mesh,
            z=last_surface,
            colorscale='Viridis',
            name='IV Surface'
        )
        fig.add_trace(surface_trace)

        frames = []
        for ts in timestamps:
            if ts not in all_surfaces:
                continue
            frames.append(
                go.Frame(
                    name=ts.strftime('%Y-%m-%d %H:%M:%S'),
                    data=[go.Surface(
                        x=moneyness_mesh,
                        y=days_mesh,
                        z=all_surfaces[ts],
                        colorscale='Viridis'
                    )]
                )
            )
        fig.frames = frames

        fig.update_layout(
            title=f'{self.ticker} SVI-Based Volatility Animation (Fixed Z-axis)',
            scene=dict(
                xaxis_title='Moneyness (Strike/Spot)',
                yaxis_title='Days to Expiry',
                zaxis=dict(
                    title='Implied Vol',
                    tickformat='.0%',
                    range=[global_zmin, global_zmax]
                )
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [
                            None,
                            {
                                'frame': {'duration': 800, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 500}
                            }
                        ]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [
                            [None],
                            {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ]
                    }
                ]
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 18},
                    'prefix': 'Time: ',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [
                            [frame.name],
                            {
                                'frame': {'duration': 300, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 300}
                            }
                        ],
                        'label': frame.name,
                        'method': 'animate'
                    }
                    for frame in frames
                ]
            }]
        )

        fig.show()

    def get_volatility_smile(self, expiry_date: str) -> pd.DataFrame:
        """
        Get the volatility smile for a specific expiration date.
        """
        current_df = self.calculate_surface()
        if current_df.empty:
            return pd.DataFrame()

        smile_data = current_df.loc[current_df['expiry'] == expiry_date]
        if smile_data.empty:
            return pd.DataFrame()

        if isinstance(smile_data, pd.Series):
            smile_data = smile_data.to_frame().T

        smile_data = smile_data.copy()
        if 'stock_price' not in smile_data.columns:
            print("No 'stock_price' column found in data.")
            return pd.DataFrame()

        spot_price = float(smile_data['stock_price'].iloc[0])
        smile_data['moneyness'] = smile_data['strike'] / spot_price
        if 'days_float' in smile_data.columns:
            smile_data['days_to_expiry'] = smile_data['days_float']
        elif 'days_int' in smile_data.columns:
            smile_data['days_to_expiry'] = smile_data['days_int'].astype(float)
        else:
            smile_data['days_to_expiry'] = np.nan

        smile_data.sort_values('strike', inplace=True)

        keep_cols = [
            'option_type', 'strike', 'impl_vol', 'moneyness',
            'bid', 'ask', 'volume', 'open_interest',
            'days_to_expiry', 'expiry'
        ]
        keep_cols = [col for col in keep_cols if col in smile_data.columns]

        return smile_data[keep_cols].reset_index(drop=True)

    def get_all_smiles(self) -> Dict[str, pd.DataFrame]:
        """
        Get volatility smiles for all available expiration dates.
        """
        current_df = self.calculate_surface()
        if current_df.empty:
            return {}

        expiry_dates = sorted(current_df['expiry'].unique())
        return {
            expiry: self.get_volatility_smile(expiry)
            for expiry in expiry_dates
        }

    def plot_volatility_smile(self, expiry_date: str, use_moneyness: bool = True) -> None:
        """
        Create an interactive plot of the volatility smile for a specific expiration date.
        """
        smile_data = self.get_volatility_smile(expiry_date)
        if smile_data.empty:
            print(f"No data for expiry {expiry_date}.")
            return

        x_values = smile_data['moneyness'] if use_moneyness else smile_data['strike']
        x_label = 'Moneyness (Strike/Spot)' if use_moneyness else 'Strike Price ($)'

        fig = go.Figure()

        # Scatter with lines
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=smile_data['impl_vol'],
                mode='lines+markers',
                name=f'{expiry_date} {self.ticker}',
                hovertemplate=(
                    f"{x_label}: {{x:.2f}}<br>"
                    + "IV: %{y:.1%}<br>"
                    + "Type: %{customdata[0]}<br>"
                    + "Volume: %{customdata[1]}<br>"
                    + "OI: %{customdata[2]}"
                ),
                customdata=np.column_stack((
                    smile_data['option_type'],
                    smile_data['volume'],
                    smile_data['open_interest']
                ))
            )
        )

        # Weighted marker sizes based on volume
        if smile_data['volume'].notna().any():
            max_volume = smile_data['volume'].max()
            marker_sizes = (
                np.sqrt(smile_data['volume'].fillna(0) / max_volume) * 15
                if max_volume > 0
                else 8
            )
        else:
            marker_sizes = 8

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=smile_data['impl_vol'],
                mode='markers',
                marker=dict(
                    size=marker_sizes,
                    color=smile_data['impl_vol'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Implied Vol')
                ),
                name='Volume weighting',
                hoverinfo='skip'
            )
        )

        days_to_expiry = smile_data['days_to_expiry'].iloc[0]
        fig.update_layout(
            title=f'Volatility Smile for {self.ticker} ({expiry_date}, {days_to_expiry:.1f} days)',
            xaxis_title=x_label,
            yaxis_title='Implied Volatility',
            yaxis_tickformat='.0%',
            template='plotly_white'
        )

        # Vertical line at moneyness=1 or near ATM strike
        if use_moneyness:
            fig.add_vline(x=1.0, line_dash='dash', line_color='gray')
        else:
            nearest_atm_index = (smile_data['moneyness'] - 1.0).abs().idxmin()
            atm_strike = smile_data.loc[nearest_atm_index, 'strike']
            fig.add_vline(x=atm_strike, line_dash='dash', line_color='gray')

        fig.show()

    def plot_smile_surface(self) -> None:
        """
        Create a 3D visualization of all volatility smiles arranged by expiry.
        """
        current_df = self.calculate_surface()
        if current_df.empty:
            print("No current data available.")
            return

        fig = go.Figure()
        for expiry in sorted(current_df['expiry'].unique()):
            smile_data = self.get_volatility_smile(expiry)
            if smile_data.empty:
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=smile_data['moneyness'],
                    y=smile_data['days_to_expiry'],
                    z=smile_data['impl_vol'],
                    mode='lines+markers',
                    name=f"{expiry} ({smile_data['days_to_expiry'].iloc[0]:.1f}d)",
                    marker=dict(
                        size=4,
                        color=smile_data['impl_vol'],
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )
            )

        fig.update_layout(
            title=f'Volatility Surface for {self.ticker} (Smile View)',
            scene=dict(
                xaxis_title='Moneyness (Strike/Spot)',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility',
                zaxis=dict(tickformat='.0%')
            ),
            showlegend=True,
            template='plotly_white'
        )

        fig.show()
