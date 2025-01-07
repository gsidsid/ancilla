from typing import Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import pytz
from dataclasses import asdict
from ancilla.models import OptionData
from ancilla.formulae.volatility import (
    create_volatility_surface,
    estimate_liquidity_multiplier,
)
from ancilla.utils.logging import VisualizerLogger
import time
import json
import traceback


class MarketVisualizer:
    """A visualization module for market data that integrates with PolygonDataProvider."""

    def __init__(self, data_provider):
        self.provider = data_provider
        self.utc_tz = pytz.UTC
        self.eastern_tz = pytz.timezone("US/Eastern")
        self.logger = VisualizerLogger("market").get_logger()
        self.logger.info(
            "Initialized MarketVisualizer with provider: %s",
            type(data_provider).__name__,
        )

    def _log_dataframe_stats(self, df: pd.DataFrame, prefix: str = "") -> None:
        """Helper to log summary statistics for a DataFrame"""
        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "numeric_summary": df.describe().to_dict(),
        }
        self.logger.debug(
            f"{prefix} DataFrame stats:\n{json.dumps(stats, indent=2, default=str)}"
        )

    def _ensure_tz_aware(self, dt: datetime) -> datetime:
        """Ensure datetime is timezone-aware, converting to UTC if needed."""
        if dt.tzinfo is None:
            self.logger.debug(f"Converting naive datetime to UTC: {dt}")
            return self.utc_tz.localize(dt)
        return dt.astimezone(self.utc_tz)

    def _prepare_surface_data(
        self,
        options_data: List[OptionData],
        target_date: datetime,
        moneyness_range: Tuple[float, float],
    ) -> Optional[pd.DataFrame]:
        """Prepare options data for surface visualization."""
        self.logger.info(
            f"Preparing surface data for {len(options_data)} options contracts"
        )

        # Log input data statistics for debugging flat surface
        iv_stats = {
            "raw_iv_range": (
                min(
                    opt.implied_volatility
                    for opt in options_data
                    if opt.implied_volatility is not None
                ),
                max(
                    opt.implied_volatility
                    for opt in options_data
                    if opt.implied_volatility is not None
                ),
            ),
        }
        self.logger.debug(
            f"Input IV statistics:\n{json.dumps(iv_stats, default=str, indent=2)}"
        )

        dfs = []
        for contract_type in ["call", "put"]:
            options = [
                opt for opt in options_data if opt.contract_type == contract_type
            ]
            if not options:
                self.logger.warning(f"No {contract_type} options found in data")
                continue

            self.logger.debug(f"Processing {len(options)} {contract_type} options")

            # Process options with explicit dtype handling
            df = pd.DataFrame(
                [
                    {
                        "strike": float(opt.strike),
                        "expiration": opt.expiration,
                        "iv": float(opt.implied_volatility),
                        "underlying_price": float(opt.underlying_price),
                        "volume": int(opt.volume) if opt.volume is not None else 0,
                        "delta": float(opt.delta) if opt.delta is not None else 0.0,
                        "gamma": float(opt.gamma) if opt.gamma is not None else 0.0,
                        "type": contract_type,
                    }
                    for opt in options
                ]
            )

            df["moneyness"] = df["strike"] / df["underlying_price"]
            df["tte"] = df["expiration"].apply(
                lambda x: max(0.0, (x - target_date).total_seconds() / (24 * 3600))
            )

            # Convert volume to integer type for weights
            df["volume"] = df["volume"].fillna(0).astype(int)
            df["weight"] = df["volume"].copy()

            # Log pre-weighting stats
            pre_weight_stats = {
                "iv_range": (float(df["iv"].min()), float(df["iv"].max())),
                "volume_range": (int(df["volume"].min()), int(df["volume"].max())),
                "moneyness_range": (
                    float(df["moneyness"].min()),
                    float(df["moneyness"].max()),
                ),
                "tte_range": (float(df["tte"].min()), float(df["tte"].max())),
            }
            self.logger.debug(
                f"{contract_type} pre-weight stats:\n{json.dumps(pre_weight_stats, indent=2)}"
            )

            # Apply weights with proper integer handling
            delta_mask = df["delta"].between(-0.4, 0.4)
            moneyness_mask = df["moneyness"].between(0.95, 1.05)

            df.loc[delta_mask, "weight"] = df.loc[delta_mask, "weight"] * 2
            df.loc[moneyness_mask, "weight"] = (
                df.loc[moneyness_mask, "weight"] * 1.5
            ).astype(int)

            # Convert to integer after multiplication
            df["weight"] = df["weight"].round().astype(int)

            dfs.append(df)

            # Log post-weighting stats
            post_weight_stats = {
                "weight_range": (int(df["weight"].min()), int(df["weight"].max())),
                "weighted_count": int(len(df[df["weight"] > df["volume"]])),
            }
            self.logger.debug(
                f"{contract_type} post-weight stats:\n{json.dumps(post_weight_stats, indent=2)}"
            )

        if not dfs:
            self.logger.error("No valid options data to process")
            return None

        result = pd.concat(dfs)

        # Log final surface data statistics
        final_stats = {
            "total_points": len(result),
            "iv_range": (float(result["iv"].min()), float(result["iv"].max())),
            "moneyness_range": (
                float(result["moneyness"].min()),
                float(result["moneyness"].max()),
            ),
            "tte_range": (float(result["tte"].min()), float(result["tte"].max())),
        }
        self.logger.debug(
            f"Final surface data statistics:\n{json.dumps(final_stats, indent=2)}"
        )

        return result

    def interpolate_frames(self, frames, n_intermediate=3):
        """
        Create intermediate frames between existing frames for smoother animation.

        Args:
            frames: List of existing go.Frame objects
            n_intermediate: Number of frames to insert between each pair of existing frames

        Returns:
            List of go.Frame objects including interpolated frames
        """
        if len(frames) < 2:
            return frames

        interpolated_frames = []

        for i in range(len(frames) - 1):
            # Add the current frame
            interpolated_frames.append(frames[i])

            # Get the current and next frame data
            current_z = frames[i].data[0].z
            next_z = frames[i + 1].data[0].z
            current_date = pd.to_datetime(frames[i].name)
            next_date = pd.to_datetime(frames[i + 1].name)
            date_diff = (next_date - current_date) / (n_intermediate + 1)

            # Create intermediate frames
            for j in range(n_intermediate):
                # Interpolate the Z values
                alpha = (j + 1) / (n_intermediate + 1)
                interpolated_z = current_z * (1 - alpha) + next_z * alpha

                # Create interpolated date string
                interpolated_date = current_date + (date_diff * (j + 1))
                date_str = interpolated_date.strftime("%Y-%m-%d")

                # Create new frame with interpolated data
                interpolated_frame = go.Frame(
                    data=[
                        go.Surface(
                            x=frames[i].data[0].x,
                            y=frames[i].data[0].y,
                            z=interpolated_z,
                            colorscale="Viridis",
                            colorbar_title="IV",
                        )
                    ],
                    name=f"{date_str}_interpolated_{j+1}",
                )
                interpolated_frames.append(interpolated_frame)

        # Add the last frame
        interpolated_frames.append(frames[-1])

        return interpolated_frames

    def plot_volatility_surfaces(
        self,
        ticker: str,
        date_range: Tuple[datetime, datetime],
        expiration_range: Tuple[int, int] = (7, 90),
        moneyness_range: Tuple[float, float] = (0.85, 1.15),
        delta_range: Tuple[float, float] = (0, 1),
        frame_duration: int = 100,
        n_interpolated_frames: int = 3,
        max_workers: int = 10,
    ) -> Optional[go.Figure]:
        """Create an animated volatility surface with fixed surface generation."""
        overall_start_time = time.time()

        # Ensure dates are timezone-aware
        start_date = date_range[0]
        end_date = date_range[-1]
        start_date = self._ensure_tz_aware(start_date)
        end_date = self._ensure_tz_aware(end_date)

        # Calculate expiration range in days
        min_days = min(expiration_range)
        max_days = max(expiration_range)

        self.logger.info(f"Creating volatility surface animation for {ticker}")
        self.logger.debug(
            f"Animation parameters: start={start_date}, end={end_date} expiry_range={expiration_range} moneyness_range={moneyness_range}"
        )

        try:
            import concurrent.futures

            date_range = [
                start_date + timedelta(days=x)
                for x in range((end_date - start_date).days + 1)
            ]  # type: ignore

            tte_min = min_days
            tte_max = max_days

            frames = []
            z_min, z_max = float("inf"), float("-inf")
            surface_stats_over_time = []

            def fetch_surface_for_date(date: datetime) -> Tuple[str, Optional[Tuple]]:
                """Helper function to fetch and process surface data for a single date"""
                try:
                    fetch_start = time.time()

                    options_data = self.provider.get_options_chain(
                        ticker,
                        reference_date=date,
                        min_days=min_days,
                        expiration_range_days=max_days,
                    )
                    if not options_data:
                        self.logger.warning(f"No options data available for {date}")
                        return (date.strftime("%Y-%m-%d"), None)

                    df = self._prepare_surface_data(options_data, date, moneyness_range)
                    if df is None:
                        return (date.strftime("%Y-%m-%d"), None)

                    # Create grid using global TTE range
                    n_money_points = max(
                        50, int((moneyness_range[1] - moneyness_range[0]) * 100)
                    )
                    n_tte_points = max(50, int((max_days - min_days) * 0.75))
                    actual_max_expiry = float(df["tte"].max())
                    money_points = np.linspace(
                        moneyness_range[0], moneyness_range[1], n_money_points
                    )
                    tte_points = np.linspace(tte_min, actual_max_expiry, n_tte_points)
                    X, Y = np.meshgrid(money_points, tte_points)

                    current_price = df["underlying_price"].iloc[0]
                    strike_points = money_points * current_price

                    # Log input parameters for surface creation
                    surface_input_stats = {
                        "num_points": len(df),
                        "strike_range": (
                            float(df["strike"].min()),
                            float(df["strike"].max()),
                        ),
                        "iv_range": (float(df["iv"].min()), float(df["iv"].max())),
                        "tte_range": (float(df["tte"].min()), float(df["tte"].max())),
                    }
                    self.logger.debug(
                        f"Surface creation inputs:\n{json.dumps(surface_input_stats, indent=2)}"
                    )

                    Z = create_volatility_surface(
                        strikes=np.array(df["strike"].values),
                        expiries=np.array(df["tte"].values),
                        ivs=np.array(df["iv"].values),
                        new_strikes=strike_points,
                        new_expiries=tte_points,
                    )

                    if Z is None:
                        self.logger.error(f"Surface creation failed for {date}")
                        return (date.strftime("%Y-%m-%d"), None)

                    # Calculate and log surface statistics
                    stats = {
                        "date": date.strftime("%Y-%m-%d"),
                        "z_min": float(np.min(Z)),
                        "z_max": float(np.max(Z)),
                        "z_mean": float(np.mean(Z)),
                        "z_std": float(np.std(Z)),
                        "z_zeros": int(np.sum(Z == 0)),
                        "z_nans": int(np.sum(np.isnan(Z))),
                        "processing_time": time.time() - fetch_start,
                    }

                    if stats["z_std"] < 0.001:
                        self.logger.warning(
                            f"WARNING: Nearly flat surface detected for {date}"
                        )
                        self.logger.debug(
                            f"Surface statistics:\n{json.dumps(stats, indent=2)}"
                        )

                    # Apply light smoothing
                    Z = gaussian_filter(Z, sigma=0.3)

                    return (date.strftime("%Y-%m-%d"), (X, Y, Z, stats))

                except Exception as e:
                    self.logger.error(f"Error processing {date}: {str(e)}")
                    self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                    return (date.strftime("%Y-%m-%d"), None)

            # Use ThreadPoolExecutor for parallel processing
            max_workers = min(max_workers, len(date_range))
            self.logger.info(f"Starting parallel processing with {max_workers} workers")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(fetch_surface_for_date, date) for date in date_range
                ]

                for future in concurrent.futures.as_completed(futures):
                    date_str, result = future.result()
                    if result is not None:
                        X, Y, Z, stats = result
                        surface_stats_over_time.append(stats)

                        z_min = min(z_min, float(np.min(Z)))
                        z_max = max(z_max, float(np.max(Z)))

                        frames.append(
                            go.Frame(
                                data=[
                                    go.Surface(
                                        x=X,
                                        y=Y,
                                        z=Z,
                                        colorscale="Viridis",
                                        colorbar_title="IV",
                                    )
                                ],
                                name=date_str,
                            )
                        )

            if not frames:
                self.logger.error("No valid frames generated for animation")
                return None

            # Sort frames by date
            frames.sort(key=lambda f: f.name)

            # Get max y axis value from first frame
            actual_max_expiry = max([f.data[0].y.max() for f in frames])

            # Re-calculate min max values for iv across all frames
            for frame in frames:
                for surface in frame.data:
                    z_min = min(z_min, float(surface.z.min()))
                    z_max = max(z_max, float(surface.z.max()))

            # Analyze surface evolution
            evolution_stats = {
                "total_frames": len(frames),
                "z_range": (float(z_min), float(z_max)),
                "processing_times": {
                    "mean": float(
                        np.mean([s["processing_time"] for s in surface_stats_over_time])
                    ),
                    "max": float(
                        np.max([s["processing_time"] for s in surface_stats_over_time])
                    ),
                },
                "surface_stability": {
                    "mean_std": float(
                        np.mean([s["z_std"] for s in surface_stats_over_time])
                    ),
                    "std_of_means": float(
                        np.std([s["z_mean"] for s in surface_stats_over_time])
                    ),
                },
            }
            self.logger.debug(
                f"Animation statistics:\n{json.dumps(evolution_stats, indent=2)}"
            )

            # 10 frames per second target
            frames = self.interpolate_frames(
                frames, n_intermediate=n_interpolated_frames
            )

            # Create figure with animation
            fig = go.Figure(data=[frames[0].data[0]], frames=frames)

            # Update layout with animation controls
            fig.update_layout(
                title=f"{ticker} Volatility Surface Animation",
                scene=dict(
                    xaxis_title="Moneyness",
                    yaxis_title="Time to Expiry",
                    zaxis_title="Implied Volatility",
                    yaxis=dict(range=[tte_min, actual_max_expiry]),
                    zaxis=dict(range=[0, 80]),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5),
                    ),
                ),
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": True,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    [f.name for f in frames],
                                    {
                                        "frame": {
                                            "duration": frame_duration,
                                            "redraw": True,
                                        },
                                        "transition": {
                                            "duration": frame_duration / 2,
                                            "easing": "cubic-in-out",
                                        },
                                        "fromcurrent": True,
                                        "mode": "immediate",
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                    },
                                ],
                            },
                        ],
                    }
                ],
                sliders=[
                    {
                        "currentvalue": {"prefix": "Date: ", "visible": True},
                        "pad": {"t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 0,
                        "yanchor": "top",
                        "steps": [
                            {
                                "args": [
                                    [frame.name],
                                    {
                                        "frame": {
                                            "duration": frame_duration,
                                            "redraw": True,
                                        },
                                        "mode": "immediate",
                                        "transition": {"duration": frame_duration / 2},
                                    },
                                ],
                                "label": frame.name.split("_")[
                                    0
                                ],  # Only show the date part in slider
                                "method": "animate",
                            }
                            for frame in frames
                        ],
                    }
                ],
                height=800,
                width=1200,
            )

            total_time = time.time() - overall_start_time
            self.logger.info(
                f"Animation creation completed in {total_time:.2f} seconds"
            )
            return fig

        except Exception as e:
            self.logger.error(f"Error creating surface animation: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def plot_option_chain(
        self,
        ticker: str,
        expiry_filter: Optional[List[datetime]] = None,
        plot_greeks: bool = True,
    ) -> Optional[go.Figure]:
        """Create a comprehensive option chain visualization."""
        start_time = time.time()
        self.logger.info(f"Creating option chain visualization for {ticker}")

        try:
            options_data = self.provider.get_options_chain(ticker)
            if not options_data:
                self.logger.error(f"Failed to retrieve options data for {ticker}")
                return None

            df = pd.DataFrame([asdict(opt) for opt in options_data])
            self.logger.debug("Initial options chain data:")
            self._log_dataframe_stats(df, "raw_options")

            if expiry_filter:
                df = df[df["expiration"].isin(expiry_filter)]
                self.logger.debug(
                    f"Filtered to {len(df)} options after applying expiry filter"
                )

            # Create subplots
            n_rows = 3 if plot_greeks else 2
            fig = make_subplots(
                rows=n_rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    ("Option Prices", "Volume", "Greeks")
                    if plot_greeks
                    else ("Option Prices", "Volume")
                ),
            )

            # Plot statistics for debugging
            for opt_type in ["call", "put"]:
                mask = df["contract_type"] == opt_type
                type_data = df[mask]

                stats = {
                    "count": len(type_data),
                    "price_range": (
                        float(type_data["ask"].min()),
                        float(type_data["ask"].max()),
                    ),
                    "volume_range": (
                        int(type_data["volume"].min()),
                        int(type_data["volume"].max()),
                    ),
                    "strike_range": (
                        float(type_data["strike"].min()),
                        float(type_data["strike"].max()),
                    ),
                }
                if plot_greeks:
                    stats.update(
                        {
                            "delta_range": (
                                float(type_data["delta"].min()),
                                float(type_data["delta"].max()),
                            ),
                            "gamma_range": (
                                float(type_data["gamma"].min()),
                                float(type_data["gamma"].max()),
                            ),
                        }
                    )

                self.logger.debug(
                    f"{opt_type.upper()} options statistics:\n{json.dumps(stats, indent=2)}"
                )

                color = "blue" if opt_type == "call" else "red"

                # Price plot
                fig.add_trace(
                    go.Scatter(
                        x=type_data["strike"],
                        y=type_data["ask"],
                        name=f"{opt_type.capitalize()} Ask",
                        line=dict(color=color, dash="dot"),
                        opacity=0.6,
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=type_data["strike"],
                        y=type_data["bid"],
                        name=f"{opt_type.capitalize()} Bid",
                        line=dict(color=color),
                        fill="tonexty",
                        opacity=0.3,
                    ),
                    row=1,
                    col=1,
                )

                # Volume plot
                fig.add_trace(
                    go.Bar(
                        x=type_data["strike"],
                        y=type_data["volume"],
                        name=f"{opt_type.capitalize()} Volume",
                        marker_color=color,
                        opacity=0.6,
                    ),
                    row=2,
                    col=1,
                )

                if plot_greeks:
                    self.logger.debug(f"Adding Greeks plot for {opt_type}")
                    fig.add_trace(
                        go.Scatter(
                            x=type_data["strike"],
                            y=type_data["delta"],
                            name=f"{opt_type.capitalize()} Delta",
                            line=dict(color=color),
                        ),
                        row=3,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=type_data["strike"],
                            y=type_data["gamma"],
                            name=f"{opt_type.capitalize()} Gamma",
                            line=dict(color="green", dash="dot"),
                        ),
                        row=3,
                        col=1,
                    )

            # Add current price line
            current_price = df["underlying_price"].iloc[0]  # type: ignore
            self.logger.debug(
                f"Adding reference line at current price: {current_price}"
            )
            for i in range(n_rows):
                fig.add_vline(
                    x=current_price,
                    line_dash="dash",
                    line_color="gray",
                    row=i + 1,  # type: ignore
                    col=1,  # type: ignore
                )

            fig.update_layout(
                title=f"{ticker} Option Chain Analysis",
                height=300 * n_rows,
                width=900,
                showlegend=True,
            )

            self.logger.info(
                f"Option chain visualization completed in {time.time() - start_time:.2f} seconds"
            )
            return fig

        except Exception as e:
            self.logger.error(f"Error creating option chain visualization: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def plot_technical_analysis(
        self,
        ticker: str,
        start_date: Union[str, datetime, date],
        end_date: Optional[Union[str, datetime, date]] = None,
        indicators: List[str] = ["sma", "bollinger", "volume"],
    ) -> Optional[go.Figure]:
        """Create a technical analysis chart with multiple indicators."""
        start_time = time.time()
        self.logger.info(
            f"Creating technical analysis for {ticker} with indicators: {indicators}"
        )

        try:
            df = self.provider.get_daily_bars(ticker, start_date, end_date)
            if df is None:
                self.logger.error(f"Failed to retrieve daily bars for {ticker}")
                return None

            # Calculate technical indicators
            if "sma" in indicators:
                df["SMA20"] = df["close"].rolling(window=20).mean()
                df["SMA50"] = df["close"].rolling(window=50).mean()

            if "bollinger" in indicators:
                df["BB_middle"] = df["close"].rolling(window=20).mean()
                df["BB_std"] = df["close"].rolling(window=20).std()
                df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
                df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
            )

            # Price candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="OHLC",
                ),
                row=1,
                col=1,
            )

            # Add indicators
            if "sma" in indicators:
                for sma, color in [("SMA20", "blue"), ("SMA50", "red")]:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df[sma], name=sma, line=dict(color=color)
                        ),
                        row=1,
                        col=1,
                    )

            if "bollinger" in indicators:
                # Add upper band
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_upper"],
                        name="BB Upper",
                        line=dict(color="gray", dash="dash"),
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )

                # Add lower band with fill
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BB_lower"],
                        name="BB Lower",
                        line=dict(color="gray", dash="dash"),
                        fill="tonexty",  # Fill to the trace before this one
                        fillcolor="rgba(128, 128, 128, 0.2)",  # Light gray with transparency
                        mode="lines",
                    ),
                    row=1,
                    col=1,
                )

            # Volume bars
            if "volume" in indicators:
                volume_colors = np.where(df["close"] >= df["open"], "green", "red")
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df["volume"],
                        name="Volume",
                        marker_color=volume_colors,
                        opacity=0.7,
                    ),
                    row=2,
                    col=1,
                )

            fig.update_layout(
                title=f"{ticker} Technical Analysis",
                yaxis_title="Price",
                yaxis2_title="Volume",
                xaxis_rangeslider_visible=False,
                height=800,
                width=1000,
            )

            self.logger.info(
                f"Technical analysis visualization completed in {time.time() - start_time:.2f} seconds"
            )
            return fig

        except Exception as e:
            self.logger.error(f"Error creating technical analysis: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def plot_liquidity_analysis(
        self, ticker: str, options_data: Optional[List[OptionData]] = None
    ) -> Optional[go.Figure]:
        """Create a liquidity analysis visualization for options."""
        start_time = time.time()
        self.logger.info(f"Creating liquidity analysis for {ticker}")

        try:
            if options_data is None:
                options_data = self.provider.get_options_chain(ticker)

            if not options_data:
                self.logger.error(f"No options data available for {ticker}")
                return None

            df = pd.DataFrame([asdict(opt) for opt in options_data])
            self.logger.debug("Initial options data statistics:")
            self._log_dataframe_stats(df, "options_data")

            # Calculate liquidity metrics
            df["spread"] = df["ask"] - df["bid"]
            df["spread_pct"] = df["spread"] / ((df["bid"] + df["ask"]) / 2)
            df["moneyness"] = df["strike"] / df["underlying_price"]

            liquidity_stats = {
                "avg_spread_pct": float(df["spread_pct"].mean()),
                "max_spread_pct": float(df["spread_pct"].max()),
                "avg_volume": float(df["volume"].mean()),
                "total_volume": int(df["volume"].sum()),
            }
            self.logger.debug(
                f"Liquidity statistics:\n{json.dumps(liquidity_stats, indent=2)}"
            )

            # Calculate liquidity score
            df["liquidity_score"] = df.apply(
                lambda row: estimate_liquidity_multiplier(
                    volume=row["volume"],
                    open_interest=row["open_interest"],
                    moneyness=row["moneyness"],
                    time_to_expiry=(
                        row["expiration"] - datetime.now(self.utc_tz)
                    ).total_seconds()
                    / (365.25 * 24 * 3600),
                ),
                axis=1,
            )

            score_stats = {
                "avg_liquidity_score": float(df["liquidity_score"].mean()),
                "score_std": float(df["liquidity_score"].std()),
                "score_range": (
                    float(df["liquidity_score"].min()),
                    float(df["liquidity_score"].max()),
                ),
            }
            self.logger.debug(
                f"Liquidity score statistics:\n{json.dumps(score_stats, indent=2)}"
            )

            # Create figure with subplots
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Bid-Ask Spread by Strike",
                    "Volume Distribution",
                    "Liquidity Score Heatmap",
                    "Open Interest Distribution",
                ),
            )

            # Plot bid-ask spreads
            for opt_type in ["call", "put"]:
                mask = df["contract_type"] == opt_type
                fig.add_trace(
                    go.Scatter(
                        x=df[mask]["strike"],
                        y=df[mask]["spread_pct"] * 100,
                        name=f"{opt_type.capitalize()} Spread",
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="blue" if opt_type == "call" else "red",
                            opacity=0.6,
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Volume distribution
            fig.add_trace(
                go.Histogram2d(
                    x=df["strike"],
                    y=df["expiration"],
                    z=df["volume"],
                    colorscale="Viridis",
                    name="Volume",
                ),
                row=1,
                col=2,
            )

            # Liquidity score heatmap
            fig.add_trace(
                go.Heatmap(
                    x=df["strike"],
                    y=df["expiration"],
                    z=df["liquidity_score"],
                    colorscale="RdYlBu",
                    name="Liquidity Score",
                ),
                row=2,
                col=1,
            )

            # Open interest distribution
            if ("open_interest" in df.columns) and (
                df["open_interest"].notna().any() is True
            ):
                fig.add_trace(
                    go.Histogram2d(
                        x=df["strike"],
                        y=df["expiration"],
                        z=df["open_interest"],
                        colorscale="Viridis",
                        name="Open Interest",
                    ),
                    row=2,
                    col=2,
                )
            else:
                self.logger.warning("Open interest data not available")
                fig.add_annotation(
                    text="Open Interest Data Not Available",
                    xref="x4",
                    yref="y4",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    row=2,
                    col=2,
                )

            fig.update_layout(
                title=f"{ticker} Options Liquidity Analysis",
                height=800,
                width=1200,
                showlegend=True,
            )

            self.logger.info(
                f"Liquidity analysis visualization completed in {time.time() - start_time:.2f} seconds"
            )
            return fig

        except Exception as e:
            self.logger.error(f"Error creating liquidity analysis: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return None

    def __del__(self):
        """Cleanup logging handlers on instance destruction"""
        if hasattr(self, "logger"):
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
