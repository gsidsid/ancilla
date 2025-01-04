# experiments/volatility_surface_animation.py
from ancilla.providers import PolygonDataProvider
from ancilla.visualizations import MarketVisualizer
from datetime import datetime
import dotenv
import os

dotenv.load_dotenv()

def main():
    # Initialize provider and visualizer
    provider = PolygonDataProvider(api_key=os.getenv("POLYGON_API_KEY") or "your-api-key")
    market = MarketVisualizer(provider)

    # Define date range
    end_date = datetime(2024, 12, 1)
    start_date = datetime(2024, 11, 1)
    symbol = "AAPL"

    print("Creating volatility surfaces...")
    # Limitation: Only visualizes LEAPS data which can be kinda sparse
    fig = market.plot_volatility_surfaces(
        ticker=symbol,
        date_range=(start_date, end_date),
        expiration_range=(30, 180),
        moneyness_range=(0.9, 1.1),
        delta_range=(0.1, 0.9),
        frame_duration=100,
        n_interpolated_frames=1,
        max_workers=16
    )

    if fig is not None:
        fig.show()
    else:
        print("Failed to create animation")

if __name__ == "__main__":
    main()
