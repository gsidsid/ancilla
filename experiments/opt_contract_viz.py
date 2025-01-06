# experiments/asset_viz.py
from ancilla.providers import PolygonDataProvider
from ancilla.visualizations import MarketVisualizer
from datetime import datetime, timedelta
import dotenv
import os

dotenv.load_dotenv()

def main():
    # Initialize provider and visualizer
    provider = PolygonDataProvider(api_key=os.getenv("POLYGON_API_KEY") or "your-api-key")
    market = MarketVisualizer(provider)

    symbol = "O:META240223C00385000"
    end_date = datetime(2024, 2, 20)
    start_date = datetime(2024, 2, 1)

    print(f"Running analysis for {symbol}...")

    # Create basic visualizations
    figs = []

    # Technical Analysis
    fig_tech = market.plot_technical_analysis(
        ticker=symbol,
        start_date=start_date,
        end_date=end_date,
        indicators=['sma', 'bollinger', 'volume']
    )
    if fig_tech is not None:
        figs.append(fig_tech)

    # Show all figures
    for fig in figs:
        fig.show()

if __name__ == "__main__":
    main()
