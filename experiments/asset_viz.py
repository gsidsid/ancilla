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

    symbol = "AAPL"
    end_date = datetime(2024, 11, 25)
    start_date = end_date - timedelta(days=5)

    print(f"Running combined analysis for {symbol}...")

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

    # Option Chain
    fig_opt = market.plot_option_chain(
        ticker=symbol,
        plot_greeks=True
    )
    if fig_opt is not None:
        figs.append(fig_opt)

    # Liquidity Analysis
    fig_liq = market.plot_liquidity_analysis(
        ticker=symbol
    )
    if fig_liq is not None:
        figs.append(fig_liq)

    # Show all figures
    for fig in figs:
        fig.show()

if __name__ == "__main__":
    main()
