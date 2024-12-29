from ancilla.providers import PolygonDataProvider
from ancilla.backtesting import Backtest
from ancilla.strategies import OptionsStrategy

import bt
import dotenv
import os

dotenv.load_dotenv()
provider = PolygonDataProvider(api_key=os.getenv("POLYGON_API_KEY") or "your-api-key")

class SelectOptionsByGreeks(bt.Algo):
    """
    A custom Algo to select options based on their Greeks.
    """

    def __init__(self, delta_range: tuple, gamma_min: float):
        print(f"Initializing SelectOptionsByGreeks with delta_range: {delta_range} and gamma_min: {gamma_min}")
        self.delta_range = delta_range
        self.gamma_min = gamma_min

    def __call__(self, target: bt.Strategy) -> bool:
        print(f"Selecting options based on Greeks for strategy: {target.name}")

        options = provider.get_options_chain(ticker=target.universe[0])
        selected = [
            o for o in options
            if self.delta_range[0] <= abs(o.delta or 0) <= self.delta_range[1] and (o.gamma or 0) >= self.gamma_min
        ]

        print(f"Selected {len(selected)} options for strategy: {target.name}")
        target.temp['selected'] = selected
        return True

# Example Usage
def main():
    engine = Backtest(provider, starting_capital=100000)

    # Fetch data
    tickers = ["AAPL"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    print("Fetching data...")
    data = engine.fetch_data(tickers, start_date, end_date)

    # Define strategy
    print("Defining strategy...")
    strategy = OptionsStrategy(
        name="Options Strategy",
        algos=[
            bt.algos.RunMonthly(),
            SelectOptionsByGreeks(delta_range=(0.3, 0.7), gamma_min=0.01),
            bt.algos.WeighEqually(),
            bt.algos.Rebalance()
        ],
        universe=tickers
    )

    # Run backtest
    print("Running backtest...")
    result = engine.run(strategy, data)

    # Display results
    print("Backtest results:")
    print(result.display())
    result.plot()

if __name__ == "__main__":
    main()
