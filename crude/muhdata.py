from polygon import RESTClient
import os
import dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def get_current_price(client, ticker):
    """
    Fetch current stock price from Polygon API using stock snapshot
    """
    try:
        print(f"\nFetching current price for {ticker}...")
        stock_snapshot = client.get_snapshot_ticker("stocks", ticker)
        if stock_snapshot and hasattr(stock_snapshot, 'session') and stock_snapshot.session:
            price = stock_snapshot.session.close
            print(f"Current price: ${price:.2f}")
            return price
        else:
            print("No price data available in snapshot")
            return None
    except Exception as e:
        print(f"Error fetching current price: {e}")
        print(f"Type of error: {type(e)}")
        print(f"Error details: {str(e)}")
        return None

import argparse

def get_current_price(client, ticker):
    """
    Fetch current stock price using daily open/close endpoint
    """
    try:
        print(f"\nFetching current price for {ticker}...")
        today = datetime.now()

        # Try getting today's data first
        try:
            daily_data = client.get_daily_open_close(ticker, today.strftime('%Y-%m-%d'))
            price = daily_data.close
            print(f"Using today's closing price: ${price:.2f}")
            return price
        except:
            # If today's data isn't available, try yesterday
            yesterday = today - timedelta(days=1)
            daily_data = client.get_daily_open_close_agg(ticker, yesterday.strftime('%Y-%m-%d'))
            price = daily_data.close
            print(f"Using yesterday's closing price: ${price:.2f}")
            return price

    except Exception as e:
        print(f"Error fetching daily price: {e}")
        return None

def fetch_options_data(client, ticker):
    """
    Fetch options chain data from Polygon API
    """
    print(f"\nFetching options data for {ticker}...")

    # Get current price first
    current_price = get_current_price(client, ticker)
    if current_price is None:
        print("Could not get current price")
        return None

    today = datetime.now()
    min_date = today.strftime('%Y-%m-%d')
    max_date = (today + timedelta(days=365)).strftime('%Y-%m-%d')

    options_data = []
    count = 0

    try:
        # Get iterator for options chain
        options_iter = client.list_snapshot_options_chain(
            ticker,
            params={
                "expiration_date.gte": min_date,
                "expiration_date.lte": max_date,
            }
        )

        # Get first option to examine structure
        first_option = next(options_iter)
        print("\nFirst option details:")
        print("Attributes:", dir(first_option))
        print("\nUnderlying asset:", first_option.underlying_asset)
        if hasattr(first_option.underlying_asset, '__dict__'):
            print("Underlying asset dict:", first_option.underlying_asset.__dict__)
        print("\nDetails:", first_option.details)
        if hasattr(first_option.details, '__dict__'):
            print("Details dict:", first_option.details.__dict__)

        # Try to get current price from various possible locations
        if hasattr(first_option.underlying_asset, 'value') and first_option.underlying_asset.value:
            current_price = float(first_option.underlying_asset.value)
        elif hasattr(first_option.underlying_asset, 'price') and first_option.underlying_asset.price:
            current_price = float(first_option.underlying_asset.price)

        if current_price:
            print(f"\nExtracted current price: ${current_price:.2f}")
        else:
            print("\nCould not extract current price - please enter it manually:")
            current_price = float(input("Current price for " + ticker + ": $"))


        # Process options data
        def process_option(opt):
            try:
                return {
                    'strike': opt.details.strike_price if hasattr(opt, 'details') else None,
                    'expiration': opt.details.expiration_date if hasattr(opt, 'details') else None,
                    'type': opt.details.contract_type if hasattr(opt, 'details') else None,
                    'iv': opt.implied_volatility/100 if hasattr(opt, 'implied_volatility') and opt.implied_volatility is not None else None,
                    'underlying_price': current_price,
                    'delta': opt.greeks.delta if hasattr(opt, 'greeks') and opt.greeks else None,
                    'gamma': opt.greeks.gamma if hasattr(opt, 'greeks') and opt.greeks else None,
                    'theta': opt.greeks.theta if hasattr(opt, 'greeks') and opt.greeks else None,
                    'vega': opt.greeks.vega if hasattr(opt, 'greeks') and opt.greeks else None,
                    'bid': opt.last_quote.bid if hasattr(opt, 'last_quote') and opt.last_quote else None,
                    'ask': opt.last_quote.ask if hasattr(opt, 'last_quote') and opt.last_quote else None,
                    'volume': opt.day.volume if hasattr(opt, 'day') and opt.day else None
                }
            except AttributeError as e:
                return None

        # Process first option
        first_data = process_option(first_option)
        if first_data and all(first_data[field] is not None for field in ['strike', 'expiration', 'iv']):
            options_data.append(first_data)
            count += 1

        # Process remaining options
        skipped_count = 0
        for option in options_iter:
            data = process_option(option)
            if data and all(data[field] is not None for field in ['strike', 'expiration', 'iv']):
                options_data.append(data)
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} valid options...")
            else:
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"Skipped {skipped_count} invalid options...")

        print(f"\nTotal valid options processed: {count}")
        print(f"Total options skipped: {skipped_count}")

        if not options_data:
            print("No valid options data found")
            return None

        # Create DataFrame
        df = pd.DataFrame(options_data)

        # Print DataFrame info
        print("\nDataFrame Info:")
        print(df.info())
        print("\nSample Data:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error in fetch_options_data: {e}")
        return None

        print(f"\nTotal valid options processed: {count}")
        print(f"Total options skipped: {skipped_count}")

        if not options_data:
            print("No valid options data found")
            return None

        # Create DataFrame
        df = pd.DataFrame(options_data)

        # Print DataFrame info
        print("\nDataFrame Info:")
        print(df.info())
        print("\nSample Data:")
        print(df.head())

        return df

    except Exception as e:
        print(f"Error in fetch_options_data: {e}")
        return None

def create_volatility_surface(df):
    """
    Create volatility surface from options data
    """
    if df is None or df.empty:
        return None, None, None

    try:
        print("\nCreating volatility surface...")
        print(f"Initial shape: {df.shape}")

        # Filter out rows with missing essential data
        df = df.dropna(subset=['strike', 'expiration', 'iv', 'underlying_price'])
        print(f"Shape after dropping NaN: {df.shape}")

        # Calculate moneyness
        df['moneyness'] = df['strike'] / df['underlying_price']

        # Calculate time to expiry in years
        today = datetime.now()
        df['tte'] = df['expiration'].apply(lambda x:
            (datetime.strptime(x, '%Y-%m-%d') - today).days / 365.0)

        # Filter data
        df_filtered = df[
            (df['moneyness'].between(0.5, 1.5)) &
            (df['iv'].between(0, 2)) &
            (df['tte'] > 0)
        ]
        print(f"Shape after filtering: {df_filtered.shape}")

        if df_filtered.empty:
            print("\nData ranges:")
            print(f"Moneyness range: {df['moneyness'].min():.2f} to {df['moneyness'].max():.2f}")
            print(f"IV range: {df['iv'].min():.2f} to {df['iv'].max():.2f}")
            print(f"TTE range: {df['tte'].min():.2f} to {df['tte'].max():.2f}")
            return None, None, None

        # Create grid
        moneyness_range = np.linspace(df_filtered['moneyness'].min(), df_filtered['moneyness'].max(), 50)
        tte_range = np.linspace(df_filtered['tte'].min(), df_filtered['tte'].max(), 50)
        X, Y = np.meshgrid(moneyness_range, tte_range)

        # Interpolate
        points = df_filtered[['moneyness', 'tte']].values
        values = df_filtered['iv'].values
        Z = griddata(points, values, (X, Y), method='linear')

        return X, Y, Z

    except Exception as e:
        print(f"Error in create_volatility_surface: {e}")
        return None, None, None

def plot_surface(X, Y, Z, ticker):
    """
    Plot volatility surface
    """
    if any(v is None for v in [X, Y, Z]):
        print("Cannot plot: missing data")
        return

    try:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_surface(X, Y, Z, cmap='viridis')

        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Time to Expiry (Years)')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'{ticker} Implied Volatility Surface')

        fig.colorbar(surface)
        plt.show()

    except Exception as e:
        print(f"Error in plot_surface: {e}")

def main():
    # Load environment variables
    dotenv.load_dotenv()
    api_key = os.getenv("POLYGON_API_KEY")

    if not api_key:
        print("Error: No API key found")
        return

    # Initialize client
    client = RESTClient(api_key=api_key)

    # Set parameters
    ticker = "AAPL"  # Can make this configurable if needed

    # Fetch and process options data (price is fetched inside the function now)
    df = fetch_options_data(client, ticker)

    if df is not None:
        # Create volatility surface
        X, Y, Z = create_volatility_surface(df)

        # Plot surface
        if X is not None:
            plot_surface(X, Y, Z, ticker)

            # Save data
            output_file = f"{ticker}_options_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(output_file, index=False)
            print(f"\nData saved to {output_file}")
        else:
            print("Could not create volatility surface")
    else:
        print("Failed to process options data")

if __name__ == "__main__":
    main()
