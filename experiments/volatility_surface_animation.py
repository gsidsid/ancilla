# experiments/volatility_surface_animation.py
from ancilla.providers import PolygonDataProvider
from datetime import datetime
import plotly.graph_objects as go
import dotenv
import os

dotenv.load_dotenv()

def main():
    provider = PolygonDataProvider(api_key=os.getenv("POLYGON_API_KEY") or "your-api-key")

    # Create volatility surface
    print("Creating volatility surface animation...")
    from datetime import timedelta

    # Define date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 3)
    date_range = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]

    frames = []
    # Find overall min/max values
    z_min, z_max = float('inf'), float('-inf')
    for date in date_range:
        surface = provider.get_volatility_surface("AAPL", target_date=date)
        if surface is not None:
            z_min = min(z_min, surface[2].min())
            z_max = max(z_max, surface[2].max())
            frames.append(go.Frame(
                data=[go.Surface(z=surface[2])],
                name=date.strftime("%Y-%m-%d")
            ))

    if not frames:
        print("No volatility surface data available")
        return

    fig = go.Figure(
        data=[frames[0].data[0]],
        frames=frames
    )

    # Add slider
    fig.update_layout(
        title='AAPL Volatility Surface Animation',
        scene=dict(
            xaxis_title='Moneyness',
            yaxis_title='Time to Expiry',
            zaxis_title='Implied Volatility',
            zaxis=dict(range=[z_min, z_max]),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [[
                    f.name for f in frames
                ], {'frame': {'duration': 500, 'redraw': True},
                    'transition': {'duration': 0, 'easing': 'linear'},
                    'fromcurrent': True,
                    'mode': 'immediate'}]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
            }],
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Date: '},
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                       'label': f.name,
                       'method': 'animate'} for f in frames]
        }]
    )
    fig.show()


if __name__ == "__main__":
    main()
