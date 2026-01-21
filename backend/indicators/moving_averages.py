"""
Moving Average Indicators

Implements SMA, EMA, and other moving average variants.
"""

import pandas as pd
from typing import Dict, Any, Optional
from .base import OverlayIndicator, validate_period


class SMA(OverlayIndicator):
    """
    Simple Moving Average

    Calculates the arithmetic mean of prices over a specified period.
    """

    def __init__(self, period: int = 20, source: str = 'close', color: str = '#2962FF'):
        """
        Initialize SMA indicator.

        Args:
            period: Number of bars to average (default: 20)
            source: Price source (default: 'close')
            color: Line color (default: TradingView blue)
        """
        super().__init__({
            'period': validate_period(period),
            'source': source,
            'color': color
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SMA"""
        period = self.params['period']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        sma = df[source].rolling(window=period).mean()

        return pd.DataFrame({
            'time': df['time'],
            'value': sma
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config['title'] = f"SMA({self.params['period']})"
        return config


class EMA(OverlayIndicator):
    """
    Exponential Moving Average

    Gives more weight to recent prices using exponential smoothing.
    """

    def __init__(self, period: int = 20, source: str = 'close', color: str = '#FF6D00'):
        """
        Initialize EMA indicator.

        Args:
            period: Number of bars for EMA (default: 20)
            source: Price source (default: 'close')
            color: Line color (default: TradingView orange)
        """
        super().__init__({
            'period': validate_period(period),
            'source': source,
            'color': color
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA"""
        period = self.params['period']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        ema = df[source].ewm(span=period, adjust=False).mean()

        return pd.DataFrame({
            'time': df['time'],
            'value': ema
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config['title'] = f"EMA({self.params['period']})"
        return config


class WMA(OverlayIndicator):
    """
    Weighted Moving Average

    Linearly weighted moving average giving more weight to recent prices.
    """

    def __init__(self, period: int = 20, source: str = 'close', color: str = '#4CAF50'):
        """
        Initialize WMA indicator.

        Args:
            period: Number of bars for WMA (default: 20)
            source: Price source (default: 'close')
            color: Line color (default: green)
        """
        super().__init__({
            'period': validate_period(period),
            'source': source,
            'color': color
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate WMA"""
        period = self.params['period']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        # Create weights (linearly decreasing)
        weights = pd.Series(range(1, period + 1))
        weight_sum = weights.sum()

        def weighted_average(x):
            if len(x) < period:
                return pd.NA
            return (x * weights).sum() / weight_sum

        wma = df[source].rolling(window=period).apply(weighted_average, raw=False)

        return pd.DataFrame({
            'time': df['time'],
            'value': wma
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config['title'] = f"WMA({self.params['period']})"
        return config


class VWAP(OverlayIndicator):
    """
    Volume Weighted Average Price

    Average price weighted by volume. Resets daily.
    Note: This implementation is a cumulative VWAP within the dataset.
    """

    def __init__(self, color: str = '#9C27B0'):
        """
        Initialize VWAP indicator.

        Args:
            color: Line color (default: purple)
        """
        super().__init__({'color': color})

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP"""
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Calculate cumulative volume * price and cumulative volume
        cum_vol_price = (typical_price * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()

        # Avoid division by zero
        vwap = cum_vol_price / cum_vol.replace(0, 1)

        return pd.DataFrame({
            'time': df['time'],
            'value': vwap
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config['title'] = "VWAP"
        config['lineWidth'] = 2
        return config


class BollingerBands(OverlayIndicator):
    """
    Bollinger Bands

    Shows volatility bands around a moving average.
    Returns three series: middle (SMA), upper, and lower bands.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        source: str = 'close',
        color: str = '#2962FF'
    ):
        """
        Initialize Bollinger Bands.

        Args:
            period: SMA period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            source: Price source (default: 'close')
            color: Line color (default: blue)
        """
        super().__init__({
            'period': validate_period(period),
            'std_dev': std_dev,
            'source': source,
            'color': color
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        period = self.params['period']
        std_dev = self.params['std_dev']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        # Calculate middle band (SMA)
        middle = df[source].rolling(window=period).mean()

        # Calculate standard deviation
        std = df[source].rolling(window=period).std()

        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return pd.DataFrame({
            'time': df['time'],
            'middle': middle,
            'upper': upper,
            'lower': lower
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        """Bollinger Bands need special rendering (3 lines)"""
        return {
            'type': 'bands',
            'pane': 'main',
            'color': self.params['color'],
            'title': f"BB({self.params['period']}, {self.params['std_dev']})",
            'lineWidth': 1,
            'fillOpacity': 0.1
        }


if __name__ == '__main__':
    # Example usage and testing
    import numpy as np

    # Create sample data
    sample_df = pd.DataFrame({
        'time': range(100),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    print("Testing Moving Average Indicators\n")

    # Test SMA
    sma = SMA(period=20)
    sma_result = sma.calculate(sample_df)
    print(f"SMA: {sma.get_display_name()}")
    print(f"  Values: {len(sma_result)} bars")
    print(f"  Config: {sma.plot_config}")
    print()

    # Test EMA
    ema = EMA(period=20)
    ema_result = ema.calculate(sample_df)
    print(f"EMA: {ema.get_display_name()}")
    print(f"  Values: {len(ema_result)} bars")
    print(f"  Config: {ema.plot_config}")
    print()

    # Test Bollinger Bands
    bb = BollingerBands(period=20, std_dev=2.0)
    bb_result = bb.calculate(sample_df)
    print(f"Bollinger Bands: {bb.get_display_name()}")
    print(f"  Values: {len(bb_result)} bars")
    print(f"  Columns: {list(bb_result.columns)}")
    print(f"  Config: {bb.plot_config}")
