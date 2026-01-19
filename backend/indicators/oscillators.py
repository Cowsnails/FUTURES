"""
Oscillator Indicators

Implements RSI, MACD, Stochastic, and other momentum oscillators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import SeparatePaneIndicator, validate_period


class RSI(SeparatePaneIndicator):
    """
    Relative Strength Index

    Momentum oscillator measuring speed and change of price movements.
    Range: 0-100, with 70 = overbought, 30 = oversold.
    """

    def __init__(self, period: int = 14, source: str = 'close', color: str = '#7B1FA2'):
        """
        Initialize RSI indicator.

        Args:
            period: Number of bars for RSI calculation (default: 14)
            source: Price source (default: 'close')
            color: Line color (default: purple)
        """
        super().__init__({
            'period': validate_period(period, min_period=2),
            'source': source,
            'color': color,
            'priceScaleId': 'rsi'
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        period = self.params['period']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        # Calculate price changes
        delta = df[source].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return pd.DataFrame({
            'time': df['time'],
            'value': rsi
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config.update({
            'title': f"RSI({self.params['period']})",
            'levels': [30, 70],  # Overbought/oversold levels
            'levelColors': ['#ef5350', '#26a69a'],
            'scaleMargins': {'top': 0.1, 'bottom': 0.1}
        })
        return config


class MACD(SeparatePaneIndicator):
    """
    Moving Average Convergence Divergence

    Trend-following momentum indicator showing relationship between two EMAs.
    Returns three series: MACD line, signal line, and histogram.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        source: str = 'close'
    ):
        """
        Initialize MACD indicator.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            source: Price source (default: 'close')
        """
        super().__init__({
            'fast': validate_period(fast),
            'slow': validate_period(slow),
            'signal': validate_period(signal),
            'source': source,
            'priceScaleId': 'macd'
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        fast = self.params['fast']
        slow = self.params['slow']
        signal_period = self.params['signal']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        # Calculate EMAs
        ema_fast = df[source].ewm(span=fast, adjust=False).mean()
        ema_slow = df[source].ewm(span=slow, adjust=False).mean()

        # Calculate MACD line
        macd_line = ema_fast - ema_slow

        # Calculate signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'time': df['time'],
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        """MACD needs special rendering (2 lines + histogram)"""
        return {
            'type': 'macd',
            'pane': 'separate',
            'title': f"MACD({self.params['fast']},{self.params['slow']},{self.params['signal']})",
            'macdColor': '#2962FF',
            'signalColor': '#FF6D00',
            'histogramUpColor': 'rgba(38, 166, 154, 0.5)',
            'histogramDownColor': 'rgba(239, 83, 80, 0.5)',
            'priceScaleId': 'macd'
        }


class Stochastic(SeparatePaneIndicator):
    """
    Stochastic Oscillator

    Momentum indicator comparing closing price to price range over time.
    Range: 0-100, with 80 = overbought, 20 = oversold.
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth: int = 3,
        color_k: str = '#2962FF',
        color_d: str = '#FF6D00'
    ):
        """
        Initialize Stochastic indicator.

        Args:
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            smooth: Smoothing period (default: 3)
            color_k: %K line color (default: blue)
            color_d: %D line color (default: orange)
        """
        super().__init__({
            'k_period': validate_period(k_period),
            'd_period': validate_period(d_period),
            'smooth': validate_period(smooth),
            'color_k': color_k,
            'color_d': color_d,
            'priceScaleId': 'stoch'
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic"""
        k_period = self.params['k_period']
        d_period = self.params['d_period']
        smooth = self.params['smooth']

        # Calculate lowest low and highest high
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        # Calculate %K (raw stochastic)
        k_raw = 100 * (df['close'] - low_min) / (high_max - low_min).replace(0, 1e-10)

        # Smooth %K
        k = k_raw.rolling(window=smooth).mean()

        # Calculate %D (signal line)
        d = k.rolling(window=d_period).mean()

        return pd.DataFrame({
            'time': df['time'],
            'k': k,
            'd': d
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        """Stochastic needs two lines"""
        return {
            'type': 'stochastic',
            'pane': 'separate',
            'title': f"Stoch({self.params['k_period']},{self.params['d_period']})",
            'kColor': self.params['color_k'],
            'dColor': self.params['color_d'],
            'levels': [20, 80],
            'levelColors': ['#ef5350', '#26a69a'],
            'priceScaleId': 'stoch'
        }


class CCI(SeparatePaneIndicator):
    """
    Commodity Channel Index

    Measures deviation from average price. Oscillates around zero.
    Range: typically -100 to +100, with +100 = overbought, -100 = oversold.
    """

    def __init__(self, period: int = 20, color: str = '#9C27B0'):
        """
        Initialize CCI indicator.

        Args:
            period: Number of bars (default: 20)
            color: Line color (default: purple)
        """
        super().__init__({
            'period': validate_period(period),
            'color': color,
            'priceScaleId': 'cci'
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate CCI"""
        period = self.params['period']

        # Calculate typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Calculate SMA of typical price
        sma_tp = tp.rolling(window=period).mean()

        # Calculate mean absolute deviation
        mad = tp.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(),
            raw=False
        )

        # Calculate CCI
        cci = (tp - sma_tp) / (0.015 * mad).replace(0, 1e-10)

        return pd.DataFrame({
            'time': df['time'],
            'value': cci
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config.update({
            'title': f"CCI({self.params['period']})",
            'levels': [-100, 0, 100],
            'levelColors': ['#ef5350', '#758696', '#26a69a']
        })
        return config


class ROC(SeparatePaneIndicator):
    """
    Rate of Change

    Measures percentage change in price over a specified period.
    Oscillates around zero.
    """

    def __init__(self, period: int = 12, source: str = 'close', color: str = '#FF6D00'):
        """
        Initialize ROC indicator.

        Args:
            period: Number of bars (default: 12)
            source: Price source (default: 'close')
            color: Line color (default: orange)
        """
        super().__init__({
            'period': validate_period(period),
            'source': source,
            'color': color,
            'priceScaleId': 'roc'
        })

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ROC"""
        period = self.params['period']
        source = self.params['source']

        if source not in df.columns:
            raise ValueError(f"Source column '{source}' not found in DataFrame")

        # Calculate rate of change
        roc = 100 * (df[source] - df[source].shift(period)) / df[source].shift(period).replace(0, 1e-10)

        return pd.DataFrame({
            'time': df['time'],
            'value': roc
        })

    @property
    def plot_config(self) -> Dict[str, Any]:
        config = super().plot_config
        config.update({
            'title': f"ROC({self.params['period']})",
            'levels': [0],
            'levelColors': ['#758696']
        })
        return config


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

    print("Testing Oscillator Indicators\n")

    # Test RSI
    rsi = RSI(period=14)
    rsi_result = rsi.calculate(sample_df)
    print(f"RSI: {rsi.get_display_name()}")
    print(f"  Values: {len(rsi_result)} bars")
    print(f"  Range: {rsi_result['value'].min():.2f} - {rsi_result['value'].max():.2f}")
    print(f"  Config: {rsi.plot_config}")
    print()

    # Test MACD
    macd = MACD(fast=12, slow=26, signal=9)
    macd_result = macd.calculate(sample_df)
    print(f"MACD: {macd.get_display_name()}")
    print(f"  Values: {len(macd_result)} bars")
    print(f"  Columns: {list(macd_result.columns)}")
    print(f"  Config: {macd.plot_config}")
    print()

    # Test Stochastic
    stoch = Stochastic(k_period=14, d_period=3)
    stoch_result = stoch.calculate(sample_df)
    print(f"Stochastic: {stoch.get_display_name()}")
    print(f"  Values: {len(stoch_result)} bars")
    print(f"  Columns: {list(stoch_result.columns)}")
    print(f"  Config: {stoch.plot_config}")
