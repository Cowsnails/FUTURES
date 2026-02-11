"""
Volume Delta Indicators

Technical indicators derived from TastyTrade's volume delta data:
- VolumeDelta: Per-bar delta histogram (buy_vol - sell_vol)
- CumulativeDelta: Session cumulative volume delta (CVD) line
- DeltaDivergence: Detects price/delta divergences

These integrate with the existing indicator system and can be added
via the indicator manager like any other indicator.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from .base import SeparatePaneIndicator

logger = logging.getLogger(__name__)


class VolumeDelta(SeparatePaneIndicator):
    """
    Per-bar volume delta histogram.

    Shows buy_volume - sell_volume for each bar as a colored histogram.
    Green bars = net buying (positive delta), Red bars = net selling (negative delta).

    Requires delta data from TastyTrade's TimeAndSale stream to be present
    in the DataFrame columns: 'buy_volume', 'sell_volume'.
    Falls back to zero if delta columns are not available.
    """

    def __init__(self, **params):
        defaults = {
            'color_up': params.get('color_up', '#26A69A'),    # Green for positive delta
            'color_down': params.get('color_down', '#EF5350'),  # Red for negative delta
        }
        defaults.update(params)
        super().__init__(params=defaults)
        self.name = 'VolumeDelta'

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-bar volume delta."""
        self.validate_dataframe(df)

        result = pd.DataFrame({'time': df['time']})

        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            result['delta'] = df['buy_volume'].astype(float) - df['sell_volume'].astype(float)
            result['buy_volume'] = df['buy_volume'].astype(float)
            result['sell_volume'] = df['sell_volume'].astype(float)
        else:
            # No delta data available - return zeros
            result['delta'] = 0.0
            result['buy_volume'] = 0.0
            result['sell_volume'] = 0.0

        # Color based on delta sign
        result['color'] = result['delta'].apply(
            lambda d: self.params['color_up'] if d >= 0 else self.params['color_down']
        )

        return result

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'type': 'histogram',
            'pane': 'separate',
            'priceScaleId': 'volume_delta',
            'color_up': self.params.get('color_up', '#26A69A'),
            'color_down': self.params.get('color_down', '#EF5350'),
            'priceLineVisible': False,
            'lastValueVisible': True,
        }


class CumulativeDelta(SeparatePaneIndicator):
    """
    Cumulative Volume Delta (CVD) line.

    Running sum of per-bar delta across the session. Resets at session
    boundaries (6:00 PM ET for CME Globex).

    Useful for identifying divergences between price and order flow:
    - Price making new highs + CVD declining = bearish divergence
    - Price making new lows + CVD rising = bullish divergence

    Requires 'buy_volume' and 'sell_volume' columns in the DataFrame.
    """

    def __init__(self, **params):
        defaults = {
            'color': params.get('color', '#2196F3'),
            'lineWidth': params.get('lineWidth', 2),
        }
        defaults.update(params)
        super().__init__(params=defaults)
        self.name = 'CumulativeDelta'

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative volume delta."""
        self.validate_dataframe(df)

        result = pd.DataFrame({'time': df['time']})

        if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
            bar_delta = df['buy_volume'].astype(float) - df['sell_volume'].astype(float)
            result['value'] = bar_delta.cumsum()
        elif 'cumulative_delta' in df.columns:
            result['value'] = df['cumulative_delta'].astype(float)
        else:
            result['value'] = 0.0

        return result

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'type': 'line',
            'pane': 'separate',
            'priceScaleId': 'cvd',
            'color': self.params.get('color', '#2196F3'),
            'lineWidth': self.params.get('lineWidth', 2),
            'priceLineVisible': False,
            'lastValueVisible': True,
        }


class DeltaDivergence(SeparatePaneIndicator):
    """
    Delta Divergence Detector.

    Identifies divergences between price action and cumulative delta:
    - Bearish divergence: Price new high + CVD lower high
    - Bullish divergence: Price new low + CVD higher low

    Uses a lookback window to compare swing highs/lows.
    """

    def __init__(self, **params):
        defaults = {
            'lookback': params.get('lookback', 20),
            'color_bullish': params.get('color_bullish', '#26A69A'),
            'color_bearish': params.get('color_bearish', '#EF5350'),
        }
        defaults.update(params)
        super().__init__(params=defaults)
        self.name = 'DeltaDivergence'

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect price/delta divergences."""
        self.validate_dataframe(df)

        lookback = self.params.get('lookback', 20)
        result = pd.DataFrame({'time': df['time']})

        if 'buy_volume' not in df.columns or 'sell_volume' not in df.columns:
            result['value'] = 0.0
            return result

        bar_delta = df['buy_volume'].astype(float) - df['sell_volume'].astype(float)
        cvd = bar_delta.cumsum()
        close = df['close'].astype(float)

        # Rolling high/low for price and CVD
        price_high = close.rolling(lookback, min_periods=1).max()
        price_low = close.rolling(lookback, min_periods=1).min()
        cvd_high = cvd.rolling(lookback, min_periods=1).max()
        cvd_low = cvd.rolling(lookback, min_periods=1).min()

        divergence = pd.Series(0.0, index=df.index)

        # Bearish: price at rolling high but CVD below its rolling high
        bearish_mask = (close >= price_high * 0.999) & (cvd < cvd_high * 0.95)
        divergence[bearish_mask] = -1.0

        # Bullish: price at rolling low but CVD above its rolling low
        bullish_mask = (close <= price_low * 1.001) & (cvd > cvd_low * 1.05)
        divergence[bullish_mask] = 1.0

        result['value'] = divergence

        return result

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'type': 'histogram',
            'pane': 'separate',
            'priceScaleId': 'delta_div',
            'color_up': self.params.get('color_bullish', '#26A69A'),
            'color_down': self.params.get('color_bearish', '#EF5350'),
            'priceLineVisible': False,
            'lastValueVisible': True,
        }
