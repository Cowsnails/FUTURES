"""
Base Indicator Class

Provides abstract base class for all technical indicators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Indicator(ABC):
    """
    Base class for all technical indicators.

    Subclasses must implement:
    - calculate(): Compute indicator values
    - plot_config: Return configuration for rendering
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize indicator with parameters.

        Args:
            params: Dictionary of indicator parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        self.id = f"{self.name}_{self._get_param_string()}"

    def _get_param_string(self) -> str:
        """Generate parameter string for unique ID"""
        if not self.params:
            return ""

        # Create string from key params
        key_params = []
        for key in ['period', 'fast', 'slow', 'signal']:
            if key in self.params:
                key_params.append(str(self.params[key]))

        return "_".join(key_params) if key_params else ""

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.

        Args:
            df: DataFrame with OHLCV data (columns: time, open, high, low, close, volume)

        Returns:
            DataFrame with 'time' column and indicator value column(s)
            Column names should match those expected by plot_config
        """
        pass

    @property
    @abstractmethod
    def plot_config(self) -> Dict[str, Any]:
        """
        Return plotting configuration for this indicator.

        Returns:
            Dictionary with plotting configuration:
            {
                'type': 'line' | 'histogram' | 'macd',
                'color': '#RRGGBB',
                'lineWidth': int,
                'pane': 'main' | 'separate',
                'priceScaleId': str (optional, for separate pane),
                'levels': [float] (optional, e.g., [30, 70] for RSI),
                ...additional type-specific config
            }
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError if invalid
        """
        required = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        return True

    def get_display_name(self) -> str:
        """Get human-readable display name"""
        if not self.params:
            return self.name

        # Add key parameters to name
        parts = [self.name]

        if 'period' in self.params:
            parts.append(f"({self.params['period']})")
        elif 'fast' in self.params and 'slow' in self.params:
            parts.append(f"({self.params['fast']},{self.params['slow']})")

        return "".join(parts)

    def calculate_safe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate indicator with error handling.

        Args:
            df: Input DataFrame

        Returns:
            Result DataFrame or None if error
        """
        try:
            self.validate_dataframe(df)
            result = self.calculate(df)

            # Drop NaN rows
            result = result.dropna()

            return result

        except Exception as e:
            logger.error(f"Error calculating {self.name}: {e}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert indicator to dictionary representation"""
        return {
            'name': self.name,
            'id': self.id,
            'display_name': self.get_display_name(),
            'params': self.params,
            'plot_config': self.plot_config
        }


class OverlayIndicator(Indicator):
    """Base class for indicators that overlay on main price chart"""

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'type': 'line',
            'pane': 'main',
            'color': self.params.get('color', '#2962FF'),
            'lineWidth': self.params.get('lineWidth', 2),
            'priceLineVisible': False,
            'lastValueVisible': True,
        }


class SeparatePaneIndicator(Indicator):
    """Base class for indicators that render in separate pane"""

    @property
    def plot_config(self) -> Dict[str, Any]:
        return {
            'type': 'line',
            'pane': 'separate',
            'color': self.params.get('color', '#7B1FA2'),
            'lineWidth': self.params.get('lineWidth', 2),
            'priceScaleId': self.params.get('priceScaleId', self.name.lower()),
            'priceLineVisible': False,
            'lastValueVisible': True,
        }


def validate_period(period: int, min_period: int = 2) -> int:
    """Validate period parameter"""
    if not isinstance(period, int):
        raise ValueError(f"Period must be integer, got {type(period)}")

    if period < min_period:
        raise ValueError(f"Period must be >= {min_period}, got {period}")

    return period


if __name__ == '__main__':
    # Example usage
    print("Indicator base classes ready")
    print("Create specific indicators by subclassing Indicator, OverlayIndicator, or SeparatePaneIndicator")
