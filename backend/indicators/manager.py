"""
Indicator Manager

Registry and management system for technical indicators.
"""

import logging
from typing import Dict, Type, List, Optional, Any
import pandas as pd

from .base import Indicator
from .moving_averages import SMA, EMA, WMA, VWAP, BollingerBands
from .oscillators import RSI, MACD, Stochastic, CCI, ROC
from .volume_delta import VolumeDelta, CumulativeDelta, DeltaDivergence

logger = logging.getLogger(__name__)


# Global indicator registry
INDICATOR_REGISTRY: Dict[str, Type[Indicator]] = {
    # Moving Averages
    'sma': SMA,
    'ema': EMA,
    'wma': WMA,
    'vwap': VWAP,
    'bb': BollingerBands,
    'bollinger': BollingerBands,

    # Oscillators
    'rsi': RSI,
    'macd': MACD,
    'stoch': Stochastic,
    'stochastic': Stochastic,
    'cci': CCI,
    'roc': ROC,

    # Volume Delta (TastyTrade order flow)
    'volume_delta': VolumeDelta,
    'delta': VolumeDelta,
    'cvd': CumulativeDelta,
    'cumulative_delta': CumulativeDelta,
    'delta_divergence': DeltaDivergence,
}


def list_available_indicators() -> List[Dict[str, Any]]:
    """
    List all available indicators with metadata.

    Returns:
        List of indicator metadata dictionaries
    """
    indicators = []

    for key, indicator_class in INDICATOR_REGISTRY.items():
        # Create instance with defaults to get info
        try:
            instance = indicator_class()
            indicators.append({
                'key': key,
                'name': instance.name,
                'display_name': instance.get_display_name(),
                'default_params': instance.params,
                'plot_config': instance.plot_config
            })
        except Exception as e:
            logger.error(f"Error listing indicator {key}: {e}")

    return indicators


class IndicatorManager:
    """
    Manages active indicators and their calculations.

    Handles adding, removing, and calculating indicators.
    """

    def __init__(self):
        """Initialize indicator manager"""
        self.active_indicators: Dict[str, Indicator] = {}
        self.calculation_cache: Dict[str, pd.DataFrame] = {}

    def add_indicator(
        self,
        indicator_type: str,
        params: Optional[Dict[str, Any]] = None,
        indicator_id: Optional[str] = None
    ) -> Optional[Indicator]:
        """
        Add an indicator.

        Args:
            indicator_type: Type of indicator (e.g., 'sma', 'rsi')
            params: Indicator parameters
            indicator_id: Optional custom ID (auto-generated if not provided)

        Returns:
            Indicator instance or None if error
        """
        try:
            # Get indicator class
            indicator_class = INDICATOR_REGISTRY.get(indicator_type.lower())

            if not indicator_class:
                logger.error(f"Unknown indicator type: {indicator_type}")
                logger.info(f"Available indicators: {list(INDICATOR_REGISTRY.keys())}")
                return None

            # Create indicator instance
            indicator = indicator_class(**(params or {}))

            # Use custom ID or generated ID
            ind_id = indicator_id or indicator.id

            # Add to active indicators
            self.active_indicators[ind_id] = indicator

            logger.info(f"Added indicator: {indicator.get_display_name()} (ID: {ind_id})")

            return indicator

        except Exception as e:
            logger.error(f"Error adding indicator {indicator_type}: {e}")
            return None

    def remove_indicator(self, indicator_id: str) -> bool:
        """
        Remove an indicator.

        Args:
            indicator_id: ID of indicator to remove

        Returns:
            True if removed, False if not found
        """
        if indicator_id in self.active_indicators:
            indicator = self.active_indicators[indicator_id]
            del self.active_indicators[indicator_id]

            # Clear cache
            if indicator_id in self.calculation_cache:
                del self.calculation_cache[indicator_id]

            logger.info(f"Removed indicator: {indicator.get_display_name()}")
            return True

        logger.warning(f"Indicator not found: {indicator_id}")
        return False

    def calculate_all(
        self,
        df: pd.DataFrame,
        use_cache: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all active indicators.

        Args:
            df: DataFrame with OHLCV data
            use_cache: Whether to use cached results

        Returns:
            Dictionary mapping indicator ID to calculation results:
            {
                'indicator_id': {
                    'data': [...],  # List of data points
                    'config': {...},  # Plot configuration
                    'metadata': {...}  # Additional metadata
                }
            }
        """
        results = {}

        for ind_id, indicator in self.active_indicators.items():
            try:
                # Check cache
                if use_cache and ind_id in self.calculation_cache:
                    cached_data = self.calculation_cache[ind_id]
                    logger.debug(f"Using cached data for {indicator.get_display_name()}")
                else:
                    # Calculate indicator
                    cached_data = indicator.calculate_safe(df)

                    if cached_data is None:
                        logger.warning(f"Failed to calculate {indicator.get_display_name()}")
                        continue

                    # Cache result
                    if use_cache:
                        self.calculation_cache[ind_id] = cached_data

                # Convert to serializable format
                data_list = cached_data.to_dict('records')

                results[ind_id] = {
                    'data': data_list,
                    'config': indicator.plot_config,
                    'metadata': {
                        'name': indicator.name,
                        'display_name': indicator.get_display_name(),
                        'params': indicator.params,
                        'data_points': len(data_list)
                    }
                }

                logger.debug(
                    f"Calculated {indicator.get_display_name()}: "
                    f"{len(data_list)} data points"
                )

            except Exception as e:
                logger.error(f"Error calculating {ind_id}: {e}")

        return results

    def update_indicator(
        self,
        indicator_id: str,
        new_bar: Dict[str, Any],
        recent_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Incrementally update a single indicator with new bar data.

        For efficiency, only recalculates using recent data.

        Args:
            indicator_id: ID of indicator to update
            new_bar: New bar data
            recent_df: Recent bars (last 100-200 for context)

        Returns:
            Updated indicator value or None if error
        """
        if indicator_id not in self.active_indicators:
            return None

        try:
            indicator = self.active_indicators[indicator_id]

            # Calculate using recent data
            result_df = indicator.calculate_safe(recent_df)

            if result_df is None or len(result_df) == 0:
                return None

            # Get most recent value
            latest = result_df.iloc[-1].to_dict()

            return {
                'indicator_id': indicator_id,
                'data': latest,
                'is_new_bar': True  # Simplified - could check if bar time changed
            }

        except Exception as e:
            logger.error(f"Error updating indicator {indicator_id}: {e}")
            return None

    def get_indicator_info(self, indicator_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an active indicator"""
        if indicator_id not in self.active_indicators:
            return None

        indicator = self.active_indicators[indicator_id]
        return indicator.to_dict()

    def list_active_indicators(self) -> List[Dict[str, Any]]:
        """List all active indicators"""
        return [
            {
                'id': ind_id,
                **indicator.to_dict()
            }
            for ind_id, indicator in self.active_indicators.items()
        ]

    def clear_all_indicators(self):
        """Remove all active indicators"""
        count = len(self.active_indicators)
        self.active_indicators.clear()
        self.calculation_cache.clear()
        logger.info(f"Cleared {count} indicators")

    def clear_cache(self):
        """Clear calculation cache"""
        self.calculation_cache.clear()
        logger.info("Cleared indicator calculation cache")

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            'active_indicators': len(self.active_indicators),
            'cached_calculations': len(self.calculation_cache),
            'indicators': self.list_active_indicators()
        }


# Convenience function for quick indicator creation
def create_indicator(indicator_type: str, **params) -> Optional[Indicator]:
    """
    Create an indicator instance.

    Args:
        indicator_type: Type of indicator
        **params: Indicator parameters

    Returns:
        Indicator instance or None

    Example:
        >>> sma = create_indicator('sma', period=20, color='#2962FF')
        >>> rsi = create_indicator('rsi', period=14)
    """
    indicator_class = INDICATOR_REGISTRY.get(indicator_type.lower())

    if not indicator_class:
        logger.error(f"Unknown indicator type: {indicator_type}")
        return None

    try:
        return indicator_class(**params)
    except Exception as e:
        logger.error(f"Error creating indicator: {e}")
        return None


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Available Indicators:")
    print("=" * 60)

    for ind in list_available_indicators():
        print(f"\n{ind['key'].upper()}: {ind['display_name']}")
        print(f"  Default params: {ind['default_params']}")

    print("\n" + "=" * 60)
    print("\nTesting Indicator Manager:")

    # Create manager
    manager = IndicatorManager()

    # Add indicators
    manager.add_indicator('sma', {'period': 20, 'color': '#2962FF'})
    manager.add_indicator('ema', {'period': 50, 'color': '#FF6D00'})
    manager.add_indicator('rsi', {'period': 14})
    manager.add_indicator('macd')

    # List active
    print(f"\nActive indicators: {len(manager.active_indicators)}")
    for ind_info in manager.list_active_indicators():
        print(f"  - {ind_info['display_name']}")

    # Create sample data
    import numpy as np
    sample_df = pd.DataFrame({
        'time': range(100),
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    # Calculate all
    print("\nCalculating indicators...")
    results = manager.calculate_all(sample_df)

    for ind_id, result in results.items():
        print(f"\n{result['metadata']['display_name']}:")
        print(f"  Data points: {result['metadata']['data_points']}")
        print(f"  Config: {result['config']['type']}, pane: {result['config']['pane']}")

    # Statistics
    print(f"\nStatistics: {manager.get_statistics()}")
