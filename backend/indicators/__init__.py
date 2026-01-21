"""
Indicator system for technical analysis
"""
from .base import Indicator, OverlayIndicator, SeparatePaneIndicator
from .moving_averages import SMA, EMA, WMA, VWAP, BollingerBands
from .oscillators import RSI, MACD, Stochastic, CCI, ROC
from .manager import (
    IndicatorManager,
    INDICATOR_REGISTRY,
    list_available_indicators,
    create_indicator
)

__all__ = [
    # Base classes
    'Indicator',
    'OverlayIndicator',
    'SeparatePaneIndicator',

    # Moving averages
    'SMA',
    'EMA',
    'WMA',
    'VWAP',
    'BollingerBands',

    # Oscillators
    'RSI',
    'MACD',
    'Stochastic',
    'CCI',
    'ROC',

    # Manager
    'IndicatorManager',
    'INDICATOR_REGISTRY',
    'list_available_indicators',
    'create_indicator',
]
