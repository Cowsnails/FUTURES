"""
Indicator system for technical analysis
"""
from .base import Indicator, OverlayIndicator, SeparatePaneIndicator
from .moving_averages import SMA, EMA, WMA, VWAP, BollingerBands
from .oscillators import RSI, MACD, Stochastic, CCI, ROC
from .volume_delta import VolumeDelta, CumulativeDelta, DeltaDivergence
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

    # Volume delta (TastyTrade order flow)
    'VolumeDelta',
    'CumulativeDelta',
    'DeltaDivergence',

    # Manager
    'IndicatorManager',
    'INDICATOR_REGISTRY',
    'list_available_indicators',
    'create_indicator',
]
