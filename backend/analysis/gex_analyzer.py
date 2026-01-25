"""
GEX (Gamma Exposure) Analyzer
==============================

Estimates gamma exposure regime using price action and volume analysis.

Since we don't have direct options data, this creates a sophisticated proxy
that estimates whether the market is in a SHORT GAMMA or LONG GAMMA regime.

SHORT GAMMA (GEX Ratio < 0.35):
- Market makers are short gamma
- Their hedging ADDS to momentum
- Trends are allowed to develop
- Expect directional moves

LONG GAMMA (GEX Ratio >= 0.35):
- Market makers are long gamma
- Their hedging DAMPENS moves
- Price gets pinned/chops
- Expect mean reversion
"""

from dataclasses import dataclass
from typing import List, Optional
from collections import deque
import numpy as np


@dataclass
class GEXMetrics:
    """GEX regime metrics"""
    timestamp: any

    # Core GEX Ratio (0.0 to 1.0+)
    gex_ratio: float = 0.5

    # Gamma Regime Classification
    regime: str = "MIXED"  # "SHORT_GAMMA", "LONG_GAMMA", "MIXED"
    regime_strength: float = 0.0  # 0-100, how strong the regime is

    # Component Factors
    volatility_factor: float = 0.0  # High vol = short gamma
    volume_concentration: float = 0.0  # Dispersed = short gamma, concentrated = long gamma
    reversion_tendency: float = 0.0  # High reversion = long gamma
    round_number_pin: float = 0.0  # High pin = long gamma

    # Regime Signals
    trending_regime: bool = False  # Short gamma regime
    pinning_regime: bool = False  # Long gamma regime

    # Supporting Data
    current_volatility: float = 0.0
    baseline_volatility: float = 0.0
    price_at_strike: bool = False  # Price near round number


class GEXAnalyzer:
    """
    Gamma Exposure Regime Analyzer

    Estimates whether market is in SHORT GAMMA or LONG GAMMA regime
    using multiple proxy indicators:

    1. Volatility Regime - expanding = short gamma
    2. Volume Distribution - concentrated = long gamma
    3. Mean Reversion - high = long gamma
    4. Strike Pin - strong = long gamma
    """

    def __init__(self, lookback_period: int = 50):
        """
        Initialize GEX analyzer

        Args:
            lookback_period: Bars to analyze for baseline calculations
        """
        self.lookback_period = lookback_period

        # Price history for calculations
        self.prices: deque = deque(maxlen=lookback_period)
        self.volumes: deque = deque(maxlen=lookback_period)
        self.highs: deque = deque(maxlen=lookback_period)
        self.lows: deque = deque(maxlen=lookback_period)

        # Metrics history
        self.metrics_history: deque = deque(maxlen=100)

    def update(self, timestamp: any, open_price: float, high: float,
               low: float, close: float, volume: int) -> GEXMetrics:
        """
        Update GEX analysis with new bar

        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume

        Returns:
            GEXMetrics with current regime analysis
        """
        # Add to history
        self.prices.append(close)
        self.volumes.append(volume)
        self.highs.append(high)
        self.lows.append(low)

        metrics = GEXMetrics(timestamp=timestamp)

        if len(self.prices) < 20:
            # Not enough data yet
            return metrics

        # Calculate each component factor
        metrics.volatility_factor = self._calculate_volatility_factor()
        metrics.volume_concentration = self._calculate_volume_concentration()
        metrics.reversion_tendency = self._calculate_reversion_tendency()
        metrics.round_number_pin = self._calculate_round_number_pin(close)

        # Calculate GEX Ratio (weighted combination)
        # Lower ratio = SHORT GAMMA (trending)
        # Higher ratio = LONG GAMMA (pinning)
        weights = {
            'volatility': 0.30,      # High vol = short gamma
            'concentration': 0.25,   # High concentration = long gamma
            'reversion': 0.30,       # High reversion = long gamma
            'pin': 0.15              # Strong pin = long gamma
        }

        # Inverse volatility factor (high vol = low GEX ratio)
        gex_components = (
            (1.0 - metrics.volatility_factor) * weights['volatility'] +
            metrics.volume_concentration * weights['concentration'] +
            metrics.reversion_tendency * weights['reversion'] +
            metrics.round_number_pin * weights['pin']
        )

        metrics.gex_ratio = max(0.0, min(1.0, gex_components))

        # Classify regime
        if metrics.gex_ratio < 0.35:
            metrics.regime = "SHORT_GAMMA"
            metrics.trending_regime = True
            metrics.regime_strength = (0.35 - metrics.gex_ratio) / 0.35 * 100
        elif metrics.gex_ratio >= 0.35 and metrics.gex_ratio < 0.65:
            metrics.regime = "MIXED"
            # Strength based on distance from middle (0.5)
            metrics.regime_strength = (1.0 - abs(metrics.gex_ratio - 0.5) / 0.15) * 100
        else:
            metrics.regime = "LONG_GAMMA"
            metrics.pinning_regime = True
            metrics.regime_strength = (metrics.gex_ratio - 0.65) / 0.35 * 100

        # Store volatility data
        metrics.current_volatility = self._calculate_volatility()
        metrics.baseline_volatility = self._calculate_baseline_volatility()

        # Check if near round number
        metrics.price_at_strike = self._is_near_round_number(close)

        # Store metrics
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_volatility_factor(self) -> float:
        """
        Calculate volatility regime factor

        Returns:
            0.0 to 1.0, where 1.0 = very high volatility (short gamma)
        """
        if len(self.highs) < 20:
            return 0.5

        # Calculate Average True Range (ATR)
        ranges = []
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)

        for i in range(1, len(prices_list)):
            true_range = max(
                highs_list[i] - lows_list[i],
                abs(highs_list[i] - prices_list[i-1]),
                abs(lows_list[i] - prices_list[i-1])
            )
            ranges.append(true_range)

        current_atr = np.mean(ranges[-14:]) if len(ranges) >= 14 else np.mean(ranges)
        baseline_atr = np.mean(ranges)

        # Volatility ratio
        if baseline_atr > 0:
            vol_ratio = current_atr / baseline_atr
            # Normalize to 0-1 range (ratio > 1.5 = max)
            return min(1.0, max(0.0, (vol_ratio - 0.5) / 1.0))

        return 0.5

    def _calculate_volume_concentration(self) -> float:
        """
        Calculate volume concentration factor

        High concentration (few big bars) = long gamma (pinning)
        Low concentration (distributed) = short gamma (trending)

        Returns:
            0.0 to 1.0, where 1.0 = very concentrated (long gamma)
        """
        if len(self.volumes) < 20:
            return 0.5

        volumes = list(self.volumes)

        # Calculate coefficient of variation (std/mean)
        mean_vol = np.mean(volumes)
        std_vol = np.std(volumes)

        if mean_vol > 0:
            cv = std_vol / mean_vol
            # Higher CV = more concentrated
            # Normalize: CV of 0.5 = 0.5, CV of 1.0+ = 1.0
            return min(1.0, cv / 1.0)

        return 0.5

    def _calculate_reversion_tendency(self) -> float:
        """
        Calculate mean reversion tendency

        Strong reversion = long gamma (price gets pulled back)
        Weak reversion = short gamma (trends continue)

        Returns:
            0.0 to 1.0, where 1.0 = strong reversion (long gamma)
        """
        if len(self.prices) < 20:
            return 0.5

        prices = np.array(list(self.prices))

        # Calculate price vs moving average
        sma_20 = np.mean(prices[-20:])

        # Count reversions in last 10 bars
        reversions = 0
        for i in range(-10, -1):
            if i-1 >= -len(prices):
                # Check if price crossed MA
                prev_above = prices[i-1] > sma_20
                curr_above = prices[i] > sma_20
                if prev_above != curr_above:
                    reversions += 1

        # Normalize: 5+ reversions = strong reversion (1.0)
        reversion_score = min(1.0, reversions / 5.0)

        # Also factor in current distance from MA
        current_price = prices[-1]
        distance_from_ma = abs(current_price - sma_20) / sma_20

        # If far from MA, reversion tendency increases
        distance_factor = 1.0 - min(1.0, distance_from_ma / 0.02)  # 2% distance = max

        # Combine
        return (reversion_score * 0.6 + distance_factor * 0.4)

    def _calculate_round_number_pin(self, price: float) -> float:
        """
        Calculate round number pinning factor

        Price near round numbers = long gamma (options cluster there)
        Price away from round numbers = short gamma

        Args:
            price: Current price

        Returns:
            0.0 to 1.0, where 1.0 = very close to round number (long gamma)
        """
        # Find nearest round numbers (based on price level)
        if price < 100:
            # Round to nearest 1.0
            round_interval = 1.0
        elif price < 1000:
            # Round to nearest 5.0
            round_interval = 5.0
        elif price < 10000:
            # Round to nearest 10.0
            round_interval = 10.0
        else:
            # Round to nearest 50.0
            round_interval = 50.0

        nearest_round = round(price / round_interval) * round_interval
        distance = abs(price - nearest_round)

        # Normalize distance (0.5% of price = max distance)
        max_distance = price * 0.005
        normalized_distance = min(1.0, distance / max_distance)

        # Invert: close = 1.0, far = 0.0
        return 1.0 - normalized_distance

    def _is_near_round_number(self, price: float, threshold: float = 0.003) -> bool:
        """
        Check if price is near a round number

        Args:
            price: Current price
            threshold: Distance threshold (as fraction of price)

        Returns:
            True if near round number
        """
        if price < 100:
            round_interval = 1.0
        elif price < 1000:
            round_interval = 5.0
        elif price < 10000:
            round_interval = 10.0
        else:
            round_interval = 50.0

        nearest_round = round(price / round_interval) * round_interval
        distance = abs(price - nearest_round)

        return distance <= (price * threshold)

    def _calculate_volatility(self) -> float:
        """Calculate current volatility (ATR based)"""
        if len(self.highs) < 14:
            return 0.0

        ranges = []
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)

        for i in range(1, len(prices_list)):
            true_range = max(
                highs_list[i] - lows_list[i],
                abs(highs_list[i] - prices_list[i-1]),
                abs(lows_list[i] - prices_list[i-1])
            )
            ranges.append(true_range)

        return np.mean(ranges[-14:]) if len(ranges) >= 14 else np.mean(ranges)

    def _calculate_baseline_volatility(self) -> float:
        """Calculate baseline volatility over full lookback period"""
        if len(self.highs) < 20:
            return 0.0

        ranges = []
        prices_list = list(self.prices)
        highs_list = list(self.highs)
        lows_list = list(self.lows)

        for i in range(1, len(prices_list)):
            true_range = max(
                highs_list[i] - lows_list[i],
                abs(highs_list[i] - prices_list[i-1]),
                abs(lows_list[i] - prices_list[i-1])
            )
            ranges.append(true_range)

        return np.mean(ranges)

    def get_current_metrics(self) -> Optional[GEXMetrics]:
        """Get most recent GEX metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def get_regime_history(self, num_bars: int = 20) -> List[str]:
        """Get history of regime classifications"""
        recent = list(self.metrics_history)[-num_bars:]
        return [m.regime for m in recent]
