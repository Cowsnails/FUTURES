"""
Gravity Indicator - Dynamic Price Magnet
=========================================

Identifies the key price level acting as a "magnet" that pulls price toward it.

Gravity is calculated using multiple factors:
1. Volume Profile POC (Point of Control) - highest volume node
2. VWAP (Volume Weighted Average Price)
3. Recent swing highs/lows
4. Fair Value Gaps (unfilled gaps)
5. High Time Frame levels

The Gravity level acts as:
- ABOVE price = Bullish bias (price pulled up)
- BELOW price = Bearish bias (price pulled down)
- AT price = Equilibrium (choppy, no bias)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import deque
import numpy as np


@dataclass
class GravityLevel:
    """A gravity level with its properties"""
    price: float
    strength: float  # 0-100, how strong the gravity is
    level_type: str  # "POC", "VWAP", "SWING_HIGH", "SWING_LOW", "FVG", "HTF"
    volume: int = 0  # Volume at this level


@dataclass
class GravityMetrics:
    """Complete gravity analysis for current bar"""
    timestamp: any

    # Primary Gravity Level
    gravity_price: float = 0.0
    gravity_strength: float = 0.0  # 0-100

    # Gravity Position
    gravity_above_price: bool = False
    gravity_below_price: bool = False
    gravity_at_price: bool = False  # Within threshold

    # Distance Metrics
    distance_to_gravity: float = 0.0  # In price points
    distance_percent: float = 0.0  # As % of price

    # Contributing Levels
    poc_level: float = 0.0
    vwap_level: float = 0.0
    swing_high: float = 0.0
    swing_low: float = 0.0

    # Gravity Bias
    bias: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    bias_strength: float = 0.0  # 0-100

    # All detected levels
    all_levels: List[GravityLevel] = None

    def __post_init__(self):
        if self.all_levels is None:
            self.all_levels = []


class GravityAnalyzer:
    """
    Dynamic Gravity Calculator

    Identifies the dominant price level acting as a magnet for price action.
    Combines multiple technical factors to find institutional interest zones.
    """

    def __init__(self, lookback_period: int = 100, volume_profile_bins: int = 50):
        """
        Initialize gravity analyzer

        Args:
            lookback_period: Bars to analyze for swing points
            volume_profile_bins: Number of price bins for volume profile
        """
        self.lookback_period = lookback_period
        self.volume_profile_bins = volume_profile_bins

        # Price and volume history
        self.prices: deque = deque(maxlen=lookback_period)
        self.highs: deque = deque(maxlen=lookback_period)
        self.lows: deque = deque(maxlen=lookback_period)
        self.closes: deque = deque(maxlen=lookback_period)
        self.volumes: deque = deque(maxlen=lookback_period)

        # VWAP tracking (cumulative)
        self.vwap_sum_pv: float = 0.0
        self.vwap_sum_v: float = 0.0

        # Fair Value Gaps
        self.fvgs: List[Tuple[float, float]] = []  # (top, bottom) of gaps

        # Metrics history
        self.metrics_history: deque = deque(maxlen=100)

    def update(self, timestamp: any, open_price: float, high: float,
               low: float, close: float, volume: int) -> GravityMetrics:
        """
        Update gravity analysis with new bar

        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume

        Returns:
            GravityMetrics with current gravity analysis
        """
        # Add to history
        typical_price = (high + low + close) / 3.0
        self.prices.append(typical_price)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(volume)

        # Update VWAP
        self.vwap_sum_pv += typical_price * volume
        self.vwap_sum_v += volume

        metrics = GravityMetrics(timestamp=timestamp)

        if len(self.prices) < 20:
            # Not enough data yet
            return metrics

        # Detect Fair Value Gaps
        self._detect_fvgs()

        # Calculate all gravity levels
        all_levels = []

        # 1. Volume Profile POC
        poc_level = self._calculate_poc()
        if poc_level:
            all_levels.append(GravityLevel(
                price=poc_level.price,
                strength=poc_level.strength,
                level_type="POC",
                volume=poc_level.volume
            ))
            metrics.poc_level = poc_level.price

        # 2. VWAP
        vwap = self._calculate_vwap()
        if vwap > 0:
            all_levels.append(GravityLevel(
                price=vwap,
                strength=70.0,  # VWAP always has high importance
                level_type="VWAP"
            ))
            metrics.vwap_level = vwap

        # 3. Recent Swing High
        swing_high = self._find_swing_high()
        if swing_high > 0:
            all_levels.append(GravityLevel(
                price=swing_high,
                strength=60.0,
                level_type="SWING_HIGH"
            ))
            metrics.swing_high = swing_high

        # 4. Recent Swing Low
        swing_low = self._find_swing_low()
        if swing_low > 0:
            all_levels.append(GravityLevel(
                price=swing_low,
                strength=60.0,
                level_type="SWING_LOW"
            ))
            metrics.swing_low = swing_low

        # 5. Unfilled Fair Value Gaps
        current_price = close
        for fvg_top, fvg_bottom in self.fvgs:
            # Check if FVG is near current price (within 2%)
            fvg_mid = (fvg_top + fvg_bottom) / 2.0
            distance = abs(current_price - fvg_mid) / current_price

            if distance <= 0.02:  # Within 2%
                all_levels.append(GravityLevel(
                    price=fvg_mid,
                    strength=50.0,
                    level_type="FVG"
                ))

        # Calculate weighted average gravity level
        if all_levels:
            total_weight = sum(level.strength for level in all_levels)
            weighted_price = sum(level.price * level.strength for level in all_levels)
            metrics.gravity_price = weighted_price / total_weight if total_weight > 0 else close

            # Calculate overall gravity strength
            # More levels agreeing = stronger gravity
            price_std = np.std([level.price for level in all_levels])
            price_range = max(level.price for level in all_levels) - min(level.price for level in all_levels)

            # Low std deviation = tight cluster = strong gravity
            if price_range > 0:
                cluster_score = 1.0 - min(1.0, price_std / price_range)
                metrics.gravity_strength = cluster_score * 100
            else:
                metrics.gravity_strength = 100.0  # Perfect agreement

            metrics.all_levels = all_levels
        else:
            # No levels found, use current price as gravity
            metrics.gravity_price = close
            metrics.gravity_strength = 0.0

        # Determine gravity position relative to price
        threshold = close * 0.002  # 0.2% threshold for "at price"

        distance = metrics.gravity_price - close
        metrics.distance_to_gravity = distance
        metrics.distance_percent = (distance / close) * 100

        if abs(distance) <= threshold:
            metrics.gravity_at_price = True
            metrics.bias = "NEUTRAL"
        elif distance > 0:
            metrics.gravity_above_price = True
            metrics.bias = "BULLISH"
            # Strength based on distance (further = weaker)
            max_distance = close * 0.02  # 2% = max distance
            metrics.bias_strength = max(0, min(100, (1.0 - abs(distance) / max_distance) * 100))
        else:
            metrics.gravity_below_price = True
            metrics.bias = "BEARISH"
            max_distance = close * 0.02
            metrics.bias_strength = max(0, min(100, (1.0 - abs(distance) / max_distance) * 100))

        # Store metrics
        self.metrics_history.append(metrics)

        return metrics

    def _calculate_poc(self) -> Optional[GravityLevel]:
        """
        Calculate Point of Control (POC) from volume profile

        Returns:
            GravityLevel for POC or None
        """
        if len(self.prices) < 20:
            return None

        # Get price range
        highs_list = list(self.highs)
        lows_list = list(self.lows)
        volumes_list = list(self.volumes)
        closes_list = list(self.closes)

        price_min = min(lows_list)
        price_max = max(highs_list)
        price_range = price_max - price_min

        if price_range == 0:
            return None

        # Create price bins
        bins = np.linspace(price_min, price_max, self.volume_profile_bins)
        volume_profile = np.zeros(len(bins) - 1)

        # Distribute volume across bins
        for i in range(len(closes_list)):
            high = highs_list[i]
            low = lows_list[i]
            volume = volumes_list[i]

            # Find bins this bar touched
            for j in range(len(bins) - 1):
                bin_low = bins[j]
                bin_high = bins[j + 1]

                # Check if bar overlaps this bin
                if not (high < bin_low or low > bin_high):
                    # Distribute volume proportionally
                    overlap = min(high, bin_high) - max(low, bin_low)
                    bar_range = high - low
                    if bar_range > 0:
                        volume_in_bin = volume * (overlap / bar_range)
                        volume_profile[j] += volume_in_bin

        # Find POC (bin with most volume)
        if np.sum(volume_profile) > 0:
            poc_index = np.argmax(volume_profile)
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2.0
            poc_volume = volume_profile[poc_index]

            # Calculate POC strength based on volume concentration
            total_volume = np.sum(volume_profile)
            concentration = poc_volume / total_volume if total_volume > 0 else 0
            strength = min(100.0, concentration * 200)  # Scale to 0-100

            return GravityLevel(
                price=poc_price,
                strength=strength,
                level_type="POC",
                volume=int(poc_volume)
            )

        return None

    def _calculate_vwap(self) -> float:
        """Calculate current VWAP"""
        if self.vwap_sum_v > 0:
            return self.vwap_sum_pv / self.vwap_sum_v
        return 0.0

    def _find_swing_high(self, lookback: int = 20) -> float:
        """
        Find most recent significant swing high

        Args:
            lookback: Bars to look back

        Returns:
            Swing high price or 0
        """
        if len(self.highs) < lookback:
            return 0.0

        highs_list = list(self.highs)[-lookback:]

        # Find local maximums
        swing_highs = []
        for i in range(2, len(highs_list) - 2):
            if (highs_list[i] > highs_list[i-1] and
                highs_list[i] > highs_list[i-2] and
                highs_list[i] > highs_list[i+1] and
                highs_list[i] > highs_list[i+2]):
                swing_highs.append(highs_list[i])

        # Return most recent significant swing high
        if swing_highs:
            return swing_highs[-1]

        # If no swing high found, return highest high
        return max(highs_list)

    def _find_swing_low(self, lookback: int = 20) -> float:
        """
        Find most recent significant swing low

        Args:
            lookback: Bars to look back

        Returns:
            Swing low price or 0
        """
        if len(self.lows) < lookback:
            return 0.0

        lows_list = list(self.lows)[-lookback:]

        # Find local minimums
        swing_lows = []
        for i in range(2, len(lows_list) - 2):
            if (lows_list[i] < lows_list[i-1] and
                lows_list[i] < lows_list[i-2] and
                lows_list[i] < lows_list[i+1] and
                lows_list[i] < lows_list[i+2]):
                swing_lows.append(lows_list[i])

        # Return most recent significant swing low
        if swing_lows:
            return swing_lows[-1]

        # If no swing low found, return lowest low
        return min(lows_list)

    def _detect_fvgs(self) -> None:
        """
        Detect Fair Value Gaps (unfilled gaps)

        A Fair Value Gap occurs when:
        - Bar[i-1] high < Bar[i+1] low (bullish FVG)
        - Bar[i-1] low > Bar[i+1] high (bearish FVG)
        """
        if len(self.highs) < 3:
            return

        highs_list = list(self.highs)
        lows_list = list(self.lows)

        # Check last 3 bars for new FVG
        if len(highs_list) >= 3:
            # Bullish FVG
            if highs_list[-3] < lows_list[-1]:
                gap = (lows_list[-1], highs_list[-3])
                # Check if price has filled this gap
                current_low = lows_list[-1]
                if current_low > gap[1]:  # Gap not filled
                    # Add if not already in list
                    if gap not in self.fvgs:
                        self.fvgs.append(gap)

            # Bearish FVG
            if lows_list[-3] > highs_list[-1]:
                gap = (lows_list[-3], highs_list[-1])
                # Check if price has filled this gap
                current_high = highs_list[-1]
                if current_high < gap[0]:  # Gap not filled
                    if gap not in self.fvgs:
                        self.fvgs.append(gap)

        # Remove filled gaps
        current_high = highs_list[-1]
        current_low = lows_list[-1]
        self.fvgs = [
            (top, bottom) for top, bottom in self.fvgs
            if not (current_low <= bottom and current_high >= top)
        ]

        # Keep only recent gaps (max 10)
        if len(self.fvgs) > 10:
            self.fvgs = self.fvgs[-10:]

    def get_current_metrics(self) -> Optional[GravityMetrics]:
        """Get most recent gravity metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def reset_vwap(self):
        """Reset VWAP calculation (for new day/session)"""
        self.vwap_sum_pv = 0.0
        self.vwap_sum_v = 0.0
