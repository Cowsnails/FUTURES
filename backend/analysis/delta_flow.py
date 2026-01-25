"""
Advanced Delta Flow Analysis System
====================================

Tracks tick-by-tick delta (buy vs sell volume) to identify:
- Institutional order flow
- Support/resistance zones
- Momentum shifts
- Absorption patterns
- Exhaustion signals

Delta Flow = Buy Volume - Sell Volume on each tick
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
import numpy as np


@dataclass
class TickData:
    """Individual tick data point"""
    time: datetime
    price: float
    size: int
    is_buy: bool  # True if aggressor was buyer, False if seller


@dataclass
class DeltaMetrics:
    """Comprehensive delta flow metrics for a bar"""
    timestamp: datetime

    # Basic Delta
    buy_volume: int = 0
    sell_volume: int = 0
    total_volume: int = 0
    raw_delta: int = 0  # buy_volume - sell_volume
    delta_percent: float = 0.0  # (raw_delta / total_volume) * 100

    # Cumulative Delta
    cumulative_delta: int = 0

    # Delta Momentum (rate of change)
    delta_momentum: float = 0.0
    delta_acceleration: float = 0.0

    # Price vs Delta Analysis
    price_change: float = 0.0
    delta_divergence: bool = False  # True if price and delta move opposite

    # Advanced Metrics
    large_trade_delta: int = 0  # Delta from trades > avg size
    small_trade_delta: int = 0  # Delta from trades < avg size

    # Absorption Analysis
    absorption_score: float = 0.0  # High volume, low price movement = absorption

    # Exhaustion Signals
    exhaustion_level: float = 0.0  # High delta, slowing momentum = exhaustion

    # Tick Statistics
    tick_count: int = 0
    avg_trade_size: float = 0.0
    max_trade_size: int = 0

    # Trade Distribution
    trades_at_ask: int = 0  # Aggressive buys
    trades_at_bid: int = 0  # Aggressive sells
    trades_between: int = 0  # Mid-spread trades


class DeltaFlowAnalyzer:
    """
    Advanced Delta Flow Analysis Engine

    Processes tick-by-tick data to calculate comprehensive delta metrics.
    Tracks multiple factors simultaneously for institutional flow detection.
    """

    def __init__(self, lookback_bars: int = 20):
        """
        Initialize delta flow analyzer

        Args:
            lookback_bars: Number of bars to keep in history for calculations
        """
        self.lookback_bars = lookback_bars

        # Current bar tracking
        self.current_bar_ticks: List[TickData] = []
        self.current_bar_start: Optional[datetime] = None
        self.last_price: Optional[float] = None

        # Historical metrics
        self.metrics_history: deque = deque(maxlen=lookback_bars)

        # Cumulative delta tracking
        self.cumulative_delta: int = 0

        # Previous bar metrics for momentum calculations
        self.prev_delta: Optional[int] = None
        self.prev_momentum: Optional[float] = None

    def process_tick(self, timestamp: datetime, price: float, size: int) -> None:
        """
        Process individual tick and classify as buy or sell

        Tick Classification:
        - Buy: Price >= last price (aggressor buyer)
        - Sell: Price < last price (aggressor seller)
        - First tick: Assume neutral (classify by next tick direction)

        Args:
            timestamp: Tick timestamp
            price: Trade price
            size: Trade size (contracts)
        """
        # Classify tick as buy or sell
        if self.last_price is None:
            # First tick - assume neutral, will classify next tick
            is_buy = True  # Default assumption
        else:
            # Classify based on price movement
            is_buy = price >= self.last_price

        # Create tick data
        tick = TickData(
            time=timestamp,
            price=price,
            size=size,
            is_buy=is_buy
        )

        # Add to current bar
        self.current_bar_ticks.append(tick)

        # Update last price
        self.last_price = price

    def finalize_bar(self,
                     bar_timestamp: datetime,
                     open_price: float,
                     close_price: float) -> DeltaMetrics:
        """
        Calculate all delta metrics for completed bar

        Args:
            bar_timestamp: Bar timestamp
            open_price: Bar open price
            close_price: Bar close price

        Returns:
            DeltaMetrics object with all calculated metrics
        """
        metrics = DeltaMetrics(timestamp=bar_timestamp)

        if not self.current_bar_ticks:
            # No ticks in this bar
            self.metrics_history.append(metrics)
            return metrics

        # Calculate basic volume metrics
        buy_ticks = [t for t in self.current_bar_ticks if t.is_buy]
        sell_ticks = [t for t in self.current_bar_ticks if not t.is_buy]

        metrics.buy_volume = sum(t.size for t in buy_ticks)
        metrics.sell_volume = sum(t.size for t in sell_ticks)
        metrics.total_volume = metrics.buy_volume + metrics.sell_volume
        metrics.raw_delta = metrics.buy_volume - metrics.sell_volume

        if metrics.total_volume > 0:
            metrics.delta_percent = (metrics.raw_delta / metrics.total_volume) * 100

        # Update cumulative delta
        self.cumulative_delta += metrics.raw_delta
        metrics.cumulative_delta = self.cumulative_delta

        # Calculate delta momentum (rate of change)
        if self.prev_delta is not None:
            metrics.delta_momentum = metrics.raw_delta - self.prev_delta

            # Calculate delta acceleration (change in momentum)
            if self.prev_momentum is not None:
                metrics.delta_acceleration = metrics.delta_momentum - self.prev_momentum

        # Price vs Delta Analysis
        metrics.price_change = close_price - open_price

        # Delta Divergence: Price and delta move in opposite directions
        # Bullish Divergence: Price down, delta up (accumulation)
        # Bearish Divergence: Price up, delta down (distribution)
        if metrics.price_change > 0 and metrics.raw_delta < 0:
            metrics.delta_divergence = True  # Bearish divergence
        elif metrics.price_change < 0 and metrics.raw_delta > 0:
            metrics.delta_divergence = True  # Bullish divergence

        # Tick statistics
        metrics.tick_count = len(self.current_bar_ticks)
        all_sizes = [t.size for t in self.current_bar_ticks]
        metrics.avg_trade_size = np.mean(all_sizes) if all_sizes else 0
        metrics.max_trade_size = max(all_sizes) if all_sizes else 0

        # Large vs Small trade analysis
        avg_size = metrics.avg_trade_size
        large_trades = [t for t in self.current_bar_ticks if t.size > avg_size]
        small_trades = [t for t in self.current_bar_ticks if t.size <= avg_size]

        metrics.large_trade_delta = sum(
            t.size if t.is_buy else -t.size for t in large_trades
        )
        metrics.small_trade_delta = sum(
            t.size if t.is_buy else -t.size for t in small_trades
        )

        # Trade distribution (ask/bid/between)
        metrics.trades_at_ask = len(buy_ticks)
        metrics.trades_at_bid = len(sell_ticks)

        # Absorption Analysis
        # High volume + low price movement = absorption zone
        price_range = abs(metrics.price_change)
        if metrics.total_volume > 0 and price_range > 0:
            metrics.absorption_score = metrics.total_volume / price_range
        elif metrics.total_volume > 0:
            # High volume, no price movement = maximum absorption
            metrics.absorption_score = 1000.0

        # Exhaustion Analysis
        # High delta + slowing momentum = potential exhaustion
        if abs(metrics.raw_delta) > 0 and metrics.delta_momentum != 0:
            # Normalize by total volume
            delta_strength = abs(metrics.raw_delta) / metrics.total_volume
            momentum_ratio = abs(metrics.delta_acceleration) / abs(metrics.delta_momentum) if metrics.delta_momentum != 0 else 0
            metrics.exhaustion_level = delta_strength * (1 - momentum_ratio)

        # Store for next iteration
        self.prev_delta = metrics.raw_delta
        self.prev_momentum = metrics.delta_momentum

        # Add to history
        self.metrics_history.append(metrics)

        # Clear current bar
        self.current_bar_ticks = []

        return metrics

    def get_delta_profile(self, num_bars: int = 10) -> Dict[str, any]:
        """
        Get comprehensive delta profile over last N bars

        Args:
            num_bars: Number of bars to analyze

        Returns:
            Dictionary with profile metrics
        """
        if not self.metrics_history:
            return {}

        recent_metrics = list(self.metrics_history)[-num_bars:]

        total_buy = sum(m.buy_volume for m in recent_metrics)
        total_sell = sum(m.sell_volume for m in recent_metrics)
        total_delta = sum(m.raw_delta for m in recent_metrics)

        # Delta trend (increasing or decreasing)
        deltas = [m.raw_delta for m in recent_metrics]
        delta_trend = "increasing" if len(deltas) > 1 and deltas[-1] > deltas[0] else "decreasing"

        # Divergence count
        divergence_count = sum(1 for m in recent_metrics if m.delta_divergence)

        # Average absorption
        avg_absorption = np.mean([m.absorption_score for m in recent_metrics])

        # Average exhaustion
        avg_exhaustion = np.mean([m.exhaustion_level for m in recent_metrics])

        return {
            "num_bars": len(recent_metrics),
            "total_buy_volume": total_buy,
            "total_sell_volume": total_sell,
            "net_delta": total_delta,
            "delta_ratio": total_buy / total_sell if total_sell > 0 else 0,
            "delta_trend": delta_trend,
            "divergence_count": divergence_count,
            "avg_absorption": avg_absorption,
            "avg_exhaustion": avg_exhaustion,
            "cumulative_delta": self.cumulative_delta
        }

    def get_current_metrics(self) -> Optional[DeltaMetrics]:
        """Get most recent delta metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def reset(self):
        """Reset all state (useful for new day or session)"""
        self.current_bar_ticks = []
        self.current_bar_start = None
        self.last_price = None
        self.metrics_history.clear()
        self.cumulative_delta = 0
        self.prev_delta = None
        self.prev_momentum = None
