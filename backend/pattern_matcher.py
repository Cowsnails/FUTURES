"""
Historical Intraday Pattern Matching System

Compares today's unfolding price action against 60 days of historical intraday
templates to find similar days and forecast probable outcomes.

Architecture:
  - Precomputes normalized templates at startup
  - Regime-filters historical pool (ATR, gap type, trend context)
  - Vectorized Pearson correlation for fast matching
  - Runs every 5 minutes from 10:00 AM to 2:00 PM ET
  - Broadcasts results via WebSocket to frontend
"""

import numpy as np
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
from datetime import datetime, time as dtime

import pytz

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("US/Eastern")

# RTH: 9:30 AM - 4:00 PM ET = 390 one-minute bars
RTH_START = dtime(9, 30)
RTH_END = dtime(16, 0)
RTH_BARS = 390

# Matching parameters
MIN_BARS_FOR_MATCH = 30       # Need at least 30 bars (30 min) of today's data
MIN_CORRELATION = 0.80        # Below this = weak match, exclude
MIN_MATCHES_REQUIRED = 3      # Need at least 3 valid matches for a forecast
TOP_N_MATCHES = 5             # Return top 5 most similar days
RECENCY_DECAY = 0.97          # Exponential decay for recency weighting (~23-day half-life)
ATR_TOLERANCE = 0.20          # ±20% ATR for regime filter
GAP_THRESHOLD = 0.003         # 0.3% gap classification threshold


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class GapType(Enum):
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    FLAT = "flat"


class TrendContext(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"


@dataclass
class DayMetadata:
    date: str                       # YYYY-MM-DD
    day_index: int                  # 0-59 (0 = most recent)
    opening_atr: float              # ATR at market open
    gap_percent: float              # Gap from prior close
    gap_type: GapType
    trend_context: TrendContext     # Prior day trend
    daily_return: float             # Full-day return (close vs open) in %
    high_of_day_bar: int            # Bar index of HOD
    low_of_day_bar: int             # Bar index of LOD
    is_half_day: bool               # Early close day
    is_extreme: bool                # |daily_return| > 3%


@dataclass
class PatternMatch:
    date: str
    correlation: float
    weighted_correlation: float     # After recency weighting
    daily_return: float
    projection_prices: list         # Remaining bars after match window (raw prices)
    projection_returns: list        # Remaining bars as % returns from match end
    metadata: dict


@dataclass
class PatternForecast:
    direction_probability: float    # 0-1, >0.5 = bullish
    direction_signal: float         # -1 to +1
    mean_move: float                # Average remaining-day return
    median_move: float
    std_dev: float
    confidence_interval_68: tuple   # (lower, upper) 1-sigma
    sample_size: int
    consensus: str                  # e.g. "4/5 bullish"
    avg_correlation: float
    confluence_score: float         # -1 to +1 for decision engine


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_intraday_prices(prices: np.ndarray) -> np.ndarray:
    """
    Two-stage normalization:
    1. Returns from open (scale-invariant across different price levels)
    2. Z-score (shape comparison)
    """
    if len(prices) == 0:
        return prices
    open_price = prices[0]
    if open_price == 0:
        return np.zeros_like(prices)
    returns_from_open = (prices / open_price - 1.0) * 100.0
    mean = np.mean(returns_from_open)
    std = np.std(returns_from_open)
    if std < 1e-8:
        return np.zeros_like(prices)
    return (returns_from_open - mean) / std


def normalize_volume(volumes: np.ndarray, avg_by_minute: np.ndarray) -> np.ndarray:
    """Time-of-day adjusted volume normalization."""
    relative = volumes / (avg_by_minute + 1e-8)
    mean = np.mean(relative)
    std = np.std(relative)
    if std < 1e-8:
        return np.zeros_like(volumes)
    return (relative - mean) / std


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def fast_correlation_batch(today_series: np.ndarray,
                           historical_matrix: np.ndarray) -> np.ndarray:
    """
    Vectorized Pearson correlation: today vs all historical days at once.
    today_series: (N,)
    historical_matrix: (D, N) where D = number of historical days
    Returns: (D,) correlation values
    """
    if len(today_series) == 0 or historical_matrix.shape[0] == 0:
        return np.array([])

    today_centered = today_series - np.mean(today_series)
    hist_centered = historical_matrix - np.mean(historical_matrix, axis=1, keepdims=True)

    today_norm = np.sqrt(np.sum(today_centered ** 2))
    hist_norms = np.sqrt(np.sum(hist_centered ** 2, axis=1))

    if today_norm < 1e-10:
        return np.zeros(historical_matrix.shape[0])

    numerator = np.dot(hist_centered, today_centered)
    denominator = hist_norms * today_norm + 1e-10

    result = numerator / denominator
    # Clamp to valid range
    return np.clip(result, -1.0, 1.0)


def apply_recency_weight(correlations: np.ndarray,
                          days_ago: np.ndarray,
                          decay: float = RECENCY_DECAY) -> np.ndarray:
    """Weight recent matches more heavily."""
    weights = decay ** days_ago
    return correlations * weights


# ═══════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_gap(gap_percent: float) -> GapType:
    if gap_percent > GAP_THRESHOLD:
        return GapType.GAP_UP
    elif gap_percent < -GAP_THRESHOLD:
        return GapType.GAP_DOWN
    return GapType.FLAT


def classify_trend(closes: np.ndarray, period: int = 9) -> TrendContext:
    """EMA-based trend context from prior day's closes."""
    if len(closes) < period:
        return TrendContext.SIDEWAYS
    # Simple EMA approximation
    ema = np.mean(closes[-period:])
    last_close = closes[-1]
    pct = (last_close - ema) / ema * 100
    if pct > 0.15:
        return TrendContext.UP
    elif pct < -0.15:
        return TrendContext.DOWN
    return TrendContext.SIDEWAYS


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 10) -> float:
    """Compute ATR from OHLC arrays."""
    if len(highs) < period + 1:
        return 0.0
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    return float(np.mean(tr[-period:]))


def filter_by_regime(today_atr: float, today_gap: float,
                     today_trend: TrendContext,
                     metadata: dict[int, DayMetadata]) -> list[int]:
    """
    Filter historical days by similar market context.
    Returns list of day indices that pass the filter.
    Progressive relaxation if too few matches.
    """
    today_gap_type = classify_gap(today_gap)

    # Strict filter: ATR + gap + trend
    strict = []
    for idx, meta in metadata.items():
        if meta.is_half_day or meta.is_extreme:
            continue
        atr_match = (abs(meta.opening_atr - today_atr) /
                     (today_atr + 1e-8)) <= ATR_TOLERANCE
        gap_match = meta.gap_type == today_gap_type
        trend_match = meta.trend_context == today_trend
        if atr_match and gap_match and trend_match:
            strict.append(idx)

    if len(strict) >= 10:
        return strict

    # Relaxed: ATR + gap only
    relaxed = []
    for idx, meta in metadata.items():
        if meta.is_half_day or meta.is_extreme:
            continue
        atr_match = (abs(meta.opening_atr - today_atr) /
                     (today_atr + 1e-8)) <= ATR_TOLERANCE
        gap_match = meta.gap_type == today_gap_type
        if atr_match and gap_match:
            relaxed.append(idx)

    if len(relaxed) >= 5:
        return relaxed

    # Fallback: ATR only
    atr_only = []
    for idx, meta in metadata.items():
        if meta.is_half_day or meta.is_extreme:
            continue
        atr_match = (abs(meta.opening_atr - today_atr) /
                     (today_atr + 1e-8)) <= ATR_TOLERANCE
        if atr_match:
            atr_only.append(idx)

    if len(atr_only) >= 5:
        return atr_only

    # Final fallback: all valid days
    return [idx for idx, meta in metadata.items()
            if not meta.is_half_day and not meta.is_extreme]


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_forecast(matches: list[PatternMatch]) -> Optional[PatternForecast]:
    """
    Build probabilistic forecast from top matched historical days.
    """
    if len(matches) < MIN_MATCHES_REQUIRED:
        return None

    returns = np.array([m.daily_return for m in matches])
    correlations = np.array([m.correlation for m in matches])

    up_count = int(np.sum(returns > 0))
    total = len(returns)
    direction_prob = up_count / total

    return PatternForecast(
        direction_probability=direction_prob,
        direction_signal=(direction_prob - 0.5) * 2.0,  # -1 to +1
        mean_move=float(np.mean(returns)),
        median_move=float(np.median(returns)),
        std_dev=float(np.std(returns)),
        confidence_interval_68=(
            float(np.percentile(returns, 16)),
            float(np.percentile(returns, 84))
        ),
        sample_size=total,
        consensus=f"{up_count}/{total} bullish",
        avg_correlation=float(np.mean(correlations)),
        confluence_score=_compute_confluence_score(direction_prob, total,
                                                   float(np.mean(correlations)))
    )


def _compute_confluence_score(direction_prob: float, sample_size: int,
                               avg_corr: float) -> float:
    """
    Convert pattern match results to -1 to +1 signal for the decision engine.
    """
    base = (direction_prob - 0.5) * 2.0  # -1 to +1
    sample_confidence = min(1.0, np.sqrt(sample_size / 20.0))
    quality_factor = max(0.0, (avg_corr - 0.8) / 0.2)  # 0 at 0.8, 1 at 1.0
    return float(base * sample_confidence * quality_factor)


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class PatternMatchEngine:
    """
    Main engine that holds precomputed templates and runs the matching pipeline.
    """

    def __init__(self):
        # Raw OHLCV: (num_days, 390, 5) — [open, high, low, close, volume]
        self.historical_ohlcv: Optional[np.ndarray] = None
        # Normalized close templates: (num_days, 390)
        self.normalized_templates: Optional[np.ndarray] = None
        # Day metadata indexed by day number
        self.day_metadata: dict[int, DayMetadata] = {}
        # Average volume per minute across all days
        self.volume_avg_by_minute: Optional[np.ndarray] = None
        # Number of historical days loaded
        self.num_days: int = 0
        # Latest match result
        self.latest_result: Optional[dict] = None
        # Track today's bar count for change detection
        self._last_bar_count: int = 0

    def load_historical_data(self, daily_bars_list: list[dict]):
        """
        Load historical data from a list of daily OHLCV arrays.

        Args:
            daily_bars_list: List of dicts, each with:
                - 'date': str (YYYY-MM-DD)
                - 'bars': list of dicts with time/open/high/low/close/volume
                - 'prior_close': float (previous day's close for gap calc)
        """
        valid_days = []
        for day_data in daily_bars_list:
            bars = day_data['bars']
            if len(bars) < RTH_BARS * 0.8:  # Allow some missing bars
                continue
            # Pad to RTH_BARS if slightly short
            ohlcv = np.zeros((RTH_BARS, 5), dtype=np.float64)
            n = min(len(bars), RTH_BARS)
            for i in range(n):
                b = bars[i]
                ohlcv[i] = [b['open'], b['high'], b['low'], b['close'], b['volume']]
            # Forward-fill any remaining bars with last known values
            if n < RTH_BARS:
                ohlcv[n:] = ohlcv[n - 1]
            valid_days.append({
                'date': day_data['date'],
                'ohlcv': ohlcv,
                'prior_close': day_data.get('prior_close', ohlcv[0, 0])
            })

        self.num_days = len(valid_days)
        if self.num_days == 0:
            logger.warning("No valid historical days loaded for pattern matching")
            return

        # Build arrays
        self.historical_ohlcv = np.zeros((self.num_days, RTH_BARS, 5), dtype=np.float64)
        self.normalized_templates = np.zeros((self.num_days, RTH_BARS), dtype=np.float64)

        for i, day in enumerate(valid_days):
            self.historical_ohlcv[i] = day['ohlcv']
            closes = day['ohlcv'][:, 3]
            self.normalized_templates[i] = normalize_intraday_prices(closes)

            # Compute metadata
            opens = day['ohlcv'][:, 0]
            highs = day['ohlcv'][:, 1]
            lows = day['ohlcv'][:, 2]
            volumes = day['ohlcv'][:, 4]

            opening_atr = compute_atr(highs[:30], lows[:30], closes[:30], period=10)
            prior_close = day['prior_close']
            gap_pct = (opens[0] - prior_close) / prior_close if prior_close else 0.0
            daily_ret = (closes[-1] / opens[0] - 1.0) * 100.0 if opens[0] else 0.0

            self.day_metadata[i] = DayMetadata(
                date=day['date'],
                day_index=i,
                opening_atr=opening_atr,
                gap_percent=gap_pct,
                gap_type=classify_gap(gap_pct),
                trend_context=TrendContext.SIDEWAYS,  # Will be set from prior day
                daily_return=daily_ret,
                high_of_day_bar=int(np.argmax(highs)),
                low_of_day_bar=int(np.argmin(lows)),
                is_half_day=len([v for v in volumes if v > 0]) < RTH_BARS * 0.7,
                is_extreme=abs(daily_ret) > 3.0
            )

        # Set trend context from prior day
        for i in range(1, self.num_days):
            prior_closes = self.historical_ohlcv[i - 1, :, 3]
            self.day_metadata[i].trend_context = classify_trend(prior_closes)

        # Volume average by minute
        all_vols = self.historical_ohlcv[:, :, 4]
        self.volume_avg_by_minute = np.mean(all_vols, axis=0)

        logger.info(f"Pattern matcher loaded {self.num_days} historical days "
                     f"({self.num_days * RTH_BARS} total bars)")

    def run_match(self, today_bars: list[dict],
                  prior_close: float = 0.0) -> Optional[dict]:
        """
        Run the full pattern matching pipeline on today's data so far.

        Args:
            today_bars: List of OHLCV dicts for today (from market open)
            prior_close: Yesterday's close for gap calculation

        Returns:
            Dict with matches, forecast, and metadata, or None if insufficient data
        """
        if self.num_days == 0 or self.normalized_templates is None:
            return None

        n_bars = len(today_bars)
        if n_bars < MIN_BARS_FOR_MATCH:
            return None

        # Cap at RTH bars
        n_bars = min(n_bars, RTH_BARS)

        # Extract today's OHLCV
        today_closes = np.array([b['close'] for b in today_bars[:n_bars]], dtype=np.float64)
        today_highs = np.array([b['high'] for b in today_bars[:n_bars]], dtype=np.float64)
        today_lows = np.array([b['low'] for b in today_bars[:n_bars]], dtype=np.float64)

        # Today's regime
        today_atr = compute_atr(today_highs[:min(30, n_bars)],
                                today_lows[:min(30, n_bars)],
                                today_closes[:min(30, n_bars)], period=10)
        today_gap = ((today_closes[0] - prior_close) / prior_close
                     if prior_close > 0 else 0.0)
        today_trend = classify_trend(today_closes)

        # Filter by regime
        filtered_idx = filter_by_regime(today_atr, today_gap, today_trend,
                                        self.day_metadata)
        if len(filtered_idx) == 0:
            return None

        # Normalize today
        today_normalized = normalize_intraday_prices(today_closes)

        # Extract same portion from historical templates
        hist_portion = self.normalized_templates[filtered_idx, :n_bars]

        # Compute correlations
        correlations = fast_correlation_batch(today_normalized, hist_portion)

        # Apply recency weighting
        days_ago = np.array([self.day_metadata[idx].day_index
                             for idx in filtered_idx], dtype=np.float64)
        weighted_corrs = apply_recency_weight(correlations, days_ago)

        # Rank by weighted correlation, take top N
        sorted_local = np.argsort(weighted_corrs)[::-1]
        top_local = sorted_local[:TOP_N_MATCHES]

        # Build match objects
        matches = []
        for local_idx in top_local:
            global_idx = filtered_idx[local_idx]
            corr = float(correlations[local_idx])
            w_corr = float(weighted_corrs[local_idx])

            if corr < MIN_CORRELATION:
                continue

            meta = self.day_metadata[global_idx]
            # Projection: remaining bars after the match window
            remaining_prices = self.historical_ohlcv[global_idx, n_bars:, 3].tolist()
            # Returns from the match-end price
            match_end_price = self.historical_ohlcv[global_idx, n_bars - 1, 3]
            if match_end_price > 0:
                remaining_returns = [
                    (p / match_end_price - 1.0) * 100.0 for p in remaining_prices
                ]
            else:
                remaining_returns = [0.0] * len(remaining_prices)

            matches.append(PatternMatch(
                date=meta.date,
                correlation=corr,
                weighted_correlation=w_corr,
                daily_return=meta.daily_return,
                projection_prices=remaining_prices,
                projection_returns=remaining_returns,
                metadata=asdict(meta)
            ))

        # Generate forecast
        forecast = generate_forecast(matches)

        result = {
            'timestamp': datetime.now(EASTERN).isoformat(),
            'bars_matched': n_bars,
            'filtered_day_count': len(filtered_idx),
            'total_days': self.num_days,
            'regime': {
                'atr': float(today_atr),
                'gap_pct': float(today_gap),
                'gap_type': classify_gap(today_gap).value,
                'trend': today_trend.value,
            },
            'matches': [
                {
                    'date': m.date,
                    'correlation': round(m.correlation, 4),
                    'weighted_correlation': round(m.weighted_correlation, 4),
                    'daily_return': round(m.daily_return, 2),
                    'projection_returns': [round(r, 4) for r in m.projection_returns],
                }
                for m in matches
            ],
            'forecast': {
                'direction_probability': round(forecast.direction_probability, 3),
                'direction_signal': round(forecast.direction_signal, 3),
                'mean_move': round(forecast.mean_move, 3),
                'median_move': round(forecast.median_move, 3),
                'std_dev': round(forecast.std_dev, 3),
                'confidence_interval_68': (
                    round(forecast.confidence_interval_68[0], 3),
                    round(forecast.confidence_interval_68[1], 3),
                ),
                'sample_size': forecast.sample_size,
                'consensus': forecast.consensus,
                'avg_correlation': round(forecast.avg_correlation, 4),
                'confluence_score': round(forecast.confluence_score, 4),
            } if forecast else None,
            'match_count': len(matches),
        }

        self.latest_result = result
        self._last_bar_count = n_bars
        return result
