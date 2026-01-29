"""
Daily Pattern Matching System with Intraday Refinement

Matches today's setup against 750 days of historical data using daily OHLCV
context + hourly intraday shape. Projects the typical rest-of-day path and
refines predictions every 30 minutes as the day unfolds.

Architecture:
  - Layer 1: Daily regime filter (ATR, gap, trend) narrows 750 → ~100-200 days
  - Layer 2: Hourly shape correlation ranks candidates by similarity
  - Forecast: Rest-of-day projection with inflection points and day phase
  - Updates every 30 min from 9:30 AM to 2:00 PM ET
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# RTH hourly slots: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 = 7 bars
RTH_HOURLY_SLOTS = 7
RTH_START_HOUR = 9   # 9:30
RTH_END_HOUR = 16    # 16:00

MIN_CORRELATION = 0.75       # Minimum for a valid match
MIN_STRONG_CORRELATION = 0.80
MIN_MATCHES_REQUIRED = 3
TOP_N_MATCHES = 8            # Return top 8 matches
RECENCY_DECAY = 0.97         # ~23-day half-life
ATR_TOLERANCE = 0.20         # ±20% ATR for regime filter
GAP_THRESHOLD = 0.003        # 0.3% for gap classification


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DayRecord:
    """One historical trading day with daily stats + hourly bars."""
    date_str: str
    day_index: int                    # 0 = oldest loaded
    # Daily OHLCV
    day_open: float
    day_high: float
    day_low: float
    day_close: float
    day_volume: int
    # Derived daily
    daily_range: float                # high - low
    daily_return_pct: float           # (close - open) / open * 100
    gap_pct: float                    # (open - prior_close) / prior_close
    gap_type: str                     # 'gap_up', 'gap_down', 'flat'
    atr_10: float                     # 10-period ATR at this day
    trend_context: str                # 'up', 'down', 'sideways'
    day_type: str                     # 'trend', 'reversal', 'range', 'breakout'
    is_extreme: bool                  # |daily_return| > 3%
    is_half_day: bool
    # Hourly bars (up to 7 slots for RTH)
    hourly_closes: np.ndarray         # shape (N,) where N <= 7
    hourly_opens: np.ndarray
    hourly_highs: np.ndarray
    hourly_lows: np.ndarray
    hourly_volumes: np.ndarray
    hourly_returns: np.ndarray        # % returns from day open
    # Key levels
    morning_high: float               # High of first 2 hours
    morning_low: float
    hod_bar: int                      # Bar index of high of day
    lod_bar: int                      # Bar index of low of day
    # Rest-of-day outcome (from hour 2 onward)
    rest_of_day_return: float         # % return from ~10:30 to close


# ═══════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_returns(prices: np.ndarray) -> np.ndarray:
    """Returns from open, then z-score normalize for shape comparison."""
    if len(prices) < 2:
        return np.zeros_like(prices)
    returns = (prices / prices[0] - 1.0) * 100.0
    std = np.std(returns)
    if std < 1e-8:
        return np.zeros_like(prices)
    return (returns - np.mean(returns)) / std


# ═══════════════════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════

def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two arrays. Returns 0 on error."""
    if len(a) < 2 or len(a) != len(b):
        return 0.0
    a_c = a - np.mean(a)
    b_c = b - np.mean(b)
    num = np.dot(a_c, b_c)
    den = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    if den < 1e-10:
        return 0.0
    r = num / den
    return float(np.clip(r, -1.0, 1.0))


def batch_pearson(today: np.ndarray, hist_matrix: np.ndarray) -> np.ndarray:
    """Vectorized Pearson: today (N,) vs hist_matrix (D, N) → (D,)."""
    if len(today) == 0 or hist_matrix.shape[0] == 0:
        return np.array([])
    t_c = today - np.mean(today)
    h_c = hist_matrix - np.mean(hist_matrix, axis=1, keepdims=True)
    t_norm = np.sqrt(np.sum(t_c ** 2))
    h_norms = np.sqrt(np.sum(h_c ** 2, axis=1))
    if t_norm < 1e-10:
        return np.zeros(hist_matrix.shape[0])
    return np.clip(np.dot(h_c, t_c) / (h_norms * t_norm + 1e-10), -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def classify_gap(gap_pct: float) -> str:
    if gap_pct > GAP_THRESHOLD:
        return 'gap_up'
    elif gap_pct < -GAP_THRESHOLD:
        return 'gap_down'
    return 'flat'


def classify_trend(closes: list, period: int = 9) -> str:
    """Trend context from an array of recent daily closes."""
    if len(closes) < period:
        return 'sideways'
    ema = float(np.mean(closes[-period:]))
    last = closes[-1]
    pct = (last - ema) / ema * 100.0
    if pct > 0.15:
        return 'up'
    elif pct < -0.15:
        return 'down'
    return 'sideways'


def classify_day_type(day_open, day_close, day_high, day_low, first_hour_close) -> str:
    """Classify day as trend/reversal/range/breakout."""
    daily_ret = abs(day_close - day_open) / day_open * 100 if day_open else 0
    intraday_range = (day_high - day_low) / day_open * 100 if day_open else 0
    first_hour_ret = (first_hour_close - day_open) / day_open * 100 if day_open and first_hour_close else 0
    daily_dir = day_close - day_open

    if daily_ret > 1.5 and intraday_range > 2.0:
        return 'trend'
    elif first_hour_ret != 0 and (daily_dir * first_hour_ret < 0):
        return 'reversal'
    elif intraday_range < 1.0:
        return 'range'
    return 'breakout'


def compute_atr_series(highs: list, lows: list, closes: list, period: int = 10) -> list:
    """Compute ATR for each bar given daily H/L/C lists."""
    n = len(highs)
    if n < 2:
        return [0.0] * n
    trs = [highs[0] - lows[0]]
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    # Simple moving average of TR
    atrs = []
    for i in range(n):
        start = max(0, i - period + 1)
        atrs.append(float(np.mean(trs[start:i + 1])))
    return atrs


# ═══════════════════════════════════════════════════════════════════════════
# FORECAST & PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_projection(matches: list, n_hours_so_far: int,
                        current_price: float) -> dict:
    """
    Build rest-of-day projection from matched historical days.
    matches: list of (DayRecord, correlation)
    n_hours_so_far: how many hourly bars we have for today (1-7)
    current_price: latest price
    """
    if not matches:
        return {}

    future_paths = []   # list of { returns: [...], weight: corr }
    for day, corr in matches:
        if len(day.hourly_closes) <= n_hours_so_far:
            continue
        remaining = day.hourly_closes[n_hours_so_far:]
        anchor = day.hourly_closes[n_hours_so_far - 1] if n_hours_so_far > 0 else day.day_open
        if anchor <= 0:
            continue
        returns = ((remaining / anchor) - 1.0) * 100.0
        future_paths.append({
            'returns': returns.tolist(),
            'weight': corr,
            'date': day.date_str,
            'day_type': day.day_type,
            'daily_return': day.daily_return_pct,
        })

    if not future_paths:
        return {}

    # Weighted average path
    max_len = max(len(p['returns']) for p in future_paths)
    weighted_sum = np.zeros(max_len)
    weight_sum = np.zeros(max_len)
    all_returns_at_end = []

    for p in future_paths:
        r = np.array(p['returns'])
        w = p['weight']
        weighted_sum[:len(r)] += r * w
        weight_sum[:len(r)] += w
        if len(r) > 0:
            all_returns_at_end.append(r[-1])

    avg_path = np.where(weight_sum > 0, weighted_sum / weight_sum, 0).tolist()

    # Projected prices
    projected_prices = [current_price * (1 + r / 100.0) for r in avg_path]

    # Inflection points (local max/min in avg path)
    inflections = []
    for i in range(1, len(avg_path) - 1):
        if avg_path[i] > avg_path[i - 1] and avg_path[i] > avg_path[i + 1]:
            inflections.append({
                'bar_offset': i,
                'type': 'typical_peak',
                'projected_price': round(projected_prices[i], 2),
            })
        elif avg_path[i] < avg_path[i - 1] and avg_path[i] < avg_path[i + 1]:
            inflections.append({
                'bar_offset': i,
                'type': 'typical_low',
                'projected_price': round(projected_prices[i], 2),
            })

    # Day phase
    hour_labels = ['9:30', '10:30', '11:30', '12:30', '13:30', '14:30', '15:30']
    current_phase = _identify_phase(n_hours_so_far, avg_path, inflections)

    # Confidence bands (16th and 84th percentile of all paths at each step)
    upper = []
    lower = []
    for step in range(max_len):
        vals = [p['returns'][step] for p in future_paths if len(p['returns']) > step]
        if vals:
            upper.append(current_price * (1 + float(np.percentile(vals, 84)) / 100))
            lower.append(current_price * (1 + float(np.percentile(vals, 16)) / 100))

    return {
        'hourly_levels': [
            {
                'bar_offset': i,
                'hour_label': hour_labels[n_hours_so_far + i] if (n_hours_so_far + i) < len(hour_labels) else f"+{i}h",
                'projected_price': round(p, 2),
                'avg_return_pct': round(avg_path[i], 3),
            }
            for i, p in enumerate(projected_prices)
        ],
        'inflection_points': inflections,
        'current_phase': current_phase,
        'end_of_day_price': round(projected_prices[-1], 2) if projected_prices else None,
        'end_of_day_return': round(avg_path[-1], 3) if avg_path else None,
        'confidence_upper': [round(v, 2) for v in upper],
        'confidence_lower': [round(v, 2) for v in lower],
        'individual_paths': [
            {
                'date': p['date'],
                'day_type': p['day_type'],
                'returns': [round(r, 4) for r in p['returns']],
            }
            for p in future_paths[:5]
        ],
    }


def _identify_phase(n_hours: int, avg_path: list, inflections: list) -> str:
    """Determine what phase of the day we're in."""
    # Time-based defaults
    if n_hours <= 1:
        return 'morning_open'
    elif n_hours <= 2:
        return 'morning_development'
    elif n_hours <= 4:
        if any(inf['type'] == 'typical_peak' and inf['bar_offset'] <= 2 for inf in inflections):
            return 'approaching_reversal_zone'
        return 'midday_assessment'
    elif n_hours <= 5:
        return 'afternoon_positioning'
    else:
        return 'late_day'

    # If there's a near inflection point, override
    for inf in inflections:
        if inf['bar_offset'] <= 1:
            if inf['type'] == 'typical_peak':
                return 'approaching_reversal_zone'
            elif inf['type'] == 'typical_low':
                return 'approaching_bounce_zone'

    return 'undefined'


def compute_confluence_score(matches: list) -> float:
    """Convert daily pattern matches into -1 to +1 score."""
    if len(matches) < MIN_MATCHES_REQUIRED:
        return 0.0
    outcomes = [day.rest_of_day_return for day, _ in matches]
    bullish = sum(1 for o in outcomes if o > 0)
    total = len(outcomes)
    dir_prob = bullish / total
    base = (dir_prob - 0.5) * 2.0
    avg_corr = float(np.mean([c for _, c in matches]))
    quality = max(0.0, (avg_corr - 0.75) / 0.25)
    consensus = abs(bullish - (total - bullish)) / total
    return float(base * quality * (0.5 + 0.5 * consensus))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class DailyPatternEngine:
    """
    Loads historical days, precomputes templates, runs the two-layer
    matching pipeline, and generates rest-of-day forecasts.
    """

    def __init__(self):
        self.days: list[DayRecord] = []
        self.num_days: int = 0
        # Precomputed normalized hourly matrices for batch correlation
        self.norm_hourly: Optional[np.ndarray] = None   # (num_days, 7)
        self.latest_result: Optional[dict] = None

    # ── Loading ──────────────────────────────────────────────────────────

    def load_from_1min_bars(self, bars_1min: list):
        """
        Build daily + hourly records from 1-minute bars.
        bar.time is ET-as-UTC unix seconds.
        Groups by date, aggregates into hourly slots, computes daily OHLCV.
        """
        from datetime import datetime as dt

        # Group bars by date
        days_map = {}  # date_str -> list of bars (sorted by time)
        for b in bars_1min:
            d = dt.utcfromtimestamp(b['time'])
            minutes = d.hour * 60 + d.minute
            # RTH filter: 9:30 (570) to 16:00 (960)
            if minutes < 570 or minutes >= 960:
                continue
            date_key = d.strftime('%Y-%m-%d')
            if date_key not in days_map:
                days_map[date_key] = []
            days_map[date_key].append(b)

        sorted_dates = sorted(days_map.keys())
        if not sorted_dates:
            logger.warning("No RTH bars found for pattern matcher")
            return

        # Build daily records
        all_daily_closes = []
        all_daily_highs = []
        all_daily_lows = []
        records = []

        for di, date_key in enumerate(sorted_dates):
            day_bars = sorted(days_map[date_key], key=lambda x: x['time'])
            if len(day_bars) < 30:  # Need at least ~30 min of data
                continue

            day_open = day_bars[0]['open']
            day_close = day_bars[-1]['close']
            day_high = max(b['high'] for b in day_bars)
            day_low = min(b['low'] for b in day_bars)
            day_vol = sum(b['volume'] for b in day_bars)

            # Aggregate to hourly slots (9:30-10:29, 10:30-11:29, etc.)
            hourly_slots = {}  # hour_key -> list of bars
            for b in day_bars:
                d = dt.utcfromtimestamp(b['time'])
                # Map to slot: 9:30-10:29 → 9, 10:30-11:29 → 10, ...
                slot_hour = d.hour if d.minute >= 30 else d.hour - 1
                if slot_hour < 9:
                    slot_hour = 9
                if slot_hour not in hourly_slots:
                    hourly_slots[slot_hour] = []
                hourly_slots[slot_hour].append(b)

            slot_keys = sorted(hourly_slots.keys())
            h_opens, h_highs, h_lows, h_closes, h_vols = [], [], [], [], []
            for sk in slot_keys:
                slot_bars = hourly_slots[sk]
                h_opens.append(slot_bars[0]['open'])
                h_closes.append(slot_bars[-1]['close'])
                h_highs.append(max(b['high'] for b in slot_bars))
                h_lows.append(min(b['low'] for b in slot_bars))
                h_vols.append(sum(b['volume'] for b in slot_bars))

            if not h_closes:
                continue

            h_closes_arr = np.array(h_closes, dtype=np.float64)
            h_returns = ((h_closes_arr / day_open) - 1.0) * 100.0 if day_open else np.zeros(len(h_closes))

            # Prior close for gap calc
            prior_close = all_daily_closes[-1] if all_daily_closes else day_open
            gap_pct = (day_open - prior_close) / prior_close if prior_close else 0.0

            # ATR
            all_daily_highs.append(day_high)
            all_daily_lows.append(day_low)
            all_daily_closes.append(day_close)
            if len(all_daily_closes) >= 2:
                atrs = compute_atr_series(all_daily_highs, all_daily_lows, all_daily_closes, 10)
                atr_val = atrs[-1]
            else:
                atr_val = day_high - day_low

            # Trend context from last 9 daily closes
            trend = classify_trend(all_daily_closes, 9)

            # Day type
            first_hour_close = h_closes[0] if h_closes else day_open
            dtype = classify_day_type(day_open, day_close, day_high, day_low, first_hour_close)

            # Morning levels
            morning_bars = min(2, len(h_highs))
            morn_high = max(h_highs[:morning_bars]) if morning_bars > 0 else day_high
            morn_low = min(h_lows[:morning_bars]) if morning_bars > 0 else day_low

            # HOD/LOD bar indices
            hod_bar = int(np.argmax(h_highs))
            lod_bar = int(np.argmin(h_lows))

            # Rest-of-day return (from hour 2 close to day close)
            rod = 0.0
            if len(h_closes) >= 2:
                rod = (day_close - h_closes[1]) / h_closes[1] * 100.0 if h_closes[1] else 0.0

            daily_ret = (day_close - day_open) / day_open * 100.0 if day_open else 0.0

            rec = DayRecord(
                date_str=date_key,
                day_index=di,
                day_open=day_open, day_high=day_high, day_low=day_low,
                day_close=day_close, day_volume=day_vol,
                daily_range=day_high - day_low,
                daily_return_pct=daily_ret,
                gap_pct=gap_pct, gap_type=classify_gap(gap_pct),
                atr_10=atr_val, trend_context=trend,
                day_type=dtype,
                is_extreme=abs(daily_ret) > 3.0,
                is_half_day=len(day_bars) < 200,
                hourly_closes=h_closes_arr,
                hourly_opens=np.array(h_opens, dtype=np.float64),
                hourly_highs=np.array(h_highs, dtype=np.float64),
                hourly_lows=np.array(h_lows, dtype=np.float64),
                hourly_volumes=np.array(h_vols, dtype=np.float64),
                hourly_returns=h_returns,
                morning_high=morn_high, morning_low=morn_low,
                hod_bar=hod_bar, lod_bar=lod_bar,
                rest_of_day_return=rod,
            )
            records.append(rec)

        self.days = records
        self.num_days = len(records)

        # Precompute normalized hourly templates (pad to 7 slots)
        if self.num_days > 0:
            self.norm_hourly = np.zeros((self.num_days, RTH_HOURLY_SLOTS), dtype=np.float64)
            for i, day in enumerate(self.days):
                n = min(len(day.hourly_closes), RTH_HOURLY_SLOTS)
                if n >= 2:
                    normed = normalize_returns(day.hourly_closes[:n])
                    self.norm_hourly[i, :n] = normed

        logger.info(f"Loaded {self.num_days} daily records for pattern matching")

    # ── Matching Pipeline ────────────────────────────────────────────────

    def run_match_from_1min(self, bars_1min: list) -> Optional[dict]:
        """
        Full pipeline: extract today's data from 1-min bars, run match.
        Called by the background scheduler.
        """
        from datetime import datetime as dt

        if self.num_days < 20:
            return None

        # Find the most recent trading day with RTH bars.
        # Timestamps are ET-as-UTC, so utcfromtimestamp gives ET values.
        if not bars_1min:
            return None

        # Collect all RTH bars grouped by date, find the latest date with enough bars
        from collections import defaultdict
        rth_by_date = defaultdict(list)
        for b in bars_1min:
            d = dt.utcfromtimestamp(b['time'])
            minutes = d.hour * 60 + d.minute
            if 570 <= minutes < 960:
                rth_by_date[d.strftime('%Y-%m-%d')].append(b)

        # Find most recent date with >= 15 RTH bars
        today_str = None
        today_bars = []
        for date_str in sorted(rth_by_date.keys(), reverse=True):
            if len(rth_by_date[date_str]) >= 15:
                today_str = date_str
                today_bars = sorted(rth_by_date[date_str], key=lambda x: x['time'])
                break

        if not today_bars:
            logger.warning("Pattern matcher: no trading day found with >= 15 RTH bars")
            return None

        # Aggregate today's bars into hourly slots
        hourly_slots = {}
        for b in today_bars:
            d = dt.utcfromtimestamp(b['time'])
            slot_hour = d.hour if d.minute >= 30 else d.hour - 1
            if slot_hour < 9:
                slot_hour = 9
            if slot_hour not in hourly_slots:
                hourly_slots[slot_hour] = []
            hourly_slots[slot_hour].append(b)

        slot_keys = sorted(hourly_slots.keys())
        today_hourly_closes = []
        for sk in slot_keys:
            sbs = hourly_slots[sk]
            today_hourly_closes.append(sbs[-1]['close'])

        if len(today_hourly_closes) < 1:
            return None

        today_hourly = np.array(today_hourly_closes, dtype=np.float64)
        n_hours = len(today_hourly)
        current_price = today_hourly[-1]

        # Yesterday's close for gap calc
        yesterday = None
        for day in reversed(self.days):
            if day.date_str < today_str:
                yesterday = day
                break
        if not yesterday:
            return None

        today_open = today_bars[0]['open']
        today_gap = (today_open - yesterday.day_close) / yesterday.day_close if yesterday.day_close else 0.0
        today_atr = yesterday.atr_10  # Use yesterday's ATR as proxy
        today_trend = yesterday.trend_context

        # ── Layer 1: Regime filter ──
        filtered = self._filter_by_regime(today_gap, today_atr, today_trend)
        if not filtered:
            return None

        # ── Layer 2: Hourly similarity ──
        today_norm = normalize_returns(today_hourly)
        # Build matrix of same-length portions from filtered days
        filtered_indices = []
        hist_portions = []
        for idx in filtered:
            day = self.days[idx]
            if len(day.hourly_closes) >= n_hours:
                portion = normalize_returns(day.hourly_closes[:n_hours])
                hist_portions.append(portion)
                filtered_indices.append(idx)

        if len(hist_portions) < MIN_MATCHES_REQUIRED:
            return None

        hist_matrix = np.array(hist_portions, dtype=np.float64)
        correlations = batch_pearson(today_norm, hist_matrix)

        # Apply recency weighting (more recent days get higher weight)
        days_ago = np.array([
            self.num_days - self.days[idx].day_index
            for idx in filtered_indices
        ], dtype=np.float64)
        weighted_corrs = correlations * (RECENCY_DECAY ** days_ago)

        # Rank by weighted correlation, take top N
        sorted_local = np.argsort(weighted_corrs)[::-1]
        top_matches = []
        for li in sorted_local[:TOP_N_MATCHES]:
            corr = float(correlations[li])
            if corr < MIN_CORRELATION:
                continue
            day = self.days[filtered_indices[li]]
            if day.is_half_day or day.is_extreme:
                continue
            top_matches.append((day, corr))

        if len(top_matches) < MIN_MATCHES_REQUIRED:
            return None

        # ── Forecast ──
        projection = generate_projection(top_matches, n_hours, current_price)
        conf_score = compute_confluence_score(top_matches)

        # Direction stats
        outcomes = [d.rest_of_day_return for d, _ in top_matches]
        bullish = sum(1 for o in outcomes if o > 0)

        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'bars_matched': n_hours,
            'filtered_day_count': len(filtered),
            'total_days': self.num_days,
            'regime': {
                'gap_pct': round(today_gap * 100, 3),
                'gap_type': classify_gap(today_gap),
                'atr': round(today_atr, 2),
                'trend': today_trend,
            },
            'matches': [
                {
                    'date': d.date_str,
                    'correlation': round(c, 4),
                    'day_type': d.day_type,
                    'daily_return': round(d.daily_return_pct, 2),
                    'rest_of_day_return': round(d.rest_of_day_return, 3),
                }
                for d, c in top_matches
            ],
            'forecast': {
                'direction_probability': round(bullish / len(top_matches), 3),
                'direction_signal': round((bullish / len(top_matches) - 0.5) * 2.0, 3),
                'mean_move': round(float(np.mean(outcomes)), 3),
                'median_move': round(float(np.median(outcomes)), 3),
                'std_dev': round(float(np.std(outcomes)), 3),
                'sample_size': len(top_matches),
                'consensus': f"{bullish}/{len(top_matches)} bullish",
                'avg_correlation': round(float(np.mean([c for _, c in top_matches])), 4),
                'confluence_score': round(conf_score, 4),
            },
            'projection': projection,
            'match_count': len(top_matches),
        }

        self.latest_result = result
        return result

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _filter_by_regime(self, today_gap: float, today_atr: float,
                          today_trend: str) -> list[int]:
        """Progressive regime filtering: strict → relaxed → fallback."""
        today_gap_type = classify_gap(today_gap)

        # Strict: ATR + gap + trend
        strict = []
        for i, day in enumerate(self.days):
            if day.is_half_day or day.is_extreme:
                continue
            atr_ok = abs(day.atr_10 - today_atr) / (today_atr + 1e-8) <= ATR_TOLERANCE
            gap_ok = day.gap_type == today_gap_type
            trend_ok = day.trend_context == today_trend
            if atr_ok and gap_ok and trend_ok:
                strict.append(i)
        if len(strict) >= 20:
            return strict

        # Relaxed: ATR + gap
        relaxed = []
        for i, day in enumerate(self.days):
            if day.is_half_day or day.is_extreme:
                continue
            atr_ok = abs(day.atr_10 - today_atr) / (today_atr + 1e-8) <= ATR_TOLERANCE
            gap_ok = day.gap_type == today_gap_type
            if atr_ok and gap_ok:
                relaxed.append(i)
        if len(relaxed) >= 10:
            return relaxed

        # ATR only
        atr_only = []
        for i, day in enumerate(self.days):
            if day.is_half_day or day.is_extreme:
                continue
            if abs(day.atr_10 - today_atr) / (today_atr + 1e-8) <= 0.30:
                atr_only.append(i)
        if len(atr_only) >= 10:
            return atr_only

        # All valid days
        return [i for i, d in enumerate(self.days)
                if not d.is_half_day and not d.is_extreme]
