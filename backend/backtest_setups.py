"""
Backtest the 23 Setup Detectors on Historical Data

Usage:
    python -m backend.backtest_setups

This loads historical 1-min bars and runs all setup detectors,
tracking signal outcomes (TARGET_HIT, STOP_HIT, TIMEOUT).
"""

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.setup_detectors import (
    SetupManager, SetupSignal, BarInput, IndicatorState
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A single backtest trade."""
    signal: SetupSignal
    entry_bar_idx: int
    exit_bar_idx: Optional[int] = None
    exit_price: Optional[float] = None
    outcome: str = "PENDING"  # TARGET_HIT, STOP_HIT, TIMEOUT_PROFIT, TIMEOUT_LOSS, TIMEOUT_SCRATCH
    r_multiple: float = 0.0
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion


@dataclass
class SetupStats:
    """Stats for a single setup."""
    name: str
    display_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    total_r: float = 0.0
    win_rate: float = 0.0
    avg_r: float = 0.0
    avg_mfe: float = 0.0
    avg_mae: float = 0.0


def resolve_trade(trade: BacktestTrade, bars: List[BarInput], start_idx: int) -> BacktestTrade:
    """
    Resolve a trade by walking forward through bars until stop, target, or timeout.
    """
    sig = trade.signal
    direction = sig.direction
    entry = sig.entry_price
    stop = sig.stop_price
    target = sig.target_price
    max_bars = sig.max_bars

    # Calculate initial risk
    initial_risk = abs(entry - stop)
    if initial_risk <= 0:
        initial_risk = 0.0001  # Avoid division by zero

    mfe = 0.0  # Max favorable excursion in R
    mae = 0.0  # Max adverse excursion in R

    for i in range(start_idx + 1, min(start_idx + max_bars + 1, len(bars))):
        bar = bars[i]
        bars_held = i - start_idx

        if direction == "LONG":
            # Track MFE/MAE
            favorable = (bar.high - entry) / initial_risk
            adverse = (entry - bar.low) / initial_risk
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            # Check stop hit (check stop first - pessimistic)
            if bar.low <= stop:
                trade.exit_bar_idx = i
                trade.exit_price = stop
                trade.outcome = "STOP_HIT"
                trade.r_multiple = -1.0
                trade.mfe = mfe
                trade.mae = mae
                return trade

            # Check target hit
            if bar.high >= target:
                trade.exit_bar_idx = i
                trade.exit_price = target
                trade.outcome = "TARGET_HIT"
                trade.r_multiple = (target - entry) / initial_risk
                trade.mfe = mfe
                trade.mae = mae
                return trade

        else:  # SHORT
            # Track MFE/MAE
            favorable = (entry - bar.low) / initial_risk
            adverse = (bar.high - entry) / initial_risk
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            # Check stop hit
            if bar.high >= stop:
                trade.exit_bar_idx = i
                trade.exit_price = stop
                trade.outcome = "STOP_HIT"
                trade.r_multiple = -1.0
                trade.mfe = mfe
                trade.mae = mae
                return trade

            # Check target hit
            if bar.low <= target:
                trade.exit_bar_idx = i
                trade.exit_price = target
                trade.outcome = "TARGET_HIT"
                trade.r_multiple = (entry - target) / initial_risk
                trade.mfe = mfe
                trade.mae = mae
                return trade

    # Timeout - calculate final P&L
    final_bar = bars[min(start_idx + max_bars, len(bars) - 1)]
    if direction == "LONG":
        pnl = final_bar.close - entry
    else:
        pnl = entry - final_bar.close

    r_mult = pnl / initial_risk
    trade.exit_bar_idx = min(start_idx + max_bars, len(bars) - 1)
    trade.exit_price = final_bar.close
    trade.r_multiple = r_mult
    trade.mfe = mfe
    trade.mae = mae

    if r_mult > 0.25:
        trade.outcome = "TIMEOUT_PROFIT"
    elif r_mult < -0.25:
        trade.outcome = "TIMEOUT_LOSS"
    else:
        trade.outcome = "TIMEOUT_SCRATCH"

    return trade


def run_backtest(bars: List[dict], verbose: bool = False) -> Dict[str, SetupStats]:
    """
    Run backtest on historical bars.

    Args:
        bars: List of bar dicts with time, open, high, low, close, volume
        verbose: Print each trade

    Returns:
        Dict of setup_name -> SetupStats
    """
    logger.info(f"Starting backtest on {len(bars)} bars...")

    # Initialize setup manager
    manager = SetupManager()
    manager.register_all_defaults()
    logger.info(f"Registered {len(manager.detectors)} detectors")

    # Convert bars to BarInput
    bar_inputs: List[BarInput] = []
    for b in bars:
        bar_inputs.append(BarInput(
            time=b.get('time', 0),
            open=b.get('open', 0),
            high=b.get('high', 0),
            low=b.get('low', 0),
            close=b.get('close', 0),
            volume=b.get('volume', 0),
        ))

    # Track all trades
    all_trades: List[BacktestTrade] = []
    stats_by_setup: Dict[str, SetupStats] = {}

    # Initialize stats for each detector
    for d in manager.detectors:
        stats_by_setup[d.name] = SetupStats(
            name=d.name,
            display_name=d.display_name
        )

    # Run through each bar
    logger.info("Processing bars...")
    signals_emitted = 0

    for i, bar_dict in enumerate(bars):
        if i % 10000 == 0:
            logger.info(f"  Processed {i}/{len(bars)} bars, {signals_emitted} signals so far...")

        # Process bar through setup manager
        signals = manager.process_bar(bar_dict)

        for sig in signals:
            signals_emitted += 1
            trade = BacktestTrade(
                signal=sig,
                entry_bar_idx=i
            )

            # Resolve the trade
            trade = resolve_trade(trade, bar_inputs, i)
            all_trades.append(trade)

            # Update stats
            stats = stats_by_setup[sig.setup_name]
            stats.total_trades += 1
            stats.total_r += trade.r_multiple
            stats.avg_mfe = (stats.avg_mfe * (stats.total_trades - 1) + trade.mfe) / stats.total_trades
            stats.avg_mae = (stats.avg_mae * (stats.total_trades - 1) + trade.mae) / stats.total_trades

            if trade.outcome == "TARGET_HIT":
                stats.wins += 1
            elif trade.outcome == "STOP_HIT":
                stats.losses += 1
            elif trade.outcome in ("TIMEOUT_PROFIT", "TIMEOUT_LOSS", "TIMEOUT_SCRATCH"):
                stats.timeouts += 1
                if trade.outcome == "TIMEOUT_PROFIT":
                    stats.wins += 1
                elif trade.outcome == "TIMEOUT_LOSS":
                    stats.losses += 1

            if verbose:
                dt = datetime.fromtimestamp(sig.bar_time, tz=timezone.utc)
                logger.info(
                    f"  {sig.setup_name}: {sig.direction} @ {sig.entry_price:.2f} "
                    f"-> {trade.outcome} ({trade.r_multiple:+.2f}R) at bar {trade.exit_bar_idx}"
                )

    # Calculate final stats
    for name, stats in stats_by_setup.items():
        if stats.total_trades > 0:
            stats.win_rate = (stats.wins / stats.total_trades) * 100
            stats.avg_r = stats.total_r / stats.total_trades

    logger.info(f"Backtest complete: {signals_emitted} signals across {len(bars)} bars")

    return stats_by_setup


def print_results(stats: Dict[str, SetupStats]):
    """Print backtest results in a nice table."""
    print("\n" + "=" * 100)
    print("BACKTEST RESULTS - 23 Setup Detectors")
    print("=" * 100)
    print(f"{'Setup':<30} {'Trades':>8} {'Win%':>8} {'Wins':>6} {'Losses':>8} {'Timeouts':>10} {'Avg R':>8} {'Total R':>10}")
    print("-" * 100)

    # Sort by total trades descending
    sorted_stats = sorted(stats.values(), key=lambda x: x.total_trades, reverse=True)

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_r = 0.0

    for s in sorted_stats:
        if s.total_trades > 0:
            print(
                f"{s.display_name:<30} {s.total_trades:>8} {s.win_rate:>7.1f}% "
                f"{s.wins:>6} {s.losses:>8} {s.timeouts:>10} "
                f"{s.avg_r:>+7.2f}R {s.total_r:>+9.1f}R"
            )
            total_trades += s.total_trades
            total_wins += s.wins
            total_losses += s.losses
            total_r += s.total_r

    print("-" * 100)
    overall_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_avg_r = total_r / total_trades if total_trades > 0 else 0
    print(f"{'TOTAL':<30} {total_trades:>8} {overall_wr:>7.1f}% {total_wins:>6} {total_losses:>8} {'':>10} {overall_avg_r:>+7.2f}R {total_r:>+9.1f}R")
    print("=" * 100)

    # Print setups with no trades
    no_trades = [s for s in sorted_stats if s.total_trades == 0]
    if no_trades:
        print(f"\nSetups with 0 trades: {', '.join(s.display_name for s in no_trades)}")


async def load_historical_bars() -> List[dict]:
    """Load historical bars from the system's cache or IB."""
    # Try to import and use the app's data loading
    try:
        from backend.cache import DataCache
        from backend.contracts import get_current_contract

        cache = DataCache()

        # Try loading from cache first
        bars = cache.load('MNQ', '1min')
        if bars is not None and len(bars) > 0:
            # Convert DataFrame to list of dicts
            bar_list = []
            for _, row in bars.iterrows():
                bar_list.append({
                    'time': int(row['date'].timestamp()) if hasattr(row['date'], 'timestamp') else int(row['date']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row.get('volume', 0)),
                })
            logger.info(f"Loaded {len(bar_list)} bars from cache")
            return bar_list
    except Exception as e:
        logger.warning(f"Could not load from cache: {e}")

    # If no cache, we need IB connection
    logger.error("No cached data found. Run the main app first to fetch historical data.")
    return []


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  SETUP DETECTOR BACKTESTER")
    print("=" * 60 + "\n")

    # Load historical data
    bars = await load_historical_bars()

    if not bars:
        print("ERROR: No historical data available.")
        print("Run the main application first to fetch and cache historical data,")
        print("or provide a data file.")
        return

    print(f"Loaded {len(bars)} bars")

    # Get date range
    if bars:
        start_dt = datetime.fromtimestamp(bars[0]['time'], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(bars[-1]['time'], tz=timezone.utc)
        print(f"Date range: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")

    # Run backtest
    stats = run_backtest(bars, verbose=False)

    # Print results
    print_results(stats)


if __name__ == "__main__":
    asyncio.run(main())
