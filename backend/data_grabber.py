"""
Data Grabber - Backward historical data fetcher with terminal confirmation.

Fetches 60 days of 1-min data per batch, going backwards up to 500 days.
Asks y/n in terminal between batches, with 3-minute pauses.
"""

import asyncio
import logging
import sys
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Callable
import pytz

from .historical_data import HistoricalDataFetcher
from .cache import DataCache
from .contracts import create_continuous_contract

logger = logging.getLogger(__name__)

# Global state for data grabber
_grabber_state: Dict = {
    'active': False,
    'symbol': None,
    'batch_number': 0,
    'total_batches': 0,
    'days_fetched': 0,
    'target_days': 500,
    'status': 'idle',  # idle, fetching, waiting_confirm, pausing, done, error, cancelled
    'message': '',
    'days_in_cache': {},  # per-symbol day counts
}

_grabber_task: Optional[asyncio.Task] = None
_confirm_event: Optional[asyncio.Event] = None
_confirm_answer: bool = False


def get_grabber_status() -> dict:
    """Return current grabber state for frontend polling."""
    return {
        'active': _grabber_state['active'],
        'symbol': _grabber_state['symbol'],
        'batch_number': _grabber_state['batch_number'],
        'total_batches': _grabber_state['total_batches'],
        'days_fetched': _grabber_state['days_fetched'],
        'target_days': _grabber_state['target_days'],
        'status': _grabber_state['status'],
        'message': _grabber_state['message'],
        'days_in_cache': _grabber_state['days_in_cache'],
    }


def _count_days_in_cache(cache: DataCache, symbol: str) -> int:
    """Count unique trading days in cached 1-min data."""
    data = cache.load(symbol, bar_size='1min', max_age_hours=None)
    if data is None or len(data) == 0:
        return 0
    # Convert timestamps to dates and count unique
    import pandas as pd
    dates = pd.to_datetime(data['time'], unit='s').dt.date
    return dates.nunique()


def update_all_day_counts(cache: DataCache):
    """Update day counts for all symbols."""
    for sym in ['MNQ', 'MES', 'MGC']:
        try:
            _grabber_state['days_in_cache'][sym] = _count_days_in_cache(cache, sym)
        except Exception:
            _grabber_state['days_in_cache'][sym] = 0


def _terminal_confirm_thread(prompt: str, loop: asyncio.AbstractEventLoop):
    """Run in a thread to get terminal y/n without blocking the event loop."""
    global _confirm_answer
    try:
        answer = input(prompt)
        _confirm_answer = answer.strip().lower() in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        _confirm_answer = False
    # Signal the async event from the thread
    loop.call_soon_threadsafe(_confirm_event.set)


async def _ask_terminal_confirm(prompt: str) -> bool:
    """Ask y/n in terminal, non-blocking for the event loop."""
    global _confirm_event, _confirm_answer
    _confirm_event = asyncio.Event()
    _confirm_answer = False

    loop = asyncio.get_running_loop()
    thread = threading.Thread(
        target=_terminal_confirm_thread,
        args=(prompt, loop),
        daemon=True
    )
    thread.start()
    await _confirm_event.wait()
    return _confirm_answer


async def start_grab(
    symbol: str,
    ib,
    cache: DataCache,
    on_complete: Optional[Callable] = None
):
    """
    Start grabbing historical data for a symbol.
    Fetches 60 days per batch going backwards, up to 500 days total.
    """
    global _grabber_task

    if _grabber_state['active']:
        logger.warning(f"Data grabber already active for {_grabber_state['symbol']}")
        return False

    _grabber_task = asyncio.create_task(_grab_loop(symbol, ib, cache, on_complete))
    return True


async def stop_grab():
    """Cancel the active grab."""
    global _grabber_task
    if _grabber_task and not _grabber_task.done():
        _grabber_task.cancel()
        try:
            await _grabber_task
        except asyncio.CancelledError:
            pass
    _grabber_state['active'] = False
    _grabber_state['status'] = 'cancelled'
    _grabber_state['message'] = 'Cancelled by user'
    _grabber_task = None


async def _grab_loop(
    symbol: str,
    ib,
    cache: DataCache,
    on_complete: Optional[Callable] = None
):
    """Main grab loop: fetch 60-day batches backwards with terminal y/n."""
    from .pacing import PacingManager

    _grabber_state['active'] = True
    _grabber_state['symbol'] = symbol
    _grabber_state['days_fetched'] = 0
    _grabber_state['status'] = 'fetching'
    _grabber_state['message'] = f'Starting data grab for {symbol}...'

    target_days = _grabber_state['target_days']  # 500
    batch_size = 60  # days per batch
    total_batches = (target_days + batch_size - 1) // batch_size
    _grabber_state['total_batches'] = total_batches
    _grabber_state['batch_number'] = 0

    try:
        # Use continuous contract to fetch across multiple expirations
        contract = create_continuous_contract(symbol)
        await ib.qualifyContractsAsync(contract)
        logger.info(f"[DataGrab] Using continuous contract: {contract}")

        # Find earliest date in existing cache
        existing = cache.load(symbol, bar_size='1min', max_age_hours=None)
        if existing is not None and len(existing) > 0:
            earliest_ts = existing['time'].min()
            current_end = datetime.fromtimestamp(earliest_ts, tz=pytz.UTC)
            logger.info(f"[DataGrab] Existing cache earliest: {current_end.strftime('%Y-%m-%d')}")
        else:
            current_end = datetime.now(pytz.UTC)
            logger.info(f"[DataGrab] No existing cache, starting from now")

        fetcher = HistoricalDataFetcher(ib=ib, cache=cache)
        days_total = 0

        for batch_num in range(1, total_batches + 1):
            _grabber_state['batch_number'] = batch_num

            # Ask terminal confirmation
            _grabber_state['status'] = 'waiting_confirm'
            prompt_msg = (
                f"\n[DataGrab] Batch {batch_num}/{total_batches} for {symbol} "
                f"({days_total} days fetched so far). "
                f"Fetch next {batch_size} days ending at "
                f"{current_end.strftime('%Y-%m-%d')}? (y/n): "
            )
            _grabber_state['message'] = f'Waiting for terminal confirmation (batch {batch_num})...'
            print(prompt_msg, end='', flush=True)

            confirmed = await _ask_terminal_confirm("")
            if not confirmed:
                _grabber_state['status'] = 'done'
                _grabber_state['message'] = f'Stopped by user after {days_total} days'
                _grabber_state['active'] = False
                logger.info(f"[DataGrab] User declined batch {batch_num}, stopping")
                break

            # Fetch this batch
            _grabber_state['status'] = 'fetching'
            _grabber_state['message'] = f'Fetching batch {batch_num}/{total_batches} ({batch_size} days)...'
            logger.info(f"[DataGrab] Fetching batch {batch_num}: {batch_size} days ending {current_end.strftime('%Y-%m-%d')}")

            start_date = current_end - timedelta(days=batch_size)

            new_data = await fetcher._fetch_year_chunked(
                contract,
                end_date=current_end,
                start_date=start_date
            )

            if new_data is not None and len(new_data) > 0:
                # Merge with existing cache
                existing = cache.load(symbol, bar_size='1min', max_age_hours=None)
                if existing is not None and len(existing) > 0:
                    import pandas as pd
                    combined = pd.concat([new_data, existing], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['time'], keep='last')
                    combined = combined.sort_values('time').reset_index(drop=True)
                else:
                    combined = new_data

                cache.save(symbol, combined, bar_size='1min')

                # Also update aggregated timeframes
                fetcher._cache_all_timeframes(symbol, combined)

                days_total += batch_size
                _grabber_state['days_fetched'] = days_total

                # Update day counts
                update_all_day_counts(cache)

                logger.info(
                    f"[DataGrab] Batch {batch_num} done: {len(new_data)} bars fetched, "
                    f"{len(combined)} total bars in cache"
                )
                print(
                    f"[DataGrab] Batch {batch_num} complete: {len(new_data)} new bars, "
                    f"{len(combined)} total in cache ({days_total} days fetched)"
                )
            else:
                logger.warning(f"[DataGrab] Batch {batch_num}: no data returned")
                print(f"[DataGrab] Batch {batch_num}: no data returned, moving further back")

            # Move end date backwards
            current_end = start_date

            # Check if we've hit target
            if days_total >= target_days:
                _grabber_state['status'] = 'done'
                _grabber_state['message'] = f'Complete! {days_total} days fetched for {symbol}'
                _grabber_state['active'] = False
                logger.info(f"[DataGrab] Target reached: {days_total} days")
                print(f"\n[DataGrab] Target reached! {days_total} days fetched for {symbol}")
                break

            # 3-minute pause between batches
            if batch_num < total_batches:
                _grabber_state['status'] = 'pausing'
                _grabber_state['message'] = f'Pausing 3 minutes before next batch...'
                print(f"[DataGrab] Pausing 3 minutes before next batch...")
                await asyncio.sleep(180)

        # Done
        if _grabber_state['status'] != 'done':
            _grabber_state['status'] = 'done'
            _grabber_state['message'] = f'Finished: {days_total} days fetched for {symbol}'
        _grabber_state['active'] = False

        # Callback
        if on_complete:
            await on_complete(symbol)

    except asyncio.CancelledError:
        _grabber_state['active'] = False
        _grabber_state['status'] = 'cancelled'
        _grabber_state['message'] = 'Grab cancelled'
        raise
    except Exception as e:
        logger.error(f"[DataGrab] Error: {e}", exc_info=True)
        _grabber_state['active'] = False
        _grabber_state['status'] = 'error'
        _grabber_state['message'] = f'Error: {str(e)}'
