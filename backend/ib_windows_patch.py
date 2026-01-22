"""
Windows-specific patches for ib_insync to fix event loop issues.

This module MUST be imported before any ib_insync imports.
"""

import sys
import asyncio

if sys.platform == 'win32':
    import ib_insync.util as util_module

    # Store original functions
    _original_run = util_module.run
    _original_schedule = util_module.schedule

    def patched_run(*awaitables, timeout=None):
        """Patched run() that uses current event loop"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if not awaitables:
            return []

        # Run the awaitables
        results = []
        for awaitable in awaitables:
            try:
                if asyncio.iscoroutine(awaitable):
                    result = loop.run_until_complete(asyncio.wait_for(awaitable, timeout))
                else:
                    result = awaitable
                results.append(result)
            except Exception as e:
                results.append(e)

        return results[0] if len(results) == 1 else results

    def patched_schedule(time, callback, *args):
        """Patched schedule() that uses current event loop"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        return loop.call_later(time, callback, *args)

    # Apply patches
    util_module.run = patched_run
    util_module.schedule = patched_schedule

    print("✓ Applied Windows-specific ib_insync patches")
