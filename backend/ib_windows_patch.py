"""
Windows-specific patches for ib_insync to fix event loop issues.

This module MUST be imported before any ib_insync imports.

The issue: ib_insync's util.getLoop() calls get_event_loop() which may return
a different loop than the one uvicorn is running in, causing "Future attached
to a different loop" errors.

The fix: Patch getLoop() to always return the currently running loop.
"""

import sys
import asyncio

if sys.platform == 'win32':
    # Patch ib_insync.util.getLoop() to use the current running loop
    import ib_insync.util as util_module

    _original_getLoop = util_module.getLoop

    def patched_getLoop():
        """Get the asyncio event loop, preferring the currently running one"""
        try:
            # Try to get the running loop first (works inside async functions)
            return asyncio.get_running_loop()
        except RuntimeError:
            # Fall back to original behavior
            return _original_getLoop()

    util_module.getLoop = patched_getLoop

    print("✓ Applied Windows-specific ib_insync patches")

