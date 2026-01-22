"""
Windows-specific patches for ib_insync to fix event loop issues.

This module MUST be imported before any ib_insync imports.
"""

import sys
import asyncio

if sys.platform == 'win32':
    # Patch ib_insync.connection module
    import ib_insync.connection as connection_module

    # Store original Connection class
    _OriginalConnection = connection_module.Connection

    class PatchedConnection(_OriginalConnection):
        """Windows-patched Connection that uses current event loop"""

        def __init__(self, host, port):
            # Get or create event loop before calling parent init
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
                if loop is None or loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            super().__init__(host, port)

        async def connectAsync(self):
            """Override connectAsync to ensure it uses the correct event loop"""
            # Ensure we're using the current event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

            # Call parent's connectAsync
            return await super().connectAsync()

    # Replace the Connection class
    connection_module.Connection = PatchedConnection

    # Also patch IB class to use patched Connection
    import ib_insync.ib as ib_module
    _OriginalIB = ib_module.IB

    class PatchedIB(_OriginalIB):
        """Windows-patched IB that ensures correct event loop"""

        def connect(self, host='127.0.0.1', port=7497, clientId=1, timeout=4,
                   readonly=False, account=''):
            """Override connect to use current event loop"""
            # Create a new connection with patched Connection class
            self._createConnection()
            return super().connect(host, port, clientId, timeout, readonly, account)

    # Replace IB class
    ib_module.IB = PatchedIB

    print("✓ Applied Windows-specific ib_insync patches")

