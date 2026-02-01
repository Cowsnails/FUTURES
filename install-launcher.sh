#!/bin/bash
# Installs the FUTURES desktop shortcut so you can double-click to launch.
# Run once: ./install-launcher.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$HOME/Desktop"
DESKTOP_FILE="$SCRIPT_DIR/futures-charting.desktop"

# Update paths in .desktop file to match current install location
sed -i "s|Exec=.*|Exec=bash -c 'cd $SCRIPT_DIR \&\& ./start.sh'|" "$DESKTOP_FILE"
sed -i "s|Icon=.*|Icon=$SCRIPT_DIR/assets/icon.png|" "$DESKTOP_FILE"

# Copy to desktop
if [ -d "$DESKTOP_DIR" ]; then
    cp "$DESKTOP_FILE" "$DESKTOP_DIR/"
    chmod +x "$DESKTOP_DIR/futures-charting.desktop"
    # Mark as trusted on GNOME
    if command -v gio &>/dev/null; then
        gio set "$DESKTOP_DIR/futures-charting.desktop" metadata::trusted true 2>/dev/null
    fi
    echo "Shortcut installed to $DESKTOP_DIR/futures-charting.desktop"
    echo "You can now double-click 'FUTURES Charting' on your desktop!"
else
    echo "No Desktop directory found at $DESKTOP_DIR"
    echo "You can manually copy futures-charting.desktop to your desktop."
fi

# Also install to applications menu
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"
cp "$DESKTOP_FILE" "$APP_DIR/"
echo "Also added to applications menu."
