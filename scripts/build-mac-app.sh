#!/bin/bash
# Build Siki.app for macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
APP_DIR="$PROJECT_DIR/macos/Siki.app"
BUILD_DIR="$PROJECT_DIR/dist"

echo "Building Siki.app..."

# Build Go binary for macOS
cd "$PROJECT_DIR"
CGO_ENABLED=0 GOOS=darwin GOARCH=arm64 go build -ldflags="-s -w" -o "$APP_DIR/Contents/Resources/siki" .

# Also build for Intel Mac (universal binary would be better but this is simpler)
# CGO_ENABLED=0 GOOS=darwin GOARCH=amd64 go build -ldflags="-s -w" -o "$APP_DIR/Contents/Resources/siki-amd64" .

# Make launcher executable
chmod +x "$APP_DIR/Contents/MacOS/siki-launcher"

# Create dist directory
mkdir -p "$BUILD_DIR"

# Create DMG (requires create-dmg: brew install create-dmg)
if command -v create-dmg &> /dev/null; then
    echo "Creating DMG..."
    create-dmg \
        --volname "Siki" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "Siki.app" 150 185 \
        --app-drop-link 450 185 \
        "$BUILD_DIR/Siki-$(date +%Y%m%d).dmg" \
        "$APP_DIR"
    echo "DMG created: $BUILD_DIR/Siki-$(date +%Y%m%d).dmg"
else
    echo "create-dmg not found. Creating zip instead..."
    cd "$PROJECT_DIR/macos"
    zip -r "$BUILD_DIR/Siki.app.zip" Siki.app
    echo "Zip created: $BUILD_DIR/Siki.app.zip"
fi

echo "Done!"
