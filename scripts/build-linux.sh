#!/bin/bash
# Build siki for Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist"

echo "Building siki for Linux..."

cd "$PROJECT_DIR"
mkdir -p "$BUILD_DIR"

# Build for Linux amd64
echo "Building for linux/amd64..."
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o "$BUILD_DIR/siki-linux-amd64" .

# Build for Linux arm64
echo "Building for linux/arm64..."
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -ldflags="-s -w" -o "$BUILD_DIR/siki-linux-arm64" .

# Create tarball
cd "$BUILD_DIR"
for arch in amd64 arm64; do
    tar -czvf "siki-linux-$arch.tar.gz" "siki-linux-$arch"
    echo "Created: siki-linux-$arch.tar.gz"
done

echo "Done!"
echo ""
echo "Installation on Linux:"
echo "  tar -xzf siki-linux-amd64.tar.gz"
echo "  sudo mv siki-linux-amd64 /usr/local/bin/siki"
echo "  siki web"
