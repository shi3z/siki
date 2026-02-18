# siki - 式神 (Shikigami) Agentic AI Environment
# Build script for multiple platforms

BINARY_NAME=siki
VERSION=0.1.0
BUILD_DIR=build
LDFLAGS=-ldflags "-s -w -X main.Version=$(VERSION)"

.PHONY: all build clean install darwin-arm64 darwin-amd64 linux-amd64 linux-arm64 windows

all: build

build:
	go build $(LDFLAGS) -o $(BINARY_NAME) .

# Cross-compilation targets
darwin-arm64:
	GOOS=darwin GOARCH=arm64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 .

darwin-amd64:
	GOOS=darwin GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-darwin-amd64 .

linux-amd64:
	GOOS=linux GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-amd64 .

linux-arm64:
	GOOS=linux GOARCH=arm64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-linux-arm64 .

windows:
	GOOS=windows GOARCH=amd64 go build $(LDFLAGS) -o $(BUILD_DIR)/$(BINARY_NAME)-windows-amd64.exe .

# Build all platforms
release: clean
	mkdir -p $(BUILD_DIR)
	$(MAKE) darwin-arm64
	$(MAKE) darwin-amd64
	$(MAKE) linux-amd64
	$(MAKE) linux-arm64
	$(MAKE) windows

# Build Mac .app bundle
mac-app: darwin-arm64
	@echo "Building Siki.app..."
	mkdir -p macos/Siki.app/Contents/{MacOS,Resources}
	cp $(BUILD_DIR)/$(BINARY_NAME)-darwin-arm64 macos/Siki.app/Contents/Resources/siki
	chmod +x macos/Siki.app/Contents/MacOS/siki-launcher
	@echo "Siki.app built at macos/Siki.app"

# Create Mac DMG
mac-dmg: mac-app
	@echo "Creating DMG..."
	mkdir -p dist
	cd macos && zip -r ../dist/Siki.app.zip Siki.app
	@echo "Created dist/Siki.app.zip"

install: build
	cp $(BINARY_NAME) /usr/local/bin/

clean:
	rm -f $(BINARY_NAME) stand
	rm -rf $(BUILD_DIR)

# Development helpers
run:
	go run . chat

web:
	go run . web

test:
	go test -v ./...

fmt:
	go fmt ./...

lint:
	golangci-lint run
