# 式神 Siki

**Siki** (式神, Shikigami) is a local agentic AI environment that runs entirely on your machine. Named after the supernatural servants in Onmyōdō tradition, Siki serves as your personal AI assistant with powerful tool-calling capabilities.

## Features

- **100% Local** - All AI processing happens on your machine
- **Single Binary** - One portable executable, no dependencies
- **Web GUI** - Beautiful chat interface with markdown support
- **Agent Tools** - File operations, command execution, web search, diagram generation
- **Twitter AI Curation** - Fetch timeline, auto-filter by topic with LLM keyword extraction, evaluate importance (HIGH/MID/LOW), and generate streaming summaries
- **Tool Result Pane** - Dedicated side panel for full tool output with auto-scroll and click-to-review past results
- **Real-time Progress** - Live SSE streaming of AI filtering, evaluation, and summarization progress
- **Multi-Model Pipeline** - LFM2.5 (fast 1.2B) for filtering/evaluation, gpt-oss (20B) for final summarization
- **Multi-Backend** - Supports Ollama, MLX-LM (Apple Silicon), and vLLM (NVIDIA GPU)
- **Image Generation** - FLUX.2 Klein 4B for text-to-image generation
- **Video Generation** - Helios for text-to-video generation
- **Mac App** - Native .app bundle for macOS

## Screenshots

<p align="center">
  <img src="docs/screenshot.png" alt="Siki Web GUI" width="600">
</p>

## Installation

### macOS (Homebrew)

```bash
brew tap shi3z/tap
brew install siki
```

### macOS (App Bundle)

Download `Siki.app.zip` from [Releases](https://github.com/shi3z/siki/releases), unzip, and drag to Applications.

### Linux

```bash
# Download binary
curl -LO https://github.com/shi3z/siki/releases/latest/download/siki-linux-amd64.tar.gz
tar -xzf siki-linux-amd64.tar.gz
sudo mv siki-linux-amd64 /usr/local/bin/siki
```

### From Source

```bash
git clone https://github.com/shi3z/siki.git
cd siki
make build
./siki web
```

## Quick Start

1. **Install Ollama** (if not already installed)
   ```bash
   brew install ollama  # macOS
   # or
   curl -fsSL https://ollama.com/install.sh | sh  # Linux
   ```

2. **Pull a model**
   ```bash
   ollama pull gpt-oss:20b
   ```

3. **Start Siki**
   ```bash
   siki web
   ```

4. Open http://localhost:3000 in your browser

## Usage

### Web GUI (Recommended)

```bash
siki web                      # Start web interface
siki web --port 8080          # Custom port
siki web --host 0.0.0.0       # Allow network access
```

### CLI Chat

```bash
siki chat                     # Interactive CLI chat
```

### Configuration

```bash
siki --backend ollama --model gpt-oss:20b web
siki --backend mlx --endpoint http://localhost:8000/v1 web
```

## Available Tools

Siki comes with powerful built-in tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `list_files` | List directory contents |
| `execute_command` | Run shell commands |
| `search_files` | Find files by pattern |
| `grep` | Search file contents |
| `web_search` | Search the internet |
| `web_fetch` | Fetch webpage content |
| `web_images` | Extract images from URLs |
| `diagram` | Generate diagrams with Graphviz |
| `twitter_timeline` | Fetch and AI-curate your Twitter timeline |
| `twitter_search` | Search Twitter with keyword queries |
| `generate_image` | Generate images with FLUX.2 Klein 4B |
| `generate_video` | Generate videos with Helios |
| `run_code` | Execute Python/JS code in sandbox |

## Backend Support

| Backend | Platform | GPU Required |
|---------|----------|--------------|
| Ollama | macOS, Linux, Windows | No |
| MLX-LM | macOS (Apple Silicon) | No (uses Neural Engine) |
| vLLM | Linux | Yes (NVIDIA) |

## Building

```bash
# Build for current platform
make build

# Build Mac app bundle
make mac-app

# Build for all platforms
make release

# Create distribution package
make mac-dmg
```

## Project Structure

```
siki/
├── main.go              # Main application
├── web/
│   └── index.html       # Web GUI (embedded)
├── macos/
│   └── Siki.app/        # Mac app bundle
├── scripts/
│   ├── build-mac-app.sh
│   └── build-linux.sh
├── Makefile
└── .goreleaser.yaml
```

## Twitter Integration

Siki can fetch your Twitter timeline, automatically filter tweets by topic, evaluate importance, and generate a curated summary — all locally.

1. Configure Twitter API credentials in the Settings panel (OAuth 1.0a)
2. Ask: "Twitterのタイムラインからニュースをまとめて"
3. Siki will:
   - Fetch your timeline via Twitter API v2
   - Extract the core topic using LFM2.5
   - Generate keywords and filter tweets by relevance
   - Evaluate each tweet's importance (HIGH/MID/LOW)
   - Generate a streaming summary with gpt-oss

All processing is visible in real-time via the tool result pane.

## Requirements

- Go 1.21+ (for building)
- Ollama, MLX-LM, or vLLM (for AI inference)
- Graphviz (optional, for diagram generation)
- Twitter API credentials (optional, for Twitter features)

## License

MIT License

## Acknowledgments

- Inspired by [picoclaw](https://github.com/sipeed/picoclaw)
- Powered by local LLMs via Ollama/MLX-LM/vLLM
- 式神 (Shikigami) concept from Japanese Onmyōdō tradition
