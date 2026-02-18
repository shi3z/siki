#!/bin/bash
# siki - 式神 (Shikigami) Setup Script
# Automatically detects platform and installs appropriate backend

set -e

echo "式神 (siki) - Setup Script"
echo "=========================="

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected: $OS $ARCH"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    echo "Please install Python 3.10 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is required but not installed."
    exit 1
fi

install_vllm() {
    echo "Installing vllm (GPU backend)..."
    pip3 install vllm --upgrade
    echo "vllm installed successfully!"
}

install_mlx() {
    echo "Installing mlx-lm (Apple Silicon backend)..."
    pip3 install mlx-lm --upgrade
    echo "mlx-lm installed successfully!"
}

install_ollama() {
    echo "Installing ollama (CPU/fallback backend)..."
    if [[ "$OS" == "Darwin" ]]; then
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    elif [[ "$OS" == "Linux" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Please install ollama manually: https://ollama.com"
        return 1
    fi
    echo "ollama installed successfully!"
}

install_hf_cli() {
    echo "Installing HuggingFace CLI..."
    pip3 install huggingface_hub[cli] --upgrade
    echo "HuggingFace CLI installed successfully!"
}

# Main installation logic
case "$OS" in
    Darwin)
        if [[ "$ARCH" == "arm64" ]]; then
            echo "Apple Silicon detected - installing mlx-lm..."
            install_mlx
        else
            echo "Intel Mac detected - installing ollama..."
            install_ollama
        fi
        ;;
    Linux)
        # Check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            echo "NVIDIA GPU detected - installing vllm..."
            install_vllm
        else
            echo "No NVIDIA GPU detected - installing ollama..."
            install_ollama
        fi
        ;;
    *)
        echo "Unknown OS: $OS"
        echo "Installing ollama as fallback..."
        install_ollama
        ;;
esac

# Always install HuggingFace CLI for model downloads
install_hf_cli

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Build siki:       make build"
echo "2. Download a model: ./siki download cyberagent/gpt-oss-20b"
echo "3. Start server:     ./siki serve gpt-oss-20b"
echo "4. Start chatting:   ./siki chat"
echo "5. Or use Web GUI:   ./siki web"
echo ""
