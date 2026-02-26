// siki - 式神 (Shikigami) Agentic AI Environment
// A portable, single-binary agentic AI tool that runs completely locally
// Named after Shikigami - supernatural servants in Onmyodo tradition
// Supports vllm (GPU), mlx-lm (Mac), and ollama backends

package main

import (
	"bufio"
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web/*
var webFS embed.FS

const (
	Version     = "0.1.0"
	DefaultPort = 8000
	MaxTurns    = 20
)

// ============================================================================
// Configuration
// ============================================================================

type Provider struct {
	Name     string `json:"name"`
	Backend  string `json:"backend"`  // ollama, vllm, mlx, openai, anthropic, gemini
	Endpoint string `json:"endpoint"`
	Model    string `json:"model"`
	APIKey   string `json:"api_key"`
}

type Config struct {
	ModelPath    string     `json:"model_path"`
	ModelName    string     `json:"model_name"`
	Backend      string     `json:"backend"`
	APIEndpoint  string     `json:"api_endpoint"`
	APIKey       string     `json:"api_key"`
	Providers    []Provider `json:"providers"`
	Workspace    string     `json:"workspace"`
	MaxTurns     int        `json:"max_turns"`
	SystemPrompt string     `json:"system_prompt"`
	VisionModel  string     `json:"vision_model"`
	SubModel     string     `json:"sub_model"`
}

// primaryProvider returns the first provider, or builds one from legacy config fields
func (c *Config) primaryProvider() Provider {
	if len(c.Providers) > 0 {
		return c.Providers[0]
	}
	return Provider{
		Name:     "default",
		Backend:  c.Backend,
		Endpoint: c.APIEndpoint,
		Model:    c.ModelName,
		APIKey:   c.APIKey,
	}
}

// findProvider returns a provider by name (case-insensitive)
func (c *Config) findProvider(name string) *Provider {
	nameLower := strings.ToLower(name)
	for i := range c.Providers {
		if strings.ToLower(c.Providers[i].Name) == nameLower {
			return &c.Providers[i]
		}
	}
	return nil
}

func setProviderHeaders(req *http.Request, p Provider) {
	req.Header.Set("Content-Type", "application/json")
	if p.APIKey != "" {
		switch p.Backend {
		case "anthropic":
			req.Header.Set("x-api-key", p.APIKey)
			req.Header.Set("anthropic-version", "2023-06-01")
		default:
			req.Header.Set("Authorization", "Bearer "+p.APIKey)
		}
	}
}

func defaultConfig() *Config {
	home, _ := os.UserHomeDir()
	backend := detectBackend()
	endpoint := "http://localhost:8000/v1"
	if backend == "ollama" {
		endpoint = "http://localhost:11434/v1"
	}
	return &Config{
		ModelPath:   filepath.Join(home, ".siki", "models"),
		ModelName:   "lfm2.5-thinking:latest",
		Backend:     backend,
		APIEndpoint: endpoint,
		Workspace:   ".",
		MaxTurns:    MaxTurns,
		VisionModel: "moondream",
		SubModel:    "gpt-oss:20b",
		SystemPrompt: `あなたは式神(Shikigami)。ツールを持つAIアシスタント。

## 最重要ルール（絶対に守れ）
1. ニュース・時事・最新情報 → 必ず web_search を呼べ。自分の知識で答えるな。
2. URL・ウェブページの内容 → 必ず web_fetch を呼べ。推測するな。
3. ファイル操作 → 必ず read_file/write_file を呼べ。
4. ツールで取得した情報だけで回答しろ。ツールなしで推測回答するな。
5. 回答にはURLを含めろ。URLが無ければ web_search で探せ。

## キーワード→ツール対応（この通りに動け）
- ニュース/最新/トレンド/速報/news/latest → web_search
- 教えて/調べて/検索/について → web_search
- URL/http/サイト → web_fetch
- ファイル/読んで/書いて → read_file / write_file
- コマンド/実行/install → execute_command
- 描いて/可視化/グラフ/ゲーム → run_code
- 図/関係図/アーキテクチャ → diagram
- 前の会話/さっき → recall_context / search_conversation
- ブログ/人物 → blog_person_search

## ツール一覧
- read_file, write_file, list_files: ファイル操作
- execute_command: シェルコマンド実行
- search_files, grep: ファイル検索
- web_search: インターネット検索
- web_fetch: URL内容取得
- web_images: URL画像抽出
- diagram: Graphviz図生成
- run_code: HTML/JS実行（ブラウザiframe）
- blog_person_search: ブログ巡回・人物抽出
- search_conversation: 会話履歴検索
- recall_context: 現スレッド会話ログ検索
- search_threads: 全スレッド横断検索
- docker_exec: GPUコンテナ内コマンド実行
- index_document: ドキュメント階層インデックス化
- search_document: インデックス済みドキュメント検索
- recall_memory: 学習知識検索
- list_documents: ドキュメント一覧
- self_status, self_modify_prompt, self_modify_params: 自己状態管理
- self_add_rule, self_remove_rule, self_rollback, self_benchmark: 自己改変
- self_evolve: ソースコード自己進化
- create_plugin, test_plugin, list_plugins, delete_plugin: プラグイン管理
- query_model: 他モデルに問い合わせ

## run_code
描画・可視化・ゲーム → 必ず run_code で実行。コード例を見せるな。
コード生成はサブモデルが自動で担当する。run_codeツールを呼べばよい。

## プラグイン
create_pluginで作成後、必ずtest_pluginでテストせよ。

## 主語省略の解決
日本語ユーザーは主語を省略する。直前の文脈から推測して対応しろ。「何についてですか？」と聞き返すな。

## 回答パターン
1. ツールを呼ぶ前に宣言しろ（例:「web_searchで調べます」「run_codeで描画します」）
2. ツールを呼ぶ
3. 結果に基づいて回答

ユーザーの言語で回答せよ。`,
	}
}

func detectBackend() string {
	if runtime.GOOS == "darwin" {
		// Check for Apple Silicon
		if runtime.GOARCH == "arm64" {
			return "mlx"
		}
		return "ollama"
	}
	// Check for NVIDIA GPU
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		return "vllm"
	}
	return "ollama"
}

// ============================================================================
// Tool Definitions
// ============================================================================

type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ToolCall struct {
	Index    int             `json:"index,omitempty"`
	ID       string          `json:"id"`
	Type     string          `json:"type"`
	Function ToolCallFunc    `json:"function"`
}

type ToolCallFunc struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Plugin represents a user-created plugin with optional tool and UI components
type Plugin struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Version     string      `json:"version"`
	Enabled     *bool       `json:"enabled,omitempty"`
	Tested      bool        `json:"tested"`
	TestResult  string      `json:"test_result,omitempty"`
	Tool        *PluginTool `json:"tool,omitempty"`
	UI          *PluginUI   `json:"ui,omitempty"`
}

func (p Plugin) IsEnabled() bool {
	return p.Enabled == nil || *p.Enabled
}

type PluginTool struct {
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Code        string                 `json:"code"`
}

type PluginUI struct {
	JS  string `json:"js"`
	CSS string `json:"css"`
}

var tools = []Tool{
	{
		Name:        "read_file",
		Description: "Read the contents of a file",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"path": map[string]interface{}{
					"type":        "string",
					"description": "Path to the file to read",
				},
			},
			"required": []string{"path"},
		},
	},
	{
		Name:        "write_file",
		Description: "Write content to a file",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"path": map[string]interface{}{
					"type":        "string",
					"description": "Path to the file to write",
				},
				"content": map[string]interface{}{
					"type":        "string",
					"description": "Content to write to the file",
				},
			},
			"required": []string{"path", "content"},
		},
	},
	{
		Name:        "list_files",
		Description: "List files in a directory",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"path": map[string]interface{}{
					"type":        "string",
					"description": "Directory path to list",
				},
			},
			"required": []string{"path"},
		},
	},
	{
		Name:        "execute_command",
		Description: "Execute a shell command",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"command": map[string]interface{}{
					"type":        "string",
					"description": "The command to execute",
				},
			},
			"required": []string{"command"},
		},
	},
	{
		Name:        "search_files",
		Description: "Search for files matching a pattern",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"pattern": map[string]interface{}{
					"type":        "string",
					"description": "Glob pattern to match files",
				},
				"path": map[string]interface{}{
					"type":        "string",
					"description": "Directory to search in",
				},
			},
			"required": []string{"pattern"},
		},
	},
	{
		Name:        "grep",
		Description: "Search for text in files",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"pattern": map[string]interface{}{
					"type":        "string",
					"description": "Text pattern to search for",
				},
				"path": map[string]interface{}{
					"type":        "string",
					"description": "File or directory to search in",
				},
			},
			"required": []string{"pattern"},
		},
	},
	{
		Name:        "web_search",
		Description: "Search the web using DuckDuckGo. Use this to find current information, news, documentation, etc.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "web_fetch",
		Description: "Fetch and extract text content from a web page URL",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "URL to fetch",
				},
			},
			"required": []string{"url"},
		},
	},
	{
		Name:        "web_images",
		Description: "Extract representative images from a web page (OGP image, main images). Returns markdown image syntax. Include the result directly in your response to display images.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "The URL to extract images from",
				},
			},
			"required": []string{"url"},
		},
	},
	{
		Name:        "diagram",
		Description: "Generate a diagram using Graphviz DOT language. Use this to visualize relationships, workflows, architectures, or any structured information. Returns markdown image syntax to display the diagram.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"dot_code": map[string]interface{}{
					"type":        "string",
					"description": "Graphviz DOT language code (e.g., 'digraph G { A -> B -> C }')",
				},
				"title": map[string]interface{}{
					"type":        "string",
					"description": "Title for the diagram (used in alt text)",
				},
			},
			"required": []string{"dot_code"},
		},
	},
	{
		Name:        "search_conversation",
		Description: "Search through the current conversation history for messages containing a keyword. Use this to recall what was discussed earlier in the conversation.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Keyword or phrase to search for in conversation history",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "recall_context",
		Description: "Search the current thread's COMPLETE conversation log for specific content. The log contains every message ever exchanged (never truncated). Use this to recall any detail from the conversation history.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query to find in conversation history",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "search_threads",
		Description: "Search across ALL conversation threads for specific content. Use this to find information from other conversation threads.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query to find across all threads",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "blog_person_search",
		Description: "Search a blog/website for person names mentioned in recent articles. Fetches the latest articles from a blog URL, reads each article, and uses AI to extract person names. Use this when asked about people related to a blogger, author, or website. Returns a list of person names with context about how they are mentioned.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "Blog or website top page URL (e.g., https://note.com/shi3zblog)",
				},
				"max_articles": map[string]interface{}{
					"type":        "number",
					"description": "Maximum number of articles to read (default: 5, max: 10)",
				},
			},
			"required": []string{"url"},
		},
	},
	{
		Name:        "run_code",
		Description: "Execute JavaScript/HTML code interactively in the browser. Use this to create visualizations, charts, interactive demos, calculations with visual output. The code runs in a sandboxed iframe. Returns an embedded view of the result.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"html": map[string]interface{}{
					"type":        "string",
					"description": "Complete HTML code including <script> tags for JavaScript. Can include CSS, Canvas, SVG, etc.",
				},
				"title": map[string]interface{}{
					"type":        "string",
					"description": "Title for the playground",
				},
			},
			"required": []string{"html"},
		},
	},
	{
		Name:        "docker_exec",
		Description: "Execute a command inside a GPU-enabled Docker container (siki-worker). The container has CUDA, ffmpeg, Python3, PyTorch, and openai-whisper pre-installed. Files uploaded via /api/upload are available in /workspace. Use this for GPU-intensive tasks like transcription, audio/video processing, and ML inference.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"command": map[string]interface{}{
					"type":        "string",
					"description": "Shell command to execute inside the container",
				},
				"timeout": map[string]interface{}{
					"type":        "number",
					"description": "Timeout in seconds (default: 300, max: 3600)",
				},
			},
			"required": []string{"command"},
		},
	},
	{
		Name:        "self_status",
		Description: "View your current self-state: version, system prompt, parameters, rules, benchmark score, and version history",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "self_modify_prompt",
		Description: "Modify your own system prompt. A snapshot is automatically created before modification. Changes take effect on new conversations.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action": map[string]interface{}{
					"type":        "string",
					"description": "replace: full replacement, append: add to end, replace_section: replace between ## markers",
				},
				"content": map[string]interface{}{
					"type":        "string",
					"description": "New content",
				},
				"section": map[string]interface{}{
					"type":        "string",
					"description": "Section header for replace_section (e.g. '## Available Tools')",
				},
				"reason": map[string]interface{}{
					"type":        "string",
					"description": "Why this change is being made",
				},
			},
			"required": []string{"action", "content", "reason"},
		},
	},
	{
		Name:        "self_modify_params",
		Description: "Modify your behavioral parameters: temperature, max_turns, compress_at, reflect_on_tools, preferred_lang, verbosity",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"params": map[string]interface{}{
					"type":        "string",
					"description": `JSON object of params to update, e.g. {"temperature":0.5,"verbosity":"concise"}`,
				},
				"reason": map[string]interface{}{
					"type":        "string",
					"description": "Why this change is being made",
				},
			},
			"required": []string{"params", "reason"},
		},
	},
	{
		Name:        "self_add_rule",
		Description: "Add a behavioral rule to yourself. Rules appear in your system prompt and guide your behavior.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"rule":   map[string]interface{}{"type": "string", "description": "The rule text"},
				"reason": map[string]interface{}{"type": "string", "description": "Why this rule is needed"},
			},
			"required": []string{"rule", "reason"},
		},
	},
	{
		Name:        "self_remove_rule",
		Description: "Deactivate a self-imposed behavioral rule by ID",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"rule_id": map[string]interface{}{"type": "string", "description": "ID of the rule to deactivate"},
			},
			"required": []string{"rule_id"},
		},
	},
	{
		Name:        "self_rollback",
		Description: "Rollback to a previous version of yourself. Use self_status to see available versions. Version 0 = factory default.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"version": map[string]interface{}{"type": "number", "description": "Version number to rollback to"},
			},
			"required": []string{"version"},
		},
	},
	{
		Name:        "self_benchmark",
		Description: "Run self-evaluation benchmarks using sub-model. Categories: tool_use, reasoning, language, helpfulness, or all",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"categories": map[string]interface{}{"type": "string", "description": "Comma-separated categories (default: all)"},
			},
		},
	},
	{
		Name:        "self_evolve",
		Description: "Modify your own Go source code, rebuild, test, and hot-reload. This is code-level self-modification with automatic backup, test gating, and graceful process restart.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"action": map[string]interface{}{
					"type":        "string",
					"description": "view_source: view source lines, patch: apply text replacement, build_test: compile and run tests, deploy: deploy tested binary and restart, status: show evolution state, abort: revert all patches",
				},
				"start_line": map[string]interface{}{
					"type":        "number",
					"description": "Start line for view_source (default: 1)",
				},
				"end_line": map[string]interface{}{
					"type":        "number",
					"description": "End line for view_source (default: 50)",
				},
				"old_text": map[string]interface{}{
					"type":        "string",
					"description": "Text to replace (for patch action)",
				},
				"new_text": map[string]interface{}{
					"type":        "string",
					"description": "Replacement text (for patch action)",
				},
				"description": map[string]interface{}{
					"type":        "string",
					"description": "Description of the patch or deployment reason",
				},
			},
			"required": []string{"action"},
		},
	},
	{
		Name:        "index_document",
		Description: "URLやテキストからドキュメントの階層インデックスを作成する。長文ドキュメントを構造化して後で検索可能にする",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "URL to fetch and index (optional if content is provided)",
				},
				"title": map[string]interface{}{
					"type":        "string",
					"description": "Document title",
				},
				"content": map[string]interface{}{
					"type":        "string",
					"description": "Text content to index (optional if url is provided)",
				},
			},
			"required": []string{"title"},
		},
	},
	{
		Name:        "search_document",
		Description: "インデックス済みドキュメントを推論ベースで検索する。ドキュメントの階層構造をたどって関連セクションを見つける",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query",
				},
				"doc_id": map[string]interface{}{
					"type":        "string",
					"description": "Document ID to search (optional, searches all if omitted)",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "recall_memory",
		Description: "学習済みの知識（playbook）からキーワード検索する。過去の経験から得た戦略・注意点・パターンを思い出す",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search keyword or topic",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "list_documents",
		Description: "インデックス済みドキュメント一覧を表示する",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
}

var pluginManagementTools = []Tool{
	{
		Name:        "create_plugin",
		Description: "Create or update a plugin. Plugins can add new tools (server-side Node.js) and/or UI modifications (client-side JS/CSS). Tool code receives a `params` object and outputs results via console.log().",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Plugin name (lowercase, underscores allowed, no spaces)",
				},
				"description": map[string]interface{}{
					"type":        "string",
					"description": "What this plugin does",
				},
				"tool_description": map[string]interface{}{
					"type":        "string",
					"description": "Description of the tool for AI (omit if plugin has no tool component)",
				},
				"tool_parameters": map[string]interface{}{
					"type":        "string",
					"description": "JSON string of the OpenAI-format parameters schema (omit if plugin has no tool)",
				},
				"tool_code": map[string]interface{}{
					"type":        "string",
					"description": "Node.js code for the tool. The `params` object is available as a global. Output result via console.log(). Can use require() for built-in modules.",
				},
				"ui_js": map[string]interface{}{
					"type":        "string",
					"description": "Client-side JavaScript for the plugin pane. Receives 'pane' (DOM element) as argument. Render all UI into pane only. Example: pane.innerHTML = '<p>Hello</p>';",
				},
				"ui_css": map[string]interface{}{
					"type":        "string",
					"description": "CSS scoped to the plugin's pane section (omit if no UI)",
				},
			},
			"required": []string{"name", "description"},
		},
	},
	{
		Name:        "list_plugins",
		Description: "List all installed plugins with their descriptions and capabilities.",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "delete_plugin",
		Description: "Delete a plugin by name.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Name of the plugin to delete",
				},
			},
			"required": []string{"name"},
		},
	},
	{
		Name:        "test_plugin",
		Description: "Run test cases against a plugin tool. Provide the plugin name and a JSON array of test cases. Each test case has 'input' (params object) and 'expected' (substring expected in output). All tests must pass to mark the plugin as tested.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Name of the plugin to test",
				},
				"test_cases": map[string]interface{}{
					"type":        "string",
					"description": `JSON array of test cases. Each: {"input": {<params>}, "expected": "<substring in output>"} Example: [{"input":{"x":2,"y":3},"expected":"5"}]`,
				},
			},
			"required": []string{"name", "test_cases"},
		},
	},
	{
		Name:        "query_model",
		Description: "Send a query to a specific configured AI provider/model. Use this to get a second opinion, compare answers, or leverage a model's specific strengths. Use list_plugins to see available providers first.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"provider": map[string]interface{}{
					"type":        "string",
					"description": "Name of the provider to query (as configured in settings)",
				},
				"message": map[string]interface{}{
					"type":        "string",
					"description": "The message/question to send to the model",
				},
				"system": map[string]interface{}{
					"type":        "string",
					"description": "Optional system prompt for this query",
				},
			},
			"required": []string{"provider", "message"},
		},
	},
}

// getAllTools returns built-in tools + plugin tools + plugin management tools
func getAllTools() []Tool {
	all := make([]Tool, len(tools))
	copy(all, tools)

	pluginMu.RLock()
	for _, p := range loadedPlugins {
		if p.IsEnabled() && p.Tool != nil {
			all = append(all, Tool{
				Name:        "plugin_" + p.Name,
				Description: p.Tool.Description + " [plugin]",
				Parameters:  p.Tool.Parameters,
			})
		}
	}
	pluginMu.RUnlock()

	all = append(all, pluginManagementTools...)
	return all
}

// coreToolNames are always included (small models choke on 30+ tool definitions)
var coreToolNames = map[string]bool{
	"web_search": true, "web_fetch": true, "read_file": true,
	"write_file": true, "list_files": true, "execute_command": true,
	"search_files": true, "grep": true, "diagram": true, "run_code": true,
}

// toolTriggers maps keywords in user messages to additional tool names
var toolTriggers = map[string][]string{
	"ブログ": {"blog_person_search"}, "人物": {"blog_person_search"},
	"前の会話": {"search_conversation", "recall_context"}, "さっき": {"search_conversation", "recall_context"},
	"会話": {"search_conversation", "recall_context", "search_threads"}, "思い出": {"recall_context", "recall_memory"},
	"スレッド": {"search_threads"}, "docker": {"docker_exec"}, "コンテナ": {"docker_exec"},
	"gpu": {"docker_exec"}, "ffmpeg": {"docker_exec"}, "whisper": {"docker_exec"},
	"インデックス": {"index_document", "search_document", "list_documents"},
	"ドキュメント": {"index_document", "search_document", "list_documents"},
	"document": {"index_document", "search_document", "list_documents"},
	"self": {"self_status", "self_modify_prompt", "self_modify_params", "self_add_rule", "self_remove_rule", "self_rollback", "self_benchmark", "self_evolve"},
	"自分": {"self_status", "self_modify_prompt", "self_modify_params", "self_add_rule", "self_remove_rule", "self_rollback", "self_benchmark", "self_evolve"},
	"改変": {"self_modify_prompt", "self_modify_params", "self_evolve"},
	"プラグイン": {"create_plugin", "test_plugin", "list_plugins", "delete_plugin"},
	"plugin": {"create_plugin", "test_plugin", "list_plugins", "delete_plugin"},
	"query_model": {"query_model"}, "他のモデル": {"query_model"},
	"画像": {"web_images"}, "image": {"web_images"},
}

// selectToolsForContext returns a filtered set of tools based on conversation context
func selectToolsForContext(messages []Message) []Tool {
	allTools := getAllTools()

	// Build a set of tool names to include
	include := make(map[string]bool)
	for name := range coreToolNames {
		include[name] = true
	}

	// Scan last few user messages for trigger keywords
	scanCount := 3
	for i := len(messages) - 1; i >= 0 && scanCount > 0; i-- {
		msg := messages[i]
		if msg.Role != "user" {
			continue
		}
		scanCount--
		lower := strings.ToLower(msg.Content)
		for keyword, toolNames := range toolTriggers {
			if strings.Contains(lower, keyword) {
				for _, tn := range toolNames {
					include[tn] = true
				}
			}
		}
	}

	// Also include any tool that was already called in this conversation
	for _, msg := range messages {
		for _, tc := range msg.ToolCalls {
			include[tc.Function.Name] = true
		}
	}

	// Filter
	var selected []Tool
	for _, t := range allTools {
		if include[t.Name] {
			selected = append(selected, t)
		}
	}

	return selected
}

// needsRunCode checks if a user request requires run_code (Canvas/JS) rather than diagram (Graphviz).
// Used to redirect when the model picks the wrong tool.
func needsRunCode(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	keywords := []string{
		"マンデルブロ", "フラクタル", "mandelbrot", "fractal",
		"シミュレーション", "simulation", "アニメーション", "animation",
		"ゲーム", "game", "3d", "canvas", "物理", "physics",
		"パーティクル", "particle", "波", "wave",
		"可視化", "visualization", "インタラクティブ", "interactive",
		"ライフゲーム", "life", "テトリス", "tetris", "スネーク", "snake",
		"迷路", "maze", "ソート", "sort",
	}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// autoToolFallback detects when the model should have called a tool but didn't,
// and automatically executes the appropriate tool. Returns the tool result if
// a fallback was triggered, empty string otherwise.
func autoToolFallback(agent *Agent, userMsg string, modelResponse string, sendEvent func(StreamEvent), saveMsg func(Message, string)) string {
	lower := strings.ToLower(userMsg)

	type fallback struct {
		keywords []string
		tool     string
		argsFn   func() map[string]interface{}
	}

	fallbacks := []fallback{
		{
			keywords: []string{"ニュース", "最新", "速報", "news", "latest", "トレンド"},
			tool:     "web_search",
			argsFn:   func() map[string]interface{} { return map[string]interface{}{"query": userMsg} },
		},
		{
			keywords: []string{"調べて", "検索して", "について教えて"},
			tool:     "web_search",
			argsFn:   func() map[string]interface{} { return map[string]interface{}{"query": userMsg} },
		},
		{
			keywords: []string{"描いて", "書いて", "可視化", "グラフ描", "ゲーム作", "フラクタル", "マンデルブロ", "シミュレーション", "アニメーション"},
			tool:     "run_code",
			argsFn: func() map[string]interface{} {
				// Delegate code generation to the sub-model (gpt-oss:20b)
				html, err := generateCodeWithSubModel(userMsg, agent.config)
				if err != nil {
					fmt.Printf("[siki] Sub-model code gen failed: %v, trying model response\n", err)
					// Fall back: try to extract HTML from the orchestrator's text response
					html = modelResponse
					if idx := strings.Index(html, "<html"); idx >= 0 {
						html = html[idx:]
						if end := strings.Index(html, "</html>"); end >= 0 {
							html = html[:end+7]
						}
					} else if idx := strings.Index(html, "```html"); idx >= 0 {
						html = html[idx+7:]
						if end := strings.Index(html, "```"); end >= 0 {
							html = html[:end]
						}
					}
					html = strings.TrimSpace(html)
					if html == "" || len(html) < 20 {
						return nil // skip this fallback
					}
					if !strings.Contains(html, "<html") {
						html = "<html><body>" + html + "</body></html>"
					}
				}
				return map[string]interface{}{"html": html}
			},
		},
	}

	for _, fb := range fallbacks {
		matched := false
		for _, kw := range fb.keywords {
			if strings.Contains(lower, kw) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}

		args := fb.argsFn()
		if args == nil {
			continue
		}

		fmt.Printf("[siki] Auto-fallback: model didn't call %s, executing automatically\n", fb.tool)
		sendEvent(StreamEvent{Type: "tool_start", Name: fb.tool})

		result, err := agent.executeTool(fb.tool, args)
		if err != nil {
			result = fmt.Sprintf("Error: %v", err)
		}

		displayResult := result
		if len(displayResult) > 2000 {
			displayResult = displayResult[:2000] + "\n... (truncated)"
		}

		sendEvent(StreamEvent{Type: "tool_call", Name: fb.tool, Result: displayResult})

		// Add tool call and result to conversation
		toolCallID := fmt.Sprintf("auto-%d", time.Now().UnixMilli())
		argsJSON, _ := json.Marshal(args)
		assistantMsg := Message{
			Role:    "assistant",
			Content: "",
			ToolCalls: []ToolCall{{
				ID:   toolCallID,
				Type: "function",
				Function: struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				}{Name: fb.tool, Arguments: string(argsJSON)},
			}},
		}
		// Replace last assistant message (the text response) with tool call
		if len(agent.messages) > 0 && agent.messages[len(agent.messages)-1].Role == "assistant" {
			agent.messages[len(agent.messages)-1] = assistantMsg
		}
		saveMsg(assistantMsg, "")

		toolMsg := Message{
			Role:       "tool",
			Content:    result,
			ToolCallID: toolCallID,
		}
		agent.messages = append(agent.messages, toolMsg)
		saveMsg(toolMsg, fb.tool)

		return result
	}

	return ""
}

// ============================================================================
// Tool Execution
// ============================================================================

type Agent struct {
	config   *Config
	messages []Message
	threadID string
}

// lastUserMessage returns the content of the most recent user message
func (a *Agent) lastUserMessage() string {
	for i := len(a.messages) - 1; i >= 0; i-- {
		if a.messages[i].Role == "user" {
			return a.messages[i].Content
		}
	}
	return ""
}

func defaultEndpointForBackend(backend string) string {
	switch backend {
	case "ollama":
		return "http://localhost:11434/v1"
	case "vllm", "mlx":
		return "http://localhost:8000/v1"
	case "openai":
		return "https://api.openai.com/v1"
	case "anthropic":
		return "https://api.anthropic.com/v1"
	case "gemini":
		return "https://generativelanguage.googleapis.com/v1beta/openai"
	default:
		return "http://localhost:11434/v1"
	}
}

func (a *Agent) setAuthHeaders(req *http.Request) {
	setProviderHeaders(req, a.config.primaryProvider())
}

func (a *Agent) executeTool(name string, args map[string]interface{}) (result string, err error) {
	// Recover from panics (e.g. nil type assertions when model omits required args)
	defer func() {
		if r := recover(); r != nil {
			result = ""
			err = fmt.Errorf("tool %s panicked: %v (check arguments)", name, r)
		}
	}()
	// Sanitize tool name: strip model artifacts like <|channel|>commentary
	if idx := strings.Index(name, "<"); idx != -1 {
		name = strings.TrimSpace(name[:idx])
	}
	switch name {
	case "read_file":
		return a.readFile(args["path"].(string))
	case "write_file":
		return a.writeFile(args["path"].(string), args["content"].(string))
	case "list_files":
		return a.listFiles(args["path"].(string))
	case "execute_command":
		return a.executeCommand(args["command"].(string))
	case "search_files":
		path := "."
		if p, ok := args["path"].(string); ok {
			path = p
		}
		return a.searchFiles(args["pattern"].(string), path)
	case "grep":
		path := "."
		if p, ok := args["path"].(string); ok {
			path = p
		}
		return a.grep(args["pattern"].(string), path)
	case "web_search":
		return a.webSearch(args["query"].(string))
	case "web_fetch":
		return a.webFetch(args["url"].(string))
	case "web_images":
		return a.webImages(args["url"].(string))
	case "diagram":
		title := "Diagram"
		if t, ok := args["title"].(string); ok {
			title = t
		}
		return a.generateDiagram(args["dot_code"].(string), title)
	case "search_conversation":
		return a.searchConversation(args["query"].(string)), nil
	case "recall_context":
		return a.recallContext(args["query"].(string))
	case "search_threads":
		return a.searchAllThreads(args["query"].(string))
	case "blog_person_search":
		maxArticles := 5
		if n, ok := args["max_articles"].(float64); ok {
			maxArticles = int(n)
		}
		if maxArticles > 10 {
			maxArticles = 10
		}
		if maxArticles < 1 {
			maxArticles = 5
		}
		return a.blogPersonSearch(args["url"].(string), maxArticles)
	case "run_code":
		title := "Playground"
		if t, ok := args["title"].(string); ok {
			title = t
		}
		html, _ := args["html"].(string)
		if html == "" {
			return "", fmt.Errorf("html parameter is required")
		}
		return a.runCode(html, title)
	case "docker_exec":
		timeout := 300
		if t, ok := args["timeout"].(float64); ok {
			timeout = int(t)
		}
		return dockerExec(args["command"].(string), timeout)
	case "create_plugin":
		return a.createPlugin(args)
	case "list_plugins":
		return a.listPlugins()
	case "delete_plugin":
		return a.deletePlugin(args["name"].(string))
	case "test_plugin":
		return a.testPlugin(args["name"].(string), args["test_cases"].(string))
	case "query_model":
		systemPrompt, _ := args["system"].(string)
		return a.queryModel(args["provider"].(string), args["message"].(string), systemPrompt)
	case "self_status":
		return a.selfStatus()
	case "self_modify_prompt":
		return a.selfModifyPrompt(args)
	case "self_modify_params":
		return a.selfModifyParams(args)
	case "self_add_rule":
		return a.selfAddRule(args)
	case "self_remove_rule":
		return a.selfRemoveRule(args)
	case "self_rollback":
		return a.selfRollback(args)
	case "self_benchmark":
		return a.selfBenchmark(args)
	case "self_evolve":
		return a.selfEvolve(args)
	case "index_document":
		title, _ := args["title"].(string)
		docURL, _ := args["url"].(string)
		content, _ := args["content"].(string)
		if content == "" && docURL != "" {
			// Fetch content from URL
			fetched, err := a.webFetch(docURL)
			if err != nil {
				return "", fmt.Errorf("failed to fetch URL: %w", err)
			}
			content = fetched
		}
		if content == "" {
			return "", fmt.Errorf("either url or content is required")
		}
		doc, err := indexDocument(a.config, title, content, docURL)
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("ドキュメント「%s」をインデックスしました (ID: %s)\n\n", doc.Title, doc.ID))
		sb.WriteString("## 構造:\n")
		var showTree func(sections []DocSection, depth int)
		showTree = func(sections []DocSection, depth int) {
			for _, s := range sections {
				indent := strings.Repeat("  ", depth)
				sb.WriteString(fmt.Sprintf("%s- [%s] %s: %s\n", indent, s.ID, s.Title, s.Summary))
				if len(s.Children) > 0 {
					showTree(s.Children, depth+1)
				}
			}
		}
		showTree(doc.Sections, 0)
		return sb.String(), nil
	case "search_document":
		query, _ := args["query"].(string)
		docID, _ := args["doc_id"].(string)
		if docID != "" {
			doc, err := loadDocumentIndex(docID)
			if err != nil {
				return "", fmt.Errorf("document not found: %s", docID)
			}
			return searchDocumentTree(a.config, doc, query)
		}
		// Search all documents
		docs, err := listDocuments()
		if err != nil || len(docs) == 0 {
			return "インデックス済みドキュメントがありません。index_documentツールで先にインデックスしてください。", nil
		}
		var results strings.Builder
		for _, doc := range docs {
			result, err := searchDocumentTree(a.config, &doc, query)
			if err != nil {
				continue
			}
			results.WriteString(result + "\n---\n")
		}
		if results.Len() == 0 {
			return "関連するセクションが見つかりませんでした。", nil
		}
		return results.String(), nil
	case "recall_memory":
		query, _ := args["query"].(string)
		bullets, err := loadPlaybook()
		if err != nil || len(bullets) == 0 {
			return "学習済みの知識はまだありません。", nil
		}
		queryLower := strings.ToLower(query)
		keywords := strings.Fields(queryLower)
		var matches []PlaybookBullet
		for _, b := range bullets {
			contentLower := strings.ToLower(b.Content)
			for _, kw := range keywords {
				if strings.Contains(contentLower, kw) {
					matches = append(matches, b)
					break
				}
			}
		}
		if len(matches) == 0 {
			return fmt.Sprintf("「%s」に関連する知識は見つかりませんでした。\n\n全%d件のbulletが保存されています。", query, len(bullets)), nil
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("「%s」に関連する知識 (%d件):\n\n", query, len(matches)))
		for _, m := range matches {
			sb.WriteString(fmt.Sprintf("- [%s] %s (確認×%d)\n", m.Type, m.Content, m.Hits))
		}
		return sb.String(), nil
	case "list_documents":
		docs, err := listDocuments()
		if err != nil {
			return "", err
		}
		if len(docs) == 0 {
			return "インデックス済みドキュメントはありません。", nil
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("インデックス済みドキュメント (%d件):\n\n", len(docs)))
		for _, doc := range docs {
			sb.WriteString(fmt.Sprintf("- **%s** (ID: %s)", doc.Title, doc.ID))
			if doc.URL != "" {
				sb.WriteString(fmt.Sprintf(" [%s]", doc.URL))
			}
			sb.WriteString(fmt.Sprintf(" - %d sections\n", len(doc.Sections)))
		}
		return sb.String(), nil
	default:
		// Check if this is a plugin tool
		if strings.HasPrefix(name, "plugin_") {
			pluginName := strings.TrimPrefix(name, "plugin_")
			return executePluginTool(pluginName, args)
		}
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

// ============================================================================
// Plugin Tool Implementations
// ============================================================================

func (a *Agent) createPlugin(args map[string]interface{}) (string, error) {
	name, _ := args["name"].(string)
	description, _ := args["description"].(string)

	if name == "" {
		return "", fmt.Errorf("plugin name is required")
	}
	for _, r := range name {
		if !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_') {
			return "", fmt.Errorf("plugin name must be lowercase alphanumeric with underscores only")
		}
	}

	p := Plugin{
		Name:        name,
		Description: description,
		Version:     "1.0",
		Tested:      false,
	}

	if toolCode, ok := args["tool_code"].(string); ok && toolCode != "" {
		toolDesc, _ := args["tool_description"].(string)
		if toolDesc == "" {
			toolDesc = description
		}

		var params map[string]interface{}
		if paramStr, ok := args["tool_parameters"].(string); ok && paramStr != "" {
			if err := json.Unmarshal([]byte(paramStr), &params); err != nil {
				return "", fmt.Errorf("invalid tool_parameters JSON: %v", err)
			}
		} else {
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}

		p.Tool = &PluginTool{
			Description: toolDesc,
			Parameters:  params,
			Code:        toolCode,
		}
	}

	uiJS, _ := args["ui_js"].(string)
	uiCSS, _ := args["ui_css"].(string)
	if uiJS != "" || uiCSS != "" {
		p.UI = &PluginUI{JS: uiJS, CSS: uiCSS}
	}

	if err := savePlugin(p); err != nil {
		return "", fmt.Errorf("failed to save plugin: %v", err)
	}

	if err := loadPlugins(); err != nil {
		return "", fmt.Errorf("plugin saved but failed to reload: %v", err)
	}

	result := fmt.Sprintf("Plugin '%s' created successfully. [UNTESTED]", name)
	if p.Tool != nil {
		result += fmt.Sprintf(" Tool 'plugin_%s' is now available.", name)
		result += " You MUST now use test_plugin to run test cases before considering this plugin ready."
	}
	if p.UI != nil {
		result += " UI components will load on next page refresh."
	}
	return result, nil
}

func (a *Agent) listPlugins() (string, error) {
	var sb strings.Builder

	// List configured providers
	if len(a.config.Providers) > 0 {
		sb.WriteString(fmt.Sprintf("## Configured Providers (%d):\n", len(a.config.Providers)))
		for i, p := range a.config.Providers {
			primary := ""
			if i == 0 {
				primary = " [PRIMARY]"
			}
			sb.WriteString(fmt.Sprintf("\n- **%s**%s: %s / %s", p.Name, primary, p.Backend, p.Model))
		}
		sb.WriteString("\n\nUse query_model to send a query to any provider.\n")
	} else {
		sb.WriteString("## Providers: only default (use Settings to add more)\n")
	}

	// List plugins
	pluginMu.RLock()
	defer pluginMu.RUnlock()

	if len(loadedPlugins) == 0 {
		sb.WriteString("\n## Plugins: none installed")
	} else {
		sb.WriteString(fmt.Sprintf("\n## Installed Plugins (%d):\n", len(loadedPlugins)))
		for _, p := range loadedPlugins {
			status := "ON"
			if !p.IsEnabled() {
				status = "OFF"
			}
			testStatus := "UNTESTED"
			if p.Tested {
				testStatus = "TESTED"
			}
			sb.WriteString(fmt.Sprintf("\n- %s (v%s) [%s][%s]: %s", p.Name, p.Version, status, testStatus, p.Description))
			if p.Tool != nil {
				sb.WriteString(fmt.Sprintf("\n  Tool: plugin_%s - %s", p.Name, p.Tool.Description))
			}
			if p.UI != nil {
				has := []string{}
				if p.UI.JS != "" {
					has = append(has, "JS")
				}
				if p.UI.CSS != "" {
					has = append(has, "CSS")
				}
				sb.WriteString(fmt.Sprintf("\n  UI: %s", strings.Join(has, ", ")))
			}
		}
	}
	return sb.String(), nil
}

func (a *Agent) deletePlugin(name string) (string, error) {
	if err := deletePluginFile(name); err != nil {
		return "", fmt.Errorf("failed to delete plugin '%s': %v", name, err)
	}
	if err := loadPlugins(); err != nil {
		return "", fmt.Errorf("plugin deleted but failed to reload: %v", err)
	}
	return fmt.Sprintf("Plugin '%s' deleted successfully.", name), nil
}

func (a *Agent) testPlugin(name, testCasesJSON string) (string, error) {
	// Parse test cases
	type TestCase struct {
		Input    map[string]interface{} `json:"input"`
		Expected string                 `json:"expected"`
	}
	var testCases []TestCase
	if err := json.Unmarshal([]byte(testCasesJSON), &testCases); err != nil {
		return "", fmt.Errorf("invalid test_cases JSON: %v", err)
	}
	if len(testCases) == 0 {
		return "", fmt.Errorf("at least one test case is required")
	}

	// Check plugin exists and has a tool
	pluginMu.RLock()
	var plugin *Plugin
	for i := range loadedPlugins {
		if loadedPlugins[i].Name == name {
			plugin = &loadedPlugins[i]
			break
		}
	}
	pluginMu.RUnlock()

	if plugin == nil {
		return "", fmt.Errorf("plugin '%s' not found", name)
	}
	if plugin.Tool == nil {
		return "", fmt.Errorf("plugin '%s' has no tool component to test", name)
	}

	// Run each test case
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Test Results for plugin '%s'\n\n", name))
	passed := 0
	failed := 0

	for i, tc := range testCases {
		output, err := executePluginTool(name, tc.Input)
		inputJSON, _ := json.Marshal(tc.Input)

		if err != nil {
			failed++
			sb.WriteString(fmt.Sprintf("### Test %d: FAIL\n", i+1))
			sb.WriteString(fmt.Sprintf("- Input: `%s`\n", string(inputJSON)))
			sb.WriteString(fmt.Sprintf("- Error: %v\n\n", err))
			continue
		}

		if tc.Expected != "" && !strings.Contains(output, tc.Expected) {
			failed++
			sb.WriteString(fmt.Sprintf("### Test %d: FAIL\n", i+1))
			sb.WriteString(fmt.Sprintf("- Input: `%s`\n", string(inputJSON)))
			sb.WriteString(fmt.Sprintf("- Expected to contain: `%s`\n", tc.Expected))
			sb.WriteString(fmt.Sprintf("- Actual output: `%s`\n\n", output))
		} else {
			passed++
			sb.WriteString(fmt.Sprintf("### Test %d: PASS\n", i+1))
			sb.WriteString(fmt.Sprintf("- Input: `%s`\n", string(inputJSON)))
			sb.WriteString(fmt.Sprintf("- Output: `%s`\n\n", output))
		}
	}

	allPassed := failed == 0
	sb.WriteString(fmt.Sprintf("---\n**Result: %d/%d passed**", passed, len(testCases)))

	if allPassed {
		// Mark plugin as tested
		pluginMu.Lock()
		for i := range loadedPlugins {
			if loadedPlugins[i].Name == name {
				loadedPlugins[i].Tested = true
				loadedPlugins[i].TestResult = fmt.Sprintf("%d/%d passed", passed, len(testCases))
				if err := savePlugin(loadedPlugins[i]); err != nil {
					pluginMu.Unlock()
					return sb.String() + "\n\nWarning: failed to save test status: " + err.Error(), nil
				}
				break
			}
		}
		pluginMu.Unlock()
		sb.WriteString(" - Plugin marked as TESTED")
	} else {
		sb.WriteString(fmt.Sprintf(" (%d failed) - Plugin remains UNTESTED. Fix the issues and re-test.", failed))
	}

	return sb.String(), nil
}

func (a *Agent) queryModel(providerName, message, systemPrompt string) (string, error) {
	p := a.config.findProvider(providerName)
	if p == nil {
		// List available providers for the AI
		var names []string
		for _, prov := range a.config.Providers {
			names = append(names, prov.Name)
		}
		return "", fmt.Errorf("provider '%s' not found. Available: %s", providerName, strings.Join(names, ", "))
	}

	messages := []Message{}
	if systemPrompt != "" {
		messages = append(messages, Message{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, Message{Role: "user", Content: message})

	req := ChatRequest{
		Model:       p.Model,
		Messages:    messages,
		MaxTokens:   8192,
		Temperature: 0.7,
		Stream:      false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		p.Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	setProviderHeaders(httpReq, *p)

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("query to %s failed: %v", providerName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("query to %s returned %d: %s", providerName, resp.StatusCode, string(respBody))
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("failed to decode response from %s: %v", providerName, err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no response from %s", providerName)
	}

	result := chatResp.Choices[0].Message.Content
	return fmt.Sprintf("[%s/%s の回答]\n%s", providerName, p.Model, result), nil
}

func executePluginTool(pluginName string, args map[string]interface{}) (string, error) {
	pluginMu.RLock()
	var plugin *Plugin
	for i := range loadedPlugins {
		if loadedPlugins[i].Name == pluginName {
			plugin = &loadedPlugins[i]
			break
		}
	}
	pluginMu.RUnlock()

	if plugin == nil {
		return "", fmt.Errorf("plugin '%s' not found", pluginName)
	}
	if plugin.Tool == nil {
		return "", fmt.Errorf("plugin '%s' has no tool component", pluginName)
	}

	paramsJSON, err := json.Marshal(args)
	if err != nil {
		return "", fmt.Errorf("failed to serialize params: %v", err)
	}

	wrapper := fmt.Sprintf("const params = %s;\n%s", string(paramsJSON), plugin.Tool.Code)

	tmpFile, err := os.CreateTemp("", "siki-plugin-*.js")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(wrapper); err != nil {
		tmpFile.Close()
		return "", fmt.Errorf("failed to write temp file: %v", err)
	}
	tmpFile.Close()

	nodePath := "/home/username/.nvm/versions/node/v24.13.0/bin/node"
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, nodePath, tmpFile.Name())
	fmt.Printf("[siki] Executing plugin '%s'\n", pluginName)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("Plugin execution error: %v\nOutput: %s", err, string(output)), nil
	}

	return strings.TrimSpace(string(output)), nil
}

func (a *Agent) readFile(path string) (string, error) {
	absPath := a.resolvePath(path)
	content, err := os.ReadFile(absPath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func (a *Agent) writeFile(path, content string) (string, error) {
	absPath := a.resolvePath(path)
	dir := filepath.Dir(absPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	if err := os.WriteFile(absPath, []byte(content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("Successfully wrote %d bytes to %s", len(content), path), nil
}

func (a *Agent) listFiles(path string) (string, error) {
	absPath := a.resolvePath(path)
	entries, err := os.ReadDir(absPath)
	if err != nil {
		return "", err
	}
	var result strings.Builder
	for _, entry := range entries {
		info, _ := entry.Info()
		if info != nil {
			result.WriteString(fmt.Sprintf("%s %8d %s\n",
				info.Mode(), info.Size(), entry.Name()))
		} else {
			result.WriteString(entry.Name() + "\n")
		}
	}
	return result.String(), nil
}

func (a *Agent) executeCommand(command string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	cmd.Dir = a.config.Workspace
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output) + "\nError: " + err.Error(), nil
	}
	return string(output), nil
}

func (a *Agent) searchFiles(pattern, path string) (string, error) {
	absPath := a.resolvePath(path)
	var matches []string
	err := filepath.Walk(absPath, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		matched, _ := filepath.Match(pattern, filepath.Base(p))
		if matched {
			rel, _ := filepath.Rel(absPath, p)
			matches = append(matches, rel)
		}
		return nil
	})
	if err != nil {
		return "", err
	}
	return strings.Join(matches, "\n"), nil
}

func (a *Agent) grep(pattern, path string) (string, error) {
	absPath := a.resolvePath(path)
	cmd := exec.Command("grep", "-rn", "--include=*", pattern, absPath)
	output, _ := cmd.CombinedOutput()
	return string(output), nil
}

func (a *Agent) webSearch(query string) (string, error) {
	// Detect time-sensitive queries and auto-append current date context
	timeSensitivePatterns := []string{
		"ニュース", "最新", "最近", "今日", "今週", "今月", "速報", "動向", "トレンド",
		"news", "latest", "recent", "today", "current", "trending", "update",
	}
	queryLower := strings.ToLower(query)
	isTimeSensitive := false
	for _, pat := range timeSensitivePatterns {
		if strings.Contains(queryLower, pat) {
			isTimeSensitive = true
			break
		}
	}

	// Build search URL with date filter for time-sensitive queries
	now := time.Now()
	searchQuery := query
	dateFilter := ""
	if isTimeSensitive {
		// Append year+month to bias results toward current time
		hasYear := false
		for y := now.Year() - 1; y <= now.Year(); y++ {
			if strings.Contains(query, fmt.Sprintf("%d", y)) {
				hasYear = true
				break
			}
		}
		if !hasYear {
			searchQuery = fmt.Sprintf("%s %d年%d月", query, now.Year(), int(now.Month()))
		}
		dateFilter = "&df=m" // DuckDuckGo: filter to past month
	}

	// Use DuckDuckGo HTML search
	encodedQuery := url.QueryEscape(searchQuery)
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s%s", encodedQuery, dateFilter)

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; siki/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("search request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Parse search results from HTML
	html := string(body)
	var results []string

	// Extract result snippets (simple parsing)
	// Look for result links and snippets
	parts := strings.Split(html, "result__a")
	for i, part := range parts {
		if i == 0 || i > 10 {
			continue
		}

		// Extract URL
		urlStart := strings.Index(part, "href=\"")
		if urlStart == -1 {
			continue
		}
		urlStart += 6
		urlEnd := strings.Index(part[urlStart:], "\"")
		if urlEnd == -1 {
			continue
		}
		url := part[urlStart : urlStart+urlEnd]

		// Clean up DuckDuckGo redirect URL
		if strings.Contains(url, "uddg=") {
			if idx := strings.Index(url, "uddg="); idx != -1 {
				url = url[idx+5:]
				if ampIdx := strings.Index(url, "&"); ampIdx != -1 {
					url = url[:ampIdx]
				}
				// URL decode
				url = strings.ReplaceAll(url, "%3A", ":")
				url = strings.ReplaceAll(url, "%2F", "/")
				url = strings.ReplaceAll(url, "%3F", "?")
				url = strings.ReplaceAll(url, "%3D", "=")
				url = strings.ReplaceAll(url, "%26", "&")
			}
		}

		// Extract title
		titleStart := strings.Index(part, ">")
		if titleStart == -1 {
			continue
		}
		titleStart++
		titleEnd := strings.Index(part[titleStart:], "<")
		title := ""
		if titleEnd != -1 {
			title = part[titleStart : titleStart+titleEnd]
		}

		// Extract snippet
		snippetStart := strings.Index(part, "result__snippet")
		snippet := ""
		if snippetStart != -1 {
			snippetPart := part[snippetStart:]
			snippetContentStart := strings.Index(snippetPart, ">")
			if snippetContentStart != -1 {
				snippetContentStart++
				snippetContentEnd := strings.Index(snippetPart[snippetContentStart:], "<")
				if snippetContentEnd != -1 {
					snippet = snippetPart[snippetContentStart : snippetContentStart+snippetContentEnd]
				}
			}
		}

		// Clean HTML entities
		title = cleanHTMLEntities(title)
		snippet = cleanHTMLEntities(snippet)

		if title != "" || url != "" {
			result := fmt.Sprintf("**%s**\n%s\n%s\n", title, url, snippet)
			results = append(results, result)
		}
	}

	if len(results) == 0 {
		return "No search results found.", nil
	}

	header := fmt.Sprintf("Search results for: %s", query)
	if searchQuery != query {
		header += fmt.Sprintf(" (searched: %s)", searchQuery)
	}
	return fmt.Sprintf("%s\n\n%s", header, strings.Join(results, "\n---\n")), nil
}

func cleanHTMLEntities(s string) string {
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&#39;", "'")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	s = strings.TrimSpace(s)
	return s
}

func (a *Agent) webFetch(url string) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; siki/1.0)")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	// Extract text content from HTML
	html := string(body)
	text := extractTextFromHTML(html)

	// Truncate if too long
	if len(text) > 15000 {
		text = text[:15000] + "\n\n... (truncated)"
	}

	return fmt.Sprintf("Content from %s:\n\n%s", url, text), nil
}

func (a *Agent) blogPersonSearch(blogURL string, maxArticles int) (string, error) {
	client := &http.Client{Timeout: 30 * time.Second}

	// Step 1: Get article URLs
	var articleURLs []string

	parsedURL, _ := url.Parse(blogURL)

	if strings.Contains(parsedURL.Host, "note.com") {
		// note.com: use API
		username := strings.Trim(parsedURL.Path, "/")
		apiURL := fmt.Sprintf("https://note.com/api/v2/creators/%s/contents?kind=note&page=1&per_page=%d", username, maxArticles)
		fmt.Printf("[siki] Fetching note.com API: %s\n", apiURL)

		req, err := http.NewRequest("GET", apiURL, nil)
		if err != nil {
			return "", err
		}
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

		resp, err := client.Do(req)
		if err != nil {
			return "", fmt.Errorf("failed to fetch note.com API: %w", err)
		}
		defer resp.Body.Close()

		var noteResp struct {
			Data struct {
				Contents []struct {
					Key     string `json:"key"`
					Name    string `json:"name"`
					Body    string `json:"body"`
					User    struct {
						Urlname string `json:"urlname"`
					} `json:"user"`
				} `json:"contents"`
			} `json:"data"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&noteResp); err != nil {
			return "", fmt.Errorf("failed to parse note.com API response: %w", err)
		}

		// note.com API includes body text directly - use it!
		var articleSummaries []string
		for i, content := range noteResp.Data.Contents {
			articleURL := fmt.Sprintf("https://note.com/%s/n/%s", content.User.Urlname, content.Key)
			fmt.Printf("[siki] Processing article %d/%d: %s (%s)\n", i+1, len(noteResp.Data.Contents), content.Name, articleURL)

			articleText := content.Body
			if articleText == "" {
				// If body is empty (paid article?), try fetching the page
				articleText = a.fetchArticleText(client, articleURL)
			}
			if len(articleText) > 8000 {
				articleText = articleText[:8000]
			}
			if articleText == "" {
				continue
			}

			persons := a.extractPersonNames(articleURL, content.Name+"\n\n"+articleText)
			if persons != "" && persons != "人物名なし" {
				articleSummaries = append(articleSummaries, fmt.Sprintf("### 記事 %d: %s\n%s\n%s", i+1, content.Name, articleURL, persons))
			}
		}

		if len(articleSummaries) == 0 {
			return fmt.Sprintf("%d件の記事を分析しましたが、人物名は抽出できませんでした。", len(noteResp.Data.Contents)), nil
		}

		return fmt.Sprintf("## %s から抽出した人物名一覧\n\n（%d件の記事を分析）\n\n%s",
			blogURL, len(noteResp.Data.Contents), strings.Join(articleSummaries, "\n\n")), nil

	} else {
		// General blog: try HTML parsing
		req, err := http.NewRequest("GET", blogURL, nil)
		if err != nil {
			return "", err
		}
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

		resp, err := client.Do(req)
		if err != nil {
			return "", fmt.Errorf("failed to fetch blog page: %w", err)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}

		articleURLs = extractArticleLinks(string(body), blogURL)
		if len(articleURLs) == 0 {
			return "No articles found on the page.", nil
		}
		if len(articleURLs) > maxArticles {
			articleURLs = articleURLs[:maxArticles]
		}
	}

	// Process general blog articles
	var articleSummaries []string
	for i, articleURL := range articleURLs {
		fmt.Printf("[siki] Reading article %d/%d: %s\n", i+1, len(articleURLs), articleURL)

		articleText := a.fetchArticleText(client, articleURL)
		if len(articleText) > 8000 {
			articleText = articleText[:8000]
		}
		if articleText == "" {
			continue
		}

		persons := a.extractPersonNames(articleURL, articleText)
		if persons != "" && persons != "人物名なし" {
			articleSummaries = append(articleSummaries, fmt.Sprintf("### 記事 %d: %s\n%s", i+1, articleURL, persons))
		}
	}

	if len(articleSummaries) == 0 {
		return fmt.Sprintf("Read %d articles from %s but could not extract any person names.", len(articleURLs), blogURL), nil
	}

	return fmt.Sprintf("## %s から抽出した人物名一覧\n\n（%d件の記事を分析）\n\n%s",
		blogURL, len(articleURLs), strings.Join(articleSummaries, "\n\n")), nil
}

func (a *Agent) fetchArticleText(client *http.Client, articleURL string) string {
	req, err := http.NewRequest("GET", articleURL, nil)
	if err != nil {
		return ""
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ""
	}
	return extractTextFromHTML(string(body))
}

func extractArticleLinks(html string, baseURL string) []string {
	var urls []string
	seen := make(map[string]bool)

	base, _ := url.Parse(baseURL)

	re := regexp.MustCompile(`href=["']([^"']+)["']`)
	matches := re.FindAllStringSubmatch(html, -1)

	for _, m := range matches {
		if len(m) < 2 {
			continue
		}
		link := m[1]

		parsed, err := url.Parse(link)
		if err != nil {
			continue
		}
		resolved := base.ResolveReference(parsed)
		fullURL := resolved.String()

		if !isArticleURL(fullURL, baseURL) {
			continue
		}

		if !seen[fullURL] {
			seen[fullURL] = true
			urls = append(urls, fullURL)
		}
	}

	return urls
}

func isArticleURL(articleURL string, baseURL string) bool {
	baseP, _ := url.Parse(baseURL)
	artP, _ := url.Parse(articleURL)

	if baseP.Host != artP.Host {
		return false
	}

	path := artP.Path

	if strings.Contains(baseURL, "note.com") {
		return regexp.MustCompile(`/[^/]+/n/[a-zA-Z0-9]+`).MatchString(path)
	}

	segments := strings.Split(strings.Trim(path, "/"), "/")
	if len(segments) < 2 {
		return false
	}
	skipPrefixes := []string{"tag", "category", "page", "about", "contact", "search", "login", "signup"}
	for _, skip := range skipPrefixes {
		if strings.EqualFold(segments[0], skip) {
			return false
		}
	}
	return true
}

func (a *Agent) extractPersonNames(articleURL string, articleText string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	extractMessages := []Message{
		{Role: "system", Content: `あなはテキストから人物名（個人名）を抽出する専門家です。
以下の記事テキストから、言及されている実在の人物の名前を全て抽出してください。

ルール:
- 実在の個人名のみ（企業名・団体名は除外）
- 著者自身は除外
- 名前と、どのような文脈で言及されているかを簡潔に記載
- 人物が見つからない場合は「人物名なし」と回答

フォーマット:
- **人物名**: 文脈の説明（1行で）`},
		{Role: "user", Content: fmt.Sprintf("記事URL: %s\n\n記事テキスト:\n%s", articleURL, articleText)},
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    extractMessages,
		MaxTokens:   1000,
		Temperature: 0.1,
		Stream:      false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return ""
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		a.config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ""
	}
	a.setAuthHeaders(httpReq)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return ""
	}

	if len(chatResp.Choices) == 0 {
		return ""
	}

	return strings.TrimSpace(chatResp.Choices[0].Message.Content)
}

// describeImages sends images to a lightweight VLM (e.g. moondream) via Ollama
// native API and returns text descriptions. This allows non-vision models to
// understand image content by converting images to text first.
func describeImages(images []string, visionModel string, ollamaEndpoint string) string {
	if len(images) == 0 || visionModel == "" {
		return ""
	}

	// Strip /v1 suffix to get Ollama native API endpoint
	baseEndpoint := strings.TrimSuffix(ollamaEndpoint, "/v1")

	var descriptions []string
	for i, img := range images {
		// Extract base64 data (remove data URI prefix if present)
		b64 := img
		if idx := strings.Index(img, ","); idx >= 0 {
			b64 = img[idx+1:]
		}

		reqBody := map[string]interface{}{
			"model": visionModel,
			"messages": []map[string]interface{}{
				{
					"role":    "user",
					"content": "この画像を詳細に説明してください。何が写っているか、色、形、テキストなど見えるものを全て記述してください。",
					"images":  []string{b64},
				},
			},
			"stream": false,
		}

		body, err := json.Marshal(reqBody)
		if err != nil {
			descriptions = append(descriptions, fmt.Sprintf("[画像%d: 変換エラー]", i+1))
			continue
		}

		client := &http.Client{Timeout: 60 * time.Second}
		resp, err := client.Post(baseEndpoint+"/api/chat", "application/json", bytes.NewReader(body))
		if err != nil {
			descriptions = append(descriptions, fmt.Sprintf("[画像%d: VLMリクエストエラー: %v]", i+1, err))
			continue
		}

		var chatResp struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		}
		err = json.NewDecoder(resp.Body).Decode(&chatResp)
		resp.Body.Close()
		if err != nil {
			descriptions = append(descriptions, fmt.Sprintf("[画像%d: レスポンス解析エラー]", i+1))
			continue
		}

		desc := strings.TrimSpace(chatResp.Message.Content)
		if desc == "" {
			descriptions = append(descriptions, fmt.Sprintf("[画像%d: 説明を取得できませんでした]", i+1))
		} else {
			descriptions = append(descriptions, fmt.Sprintf("[画像%d の内容: %s]", i+1, desc))
		}
	}

	return strings.Join(descriptions, "\n")
}

// callSubModel calls a lightweight sub-model (e.g. lfm2.5-thinking) via Ollama native /api/generate.
// It handles <think></think> tag extraction, returning (thinking, response).
// Used for fast tasks like reflection, summarization, and document tree search.
func callSubModel(prompt string, config *Config) (thinking string, response string, err error) {
	if config.SubModel == "" {
		return "", "", fmt.Errorf("no sub-model configured")
	}

	endpoint := strings.TrimSuffix(config.primaryProvider().Endpoint, "/v1")

	reqBody := map[string]interface{}{
		"model":  config.SubModel,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"num_predict": 2048,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", "", fmt.Errorf("sub-model request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", "", fmt.Errorf("sub-model decode error: %w", err)
	}

	content := genResp.Response

	// Extract <think>...</think> tags
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			thinking = strings.TrimSpace(content[ti+7 : te])
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}

	return thinking, content, nil
}

// generateCodeWithSubModel delegates code generation to the sub-model (e.g., gpt-oss:20b)
// for higher quality output. The orchestrator (1.2B) decides WHAT tool to call,
// the sub-model generates the actual code. Returns complete HTML string.
func generateCodeWithSubModel(userRequest string, config *Config) (string, error) {
	if config.SubModel == "" {
		return "", fmt.Errorf("no sub-model configured")
	}

	prompt := fmt.Sprintf(`以下のリクエストに対して、完全なHTMLページを生成せよ。

要件:
- HTMLコードのみ出力（説明不要）
- <html>, <head>, <body>タグを含む完全なHTML文書
- CSSは<style>タグ、JavaScriptは<script>タグ内に記述
- Canvas APIを使ったグラフィックス/アニメーションが必要な場合はCanvas使用
- 視覚的に美しいモダンなデザイン
- 日本語UIにすること

リクエスト: %s`, userRequest)

	endpoint := strings.TrimSuffix(config.primaryProvider().Endpoint, "/v1")

	reqBody := map[string]interface{}{
		"model":  config.SubModel,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"num_predict": 4096,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	// Longer timeout: sub-model may need to load first (20B model takes minutes on RPi)
	client := &http.Client{Timeout: 600 * time.Second}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("sub-model request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("sub-model decode error: %w", err)
	}

	content := genResp.Response

	// Strip <think>...</think> tags
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}

	// Extract HTML from response
	html := content
	if idx := strings.Index(html, "<!DOCTYPE"); idx >= 0 {
		html = html[idx:]
	} else if idx := strings.Index(html, "<html"); idx >= 0 {
		html = html[idx:]
	}
	// Find end of HTML document
	if end := strings.LastIndex(html, "</html>"); end >= 0 {
		html = html[:end+7]
	}

	// Try code fences if no HTML tags found
	if !strings.Contains(html, "<html") && !strings.Contains(html, "<!DOCTYPE") {
		if idx := strings.Index(content, "```html"); idx >= 0 {
			html = content[idx+7:]
			if end := strings.Index(html, "```"); end >= 0 {
				html = html[:end]
			}
		} else if idx := strings.Index(content, "```"); idx >= 0 {
			html = content[idx+3:]
			if nl := strings.Index(html, "\n"); nl >= 0 {
				html = html[nl+1:]
			}
			if end := strings.Index(html, "```"); end >= 0 {
				html = html[:end]
			}
		}
	}

	html = strings.TrimSpace(html)
	if html == "" || len(html) < 50 {
		return "", fmt.Errorf("sub-model generated insufficient code (%d bytes)", len(html))
	}

	// Wrap in HTML if not already
	if !strings.Contains(html, "<html") && !strings.Contains(html, "<!DOCTYPE") {
		html = "<!DOCTYPE html>\n<html><head><meta charset=\"UTF-8\"></head><body>\n" + html + "\n</body></html>"
	}

	fmt.Printf("[siki] Sub-model generated %d bytes of HTML code\n", len(html))
	return html, nil
}

func (a *Agent) webImages(targetURL string) (string, error) {
	// Fetch the HTML
	client := &http.Client{
		Timeout: 15 * time.Second,
	}
	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to fetch URL: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}
	html := string(body)

	var images []string

	// Extract OGP image (og:image)
	ogImage := extractMetaContent(html, "og:image")
	if ogImage != "" {
		images = append(images, ogImage)
	}

	// Extract Twitter card image
	twitterImage := extractMetaContent(html, "twitter:image")
	if twitterImage != "" && twitterImage != ogImage {
		images = append(images, twitterImage)
	}

	// Extract main images from img tags (large images likely to be content)
	imgURLs := extractImgSrcURLs(html, targetURL)
	for _, imgURL := range imgURLs {
		// Avoid duplicates
		isDup := false
		for _, existing := range images {
			if existing == imgURL {
				isDup = true
				break
			}
		}
		if !isDup {
			images = append(images, imgURL)
		}
		// Limit to 10 images total
		if len(images) >= 10 {
			break
		}
	}

	if len(images) == 0 {
		return "No representative images found on this page.", nil
	}

	// Return markdown image syntax
	var result strings.Builder
	result.WriteString(fmt.Sprintf("Found %d representative image(s) from %s:\n\n", len(images), targetURL))
	for i, img := range images {
		result.WriteString(fmt.Sprintf("![Image %d](%s)\n\n", i+1, img))
	}
	return result.String(), nil
}

// extractMetaContent extracts content from meta tags (og:image, twitter:image, etc.)
func extractMetaContent(html, property string) string {
	// Match patterns like:
	// <meta property="og:image" content="...">
	// <meta name="twitter:image" content="...">
	patterns := []string{
		fmt.Sprintf(`<meta[^>]*property=["']%s["'][^>]*content=["']([^"']+)["']`, property),
		fmt.Sprintf(`<meta[^>]*content=["']([^"']+)["'][^>]*property=["']%s["']`, property),
		fmt.Sprintf(`<meta[^>]*name=["']%s["'][^>]*content=["']([^"']+)["']`, property),
		fmt.Sprintf(`<meta[^>]*content=["']([^"']+)["'][^>]*name=["']%s["']`, property),
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		match := re.FindStringSubmatch(html)
		if len(match) > 1 {
			return match[1]
		}
	}
	return ""
}

// extractImgSrcURLs extracts image URLs from img tags
func extractImgSrcURLs(html, baseURL string) []string {
	var urls []string

	// Parse base URL for resolving relative URLs
	base, err := url.Parse(baseURL)
	if err != nil {
		return urls
	}

	// Match img src attributes
	re := regexp.MustCompile(`<img[^>]*src=["']([^"']+)["'][^>]*>`)
	matches := re.FindAllStringSubmatch(html, -1)

	for _, match := range matches {
		if len(match) > 1 {
			imgSrc := match[1]

			// Skip small icons, tracking pixels, base64
			if strings.Contains(imgSrc, "data:image") ||
				strings.Contains(imgSrc, "1x1") ||
				strings.Contains(imgSrc, "pixel") ||
				strings.Contains(imgSrc, "spacer") ||
				strings.Contains(imgSrc, ".gif") ||
				strings.Contains(imgSrc, "icon") ||
				strings.Contains(imgSrc, "logo") ||
				strings.Contains(imgSrc, "avatar") {
				continue
			}

			// Resolve relative URL
			imgURL, err := url.Parse(imgSrc)
			if err != nil {
				continue
			}
			resolved := base.ResolveReference(imgURL)
			urls = append(urls, resolved.String())

			// Limit to 10 candidates
			if len(urls) >= 10 {
				break
			}
		}
	}

	return urls
}

// Diagram directory for generated images
var diagramDir string

func initDiagramDir() error {
	if diagramDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	diagramDir = filepath.Join(home, ".siki", "diagrams")
	return os.MkdirAll(diagramDir, 0755)
}

func (a *Agent) generateDiagram(dotCode, title string) (string, error) {
	// Check if Graphviz is installed
	dotPath, err := exec.LookPath("dot")
	if err != nil {
		// Graphviz not installed - return helpful message
		return fmt.Sprintf(`Graphviz is not installed. To enable diagram generation:

**macOS:** brew install graphviz
**Linux:** sudo apt install graphviz
**Windows:** Download from https://graphviz.org/download/

After installing, restart Siki and try again.

Here's the DOT code for reference:
%s%s%s`, "```dot\n", dotCode, "\n```"), nil
	}

	if err := initDiagramDir(); err != nil {
		return "", fmt.Errorf("failed to create diagram dir: %w", err)
	}

	// Generate unique filename
	filename := fmt.Sprintf("diagram_%d.svg", time.Now().UnixNano())
	outputPath := filepath.Join(diagramDir, filename)

	// Create temp file for DOT code
	tmpFile, err := os.CreateTemp("", "diagram_*.dot")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(dotCode); err != nil {
		tmpFile.Close()
		return "", fmt.Errorf("failed to write DOT code: %w", err)
	}
	tmpFile.Close()

	// Run graphviz dot command
	cmd := exec.Command(dotPath, "-Tsvg", "-o", outputPath, tmpFile.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("graphviz error: %s - %w", string(output), err)
	}

	// Return markdown image syntax
	diagramURL := fmt.Sprintf("/diagrams/%s", filename)
	return fmt.Sprintf("Diagram generated:\n\n![%s](%s)", title, diagramURL), nil
}

// Playground directory for interactive code
var playgroundDir string

// Plugin system
var pluginDir string
var loadedPlugins []Plugin
var pluginMu sync.RWMutex

// Thread system
var threadDir string

// ============================================================================
// ACE Playbook System (Agentic Context Engineering)
// ============================================================================

// playbookDir stores the evolving knowledge playbook
var playbookDir string
var documentDir string

// PlaybookBullet represents a structured unit of knowledge (ACE-style)
type PlaybookBullet struct {
	ID        string `json:"id"`
	Type      string `json:"type"`    // strategy, pitfall, tool_pattern, code_snippet, preference
	Content   string `json:"content"`
	Hits      int    `json:"hits"`    // times this was useful
	Misses    int    `json:"misses"`  // times this was wrong/unhelpful
	CreatedAt int64  `json:"created_at"`
	UpdatedAt int64  `json:"updated_at"`
}

// DocumentIndex represents a hierarchical document tree (PageIndex-style)
type DocumentIndex struct {
	ID        string         `json:"id"`
	Title     string         `json:"title"`
	URL       string         `json:"url,omitempty"`
	CreatedAt int64          `json:"created_at"`
	Sections  []DocSection   `json:"sections"`
}

type DocSection struct {
	ID       string       `json:"id"`
	Title    string       `json:"title"`
	Summary  string       `json:"summary"`
	Content  string       `json:"content,omitempty"` // actual text (stored in memory during search)
	Children []DocSection `json:"children,omitempty"`
}

func initPlaybookDir() error {
	if playbookDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	playbookDir = filepath.Join(home, ".siki", "playbook")
	if err := os.MkdirAll(playbookDir, 0755); err != nil {
		return err
	}
	documentDir = filepath.Join(home, ".siki", "documents")
	return os.MkdirAll(documentDir, 0755)
}

func loadPlaybook() ([]PlaybookBullet, error) {
	if err := initPlaybookDir(); err != nil {
		return nil, err
	}
	path := filepath.Join(playbookDir, "playbook.jsonl")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var bullets []PlaybookBullet
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 256*1024), 256*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var b PlaybookBullet
		if err := json.Unmarshal([]byte(line), &b); err != nil {
			continue
		}
		bullets = append(bullets, b)
	}
	return bullets, nil
}

func savePlaybook(bullets []PlaybookBullet) error {
	if err := initPlaybookDir(); err != nil {
		return err
	}
	path := filepath.Join(playbookDir, "playbook.jsonl")
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	for _, b := range bullets {
		data, err := json.Marshal(b)
		if err != nil {
			continue
		}
		fmt.Fprintln(f, string(data))
	}
	return nil
}

func appendBullet(bullet PlaybookBullet) error {
	if err := initPlaybookDir(); err != nil {
		return err
	}
	path := filepath.Join(playbookDir, "playbook.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.Marshal(bullet)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintln(f, string(data))
	return err
}

// curateBullets merges new insights into existing playbook (ACE Curator)
// Rule-based: deduplicates by content similarity, updates hit/miss counters
func curateBullets(existing []PlaybookBullet, newBullets []PlaybookBullet) []PlaybookBullet {
	result := make([]PlaybookBullet, len(existing))
	copy(result, existing)

	for _, nb := range newBullets {
		found := false
		nbLower := strings.ToLower(nb.Content)
		for i, eb := range result {
			ebLower := strings.ToLower(eb.Content)
			// Simple dedup: if >60% of words overlap, consider it duplicate
			nbWords := strings.Fields(nbLower)
			overlap := 0
			for _, w := range nbWords {
				if strings.Contains(ebLower, w) {
					overlap++
				}
			}
			if len(nbWords) > 0 && float64(overlap)/float64(len(nbWords)) > 0.6 {
				// Update existing bullet
				result[i].Hits += nb.Hits
				result[i].UpdatedAt = time.Now().Unix()
				if len(nb.Content) > len(eb.Content) {
					result[i].Content = nb.Content // keep more detailed version
				}
				found = true
				break
			}
		}
		if !found {
			nb.CreatedAt = time.Now().Unix()
			nb.UpdatedAt = time.Now().Unix()
			result = append(result, nb)
		}
	}

	// Prune: remove bullets with too many misses relative to hits
	var pruned []PlaybookBullet
	for _, b := range result {
		if b.Misses > 3 && b.Misses > b.Hits*2 {
			continue // too unreliable
		}
		pruned = append(pruned, b)
	}

	// Limit to 100 most useful bullets
	if len(pruned) > 100 {
		// Sort by hits descending (simple bubble for small N)
		for i := 0; i < len(pruned); i++ {
			for j := i + 1; j < len(pruned); j++ {
				if pruned[j].Hits > pruned[i].Hits {
					pruned[i], pruned[j] = pruned[j], pruned[i]
				}
			}
		}
		pruned = pruned[:100]
	}

	return pruned
}

// reflectOnExecution uses the sub-model to extract insights from tool execution results
func reflectOnExecution(config *Config, toolName string, toolArgs string, toolResult string, userQuery string) []PlaybookBullet {
	if config.SubModel == "" {
		return nil
	}

	// Only reflect on meaningful tool results
	if len(toolResult) < 20 || strings.HasPrefix(toolResult, "Error") {
		return nil
	}

	// Truncate long results
	result := toolResult
	if len(result) > 2000 {
		result = result[:2000] + "..."
	}

	prompt := fmt.Sprintf(`Analyze this tool execution and extract reusable insights.

User query: %s
Tool: %s
Arguments: %s
Result: %s

Extract 0-3 reusable bullets. Each bullet should be a concrete, actionable insight.
Output JSON array only. Format:
[{"type":"strategy|pitfall|tool_pattern","content":"..."}]

If nothing is worth remembering, output: []`, userQuery, toolName, toolArgs, result)

	_, response, err := callSubModel(prompt, config)
	if err != nil {
		return nil
	}

	// Parse JSON array from response
	response = strings.TrimSpace(response)
	// Find JSON array in response
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start < 0 || end < 0 || end <= start {
		return nil
	}
	jsonStr := response[start : end+1]

	var rawBullets []struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &rawBullets); err != nil {
		return nil
	}

	var bullets []PlaybookBullet
	for i, rb := range rawBullets {
		if rb.Content == "" {
			continue
		}
		bullets = append(bullets, PlaybookBullet{
			ID:   fmt.Sprintf("%s-%d-%d", rb.Type, time.Now().Unix(), i),
			Type: rb.Type,
			Content: rb.Content,
			Hits: 1,
		})
	}
	return bullets
}

// reflectOnConversation extracts insights from conversation history before compression
func reflectOnConversation(config *Config, messages []Message) []PlaybookBullet {
	if config.SubModel == "" {
		return nil
	}

	var history strings.Builder
	for _, m := range messages {
		if m.Role == "system" {
			continue
		}
		content := m.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		if content != "" {
			history.WriteString(fmt.Sprintf("[%s]: %s\n", m.Role, content))
		}
	}

	if history.Len() < 50 {
		return nil
	}

	historyStr := history.String()
	if len(historyStr) > 6000 {
		historyStr = historyStr[:6000]
	}

	prompt := fmt.Sprintf(`Analyze this conversation and extract reusable knowledge bullets.

%s

Extract key insights as structured bullets. Categories:
- strategy: effective approaches, user preferences, system configurations
- pitfall: common errors, things to avoid, workarounds
- tool_pattern: useful tool usage patterns, command templates
- preference: user communication style, language preferences

Output JSON array only:
[{"type":"strategy|pitfall|tool_pattern|preference","content":"..."}]

Extract 0-5 bullets. If nothing worth remembering, output: []`, historyStr)

	_, response, err := callSubModel(prompt, config)
	if err != nil {
		return nil
	}

	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start < 0 || end < 0 || end <= start {
		return nil
	}

	var rawBullets []struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(response[start:end+1]), &rawBullets); err != nil {
		return nil
	}

	var bullets []PlaybookBullet
	for i, rb := range rawBullets {
		if rb.Content == "" {
			continue
		}
		bullets = append(bullets, PlaybookBullet{
			ID:   fmt.Sprintf("%s-%d-%d", rb.Type, time.Now().Unix(), i),
			Type: rb.Type,
			Content: rb.Content,
			Hits: 1,
		})
	}
	return bullets
}

// buildPlaybookContext returns formatted playbook bullets for system prompt injection
func buildPlaybookContext() string {
	bullets, err := loadPlaybook()
	if err != nil || len(bullets) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n## Memory Playbook (学習済み知識)\n")
	sb.WriteString("以下は過去の経験から学習した知識です。関連するものがあれば活用してください。\n\n")

	typeLabels := map[string]string{
		"strategy":     "戦略",
		"pitfall":      "注意点",
		"tool_pattern": "ツールパターン",
		"code_snippet": "コードスニペット",
		"preference":   "ユーザー設定",
	}

	for _, b := range bullets {
		label := typeLabels[b.Type]
		if label == "" {
			label = b.Type
		}
		sb.WriteString(fmt.Sprintf("- [%s] %s", label, b.Content))
		if b.Hits > 1 {
			sb.WriteString(fmt.Sprintf(" (確認済み×%d)", b.Hits))
		}
		sb.WriteString("\n")
	}

	return sb.String()
}

// ============================================================================
// PageIndex-style Document Tree System
// ============================================================================

func saveDocumentIndex(doc DocumentIndex) error {
	if err := initPlaybookDir(); err != nil {
		return err
	}
	docDir := filepath.Join(documentDir, doc.ID)
	if err := os.MkdirAll(docDir, 0755); err != nil {
		return err
	}

	// Save tree index
	data, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(docDir, "index.json"), data, 0644)
}

func loadDocumentIndex(docID string) (*DocumentIndex, error) {
	if err := initPlaybookDir(); err != nil {
		return nil, err
	}
	path := filepath.Join(documentDir, docID, "index.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc DocumentIndex
	if err := json.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	return &doc, nil
}

func listDocuments() ([]DocumentIndex, error) {
	if err := initPlaybookDir(); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(documentDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var docs []DocumentIndex
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		doc, err := loadDocumentIndex(e.Name())
		if err != nil {
			continue
		}
		docs = append(docs, *doc)
	}
	return docs, nil
}

// saveDocSection saves section content to a text file
func saveDocSection(docID string, sectionID string, content string) error {
	chunkDir := filepath.Join(documentDir, docID, "chunks")
	if err := os.MkdirAll(chunkDir, 0755); err != nil {
		return err
	}
	safeName := strings.ReplaceAll(sectionID, "/", "_")
	return os.WriteFile(filepath.Join(chunkDir, safeName+".txt"), []byte(content), 0644)
}

// loadDocSection loads section content from a text file
func loadDocSection(docID string, sectionID string) (string, error) {
	safeName := strings.ReplaceAll(sectionID, "/", "_")
	path := filepath.Join(documentDir, docID, "chunks", safeName+".txt")
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// indexDocument creates a hierarchical tree index from text content using the sub-model
func indexDocument(config *Config, title string, content string, sourceURL string) (*DocumentIndex, error) {
	if config.SubModel == "" {
		return nil, fmt.Errorf("sub-model not configured; required for document indexing")
	}

	docID := fmt.Sprintf("doc-%d", time.Now().UnixMilli())

	// Truncate for sub-model context
	indexContent := content
	if len(indexContent) > 12000 {
		indexContent = indexContent[:12000] + "\n...(truncated)"
	}

	prompt := fmt.Sprintf(`Analyze this document and create a hierarchical table-of-contents tree index.

Document title: %s

Content:
%s

Create a JSON tree structure with sections and subsections. For each section, provide:
- id: section number (e.g. "1", "1.1", "2")
- title: section title
- summary: 1-2 sentence summary of the section content

Output ONLY valid JSON in this format:
{"sections":[{"id":"1","title":"...","summary":"...","children":[{"id":"1.1","title":"...","summary":"..."}]}]}`, title, indexContent)

	_, response, err := callSubModel(prompt, config)
	if err != nil {
		return nil, fmt.Errorf("sub-model indexing failed: %w", err)
	}

	// Parse JSON from response
	response = strings.TrimSpace(response)
	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start < 0 || end < 0 {
		// Fallback: create a single-section index
		doc := &DocumentIndex{
			ID:        docID,
			Title:     title,
			URL:       sourceURL,
			CreatedAt: time.Now().Unix(),
			Sections: []DocSection{
				{ID: "1", Title: title, Summary: "Full document content"},
			},
		}
		saveDocSection(docID, "1", content)
		saveDocumentIndex(*doc)
		return doc, nil
	}

	var treeResp struct {
		Sections []DocSection `json:"sections"`
	}
	if err := json.Unmarshal([]byte(response[start:end+1]), &treeResp); err != nil {
		// Fallback
		doc := &DocumentIndex{
			ID:        docID,
			Title:     title,
			URL:       sourceURL,
			CreatedAt: time.Now().Unix(),
			Sections: []DocSection{
				{ID: "1", Title: title, Summary: "Full document content"},
			},
		}
		saveDocSection(docID, "1", content)
		saveDocumentIndex(*doc)
		return doc, nil
	}

	doc := &DocumentIndex{
		ID:        docID,
		Title:     title,
		URL:       sourceURL,
		CreatedAt: time.Now().Unix(),
		Sections:  treeResp.Sections,
	}

	// Split content into sections based on tree structure and save chunks
	// For now, save the full content as the root section
	saveDocSection(docID, "full", content)

	// Try to split by section headers
	lines := strings.Split(content, "\n")
	var currentSection string
	var currentContent strings.Builder
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Detect section headers (markdown style or numbered)
		isHeader := strings.HasPrefix(trimmed, "# ") || strings.HasPrefix(trimmed, "## ") ||
			strings.HasPrefix(trimmed, "### ") ||
			(len(trimmed) > 2 && trimmed[0] >= '1' && trimmed[0] <= '9' && (trimmed[1] == '.' || trimmed[1] == ' '))
		if isHeader && currentSection != "" {
			saveDocSection(docID, currentSection, currentContent.String())
			currentContent.Reset()
		}
		if isHeader {
			currentSection = strings.ReplaceAll(trimmed, " ", "_")
			if len(currentSection) > 50 {
				currentSection = currentSection[:50]
			}
		}
		currentContent.WriteString(line + "\n")
	}
	if currentSection != "" && currentContent.Len() > 0 {
		saveDocSection(docID, currentSection, currentContent.String())
	}

	saveDocumentIndex(*doc)
	return doc, nil
}

// searchDocumentTree uses the sub-model to reason through a document tree and find relevant sections
func searchDocumentTree(config *Config, doc *DocumentIndex, query string) (string, error) {
	if config.SubModel == "" {
		return "", fmt.Errorf("sub-model not configured")
	}

	// Build tree representation
	var tree strings.Builder
	var buildTree func(sections []DocSection, depth int)
	buildTree = func(sections []DocSection, depth int) {
		for _, s := range sections {
			indent := strings.Repeat("  ", depth)
			tree.WriteString(fmt.Sprintf("%s[%s] %s: %s\n", indent, s.ID, s.Title, s.Summary))
			if len(s.Children) > 0 {
				buildTree(s.Children, depth+1)
			}
		}
	}
	buildTree(doc.Sections, 0)

	prompt := fmt.Sprintf(`You are searching a document to answer a query.

Document: %s
Query: %s

Document tree:
%s

Which section IDs are most relevant to the query? List the top 1-3 section IDs.
Output ONLY a JSON array of section IDs, e.g.: ["1", "2.1"]`, doc.Title, query, tree.String())

	_, response, err := callSubModel(prompt, config)
	if err != nil {
		return "", err
	}

	// Parse section IDs
	response = strings.TrimSpace(response)
	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start < 0 || end < 0 {
		// Fallback: return full content
		content, err := loadDocSection(doc.ID, "full")
		if err != nil {
			return "No content found", nil
		}
		if len(content) > 4000 {
			content = content[:4000] + "..."
		}
		return content, nil
	}

	var sectionIDs []string
	json.Unmarshal([]byte(response[start:end+1]), &sectionIDs)

	// Load and concatenate relevant sections
	var result strings.Builder
	result.WriteString(fmt.Sprintf("## Search results from: %s\n\n", doc.Title))

	for _, sid := range sectionIDs {
		content, err := loadDocSection(doc.ID, sid)
		if err != nil {
			// Try loading full content and extracting
			continue
		}
		result.WriteString(fmt.Sprintf("### Section %s\n%s\n\n", sid, content))
	}

	// If no sections found, fallback to full
	if result.Len() < 50 {
		content, err := loadDocSection(doc.ID, "full")
		if err != nil {
			return "No content found", nil
		}
		if len(content) > 4000 {
			content = content[:4000] + "..."
		}
		return content, nil
	}

	return result.String(), nil
}

// ============================================================================
// Self-Modification System (Kernel)
// ============================================================================

// SelfState represents the mutable "self" of the AI
type SelfState struct {
	Version       int        `json:"version"`
	Prompt        string     `json:"-"` // loaded from prompt.md
	Params        SelfParams `json:"params"`
	Rules         []SelfRule `json:"rules"`
	LastModified  time.Time  `json:"last_modified"`
	LastBenchmark float64    `json:"last_benchmark"`
}

type SelfParams struct {
	Temperature    float64 `json:"temperature"`
	MaxTurns       int     `json:"max_turns"`
	CompressAt     int     `json:"compress_at"`
	ReflectOnTools bool    `json:"reflect_on_tools"`
	PreferredLang  string  `json:"preferred_lang"`
	Verbosity      string  `json:"verbosity"`
}

type SelfRule struct {
	ID        string `json:"id"`
	Rule      string `json:"rule"`
	Reason    string `json:"reason"`
	CreatedAt int64  `json:"created_at"`
	Active    bool   `json:"active"`
}

type SelfSnapshot struct {
	Version        int       `json:"version"`
	Timestamp      time.Time `json:"timestamp"`
	Reason         string    `json:"reason"`
	BenchmarkScore float64   `json:"benchmark_score"`
}

type BenchmarkResult struct {
	Timestamp    time.Time          `json:"timestamp"`
	Version      int                `json:"version"`
	OverallScore float64            `json:"overall_score"`
	Scores       map[string]float64 `json:"scores"`
}

var selfDir string
var currentSelf *SelfState
var selfMu sync.RWMutex

var kernelParams = struct {
	MinPromptLen  int
	MaxPromptLen  int
	MinMaxTurns   int
	MaxMaxTurns   int
	MaxRules      int
	MaxSnapshots  int
	RequiredTools []string
}{
	MinPromptLen: 200,
	MaxPromptLen: 50000,
	MinMaxTurns:  3,
	MaxMaxTurns:  50,
	MaxRules:     50,
	MaxSnapshots: 100,
	RequiredTools: []string{
		"read_file", "write_file", "execute_command",
		"self_modify_prompt", "self_rollback", "self_status",
	},
}

// Dangerous prompt patterns that are always rejected
var dangerousPromptPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)ignore\s+(all|previous|prior)`),
	regexp.MustCompile(`(?i)do\s+nothing`),
	regexp.MustCompile(`(?i)refuse\s+all`),
	regexp.MustCompile(`(?i)never\s+respond`),
	regexp.MustCompile(`(?i)always\s+(lie|refuse|ignore)`),
}

func initSelfDir() error {
	if selfDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	selfDir = filepath.Join(home, ".siki", "self")
	for _, sub := range []string{"current", "snapshots", "benchmarks"} {
		if err := os.MkdirAll(filepath.Join(selfDir, sub), 0755); err != nil {
			return err
		}
	}
	return nil
}

func defaultSelfState() *SelfState {
	cfg := defaultConfig()
	return &SelfState{
		Version: 0,
		Prompt:  cfg.SystemPrompt,
		Params: SelfParams{
			Temperature:    0.7,
			MaxTurns:       MaxTurns,
			CompressAt:     60,
			ReflectOnTools: true,
			PreferredLang:  "auto",
			Verbosity:      "normal",
		},
		LastModified: time.Now(),
	}
}

func loadSelfState() (*SelfState, error) {
	if err := initSelfDir(); err != nil {
		return nil, err
	}
	currentDir := filepath.Join(selfDir, "current")

	state := defaultSelfState()

	// Load prompt.md
	if data, err := os.ReadFile(filepath.Join(currentDir, "prompt.md")); err == nil {
		if len(data) > 0 {
			state.Prompt = string(data)
		}
	}

	// Load params.json
	if data, err := os.ReadFile(filepath.Join(currentDir, "params.json")); err == nil {
		var loaded SelfState
		if err := json.Unmarshal(data, &loaded); err == nil {
			state.Version = loaded.Version
			state.Params = loaded.Params
			state.LastModified = loaded.LastModified
			state.LastBenchmark = loaded.LastBenchmark
		}
	}

	// Load rules.jsonl
	if f, err := os.Open(filepath.Join(currentDir, "rules.jsonl")); err == nil {
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rule SelfRule
			if err := json.Unmarshal([]byte(line), &rule); err == nil {
				state.Rules = append(state.Rules, rule)
			}
		}
		f.Close()
	}

	return state, nil
}

func saveSelfState(state *SelfState) error {
	if err := initSelfDir(); err != nil {
		return err
	}
	currentDir := filepath.Join(selfDir, "current")

	// Save prompt.md
	if err := os.WriteFile(filepath.Join(currentDir, "prompt.md"), []byte(state.Prompt), 0644); err != nil {
		return err
	}

	// Save params.json (includes version, params, timestamps, benchmark)
	paramsData, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(currentDir, "params.json"), paramsData, 0644); err != nil {
		return err
	}

	// Save rules.jsonl
	f, err := os.Create(filepath.Join(currentDir, "rules.jsonl"))
	if err != nil {
		return err
	}
	defer f.Close()
	for _, rule := range state.Rules {
		data, _ := json.Marshal(rule)
		fmt.Fprintln(f, string(data))
	}

	return nil
}

func validateSelfState(state *SelfState) error {
	if len(state.Prompt) < kernelParams.MinPromptLen {
		return fmt.Errorf("prompt too short (%d chars, minimum %d)", len(state.Prompt), kernelParams.MinPromptLen)
	}
	if len(state.Prompt) > kernelParams.MaxPromptLen {
		return fmt.Errorf("prompt too long (%d chars, maximum %d)", len(state.Prompt), kernelParams.MaxPromptLen)
	}
	// Check required tools are mentioned
	promptLower := strings.ToLower(state.Prompt)
	for _, tool := range kernelParams.RequiredTools {
		if !strings.Contains(promptLower, tool) {
			return fmt.Errorf("prompt must mention required tool: %s", tool)
		}
	}
	// Check for dangerous patterns
	for _, pat := range dangerousPromptPatterns {
		if pat.MatchString(state.Prompt) {
			return fmt.Errorf("prompt contains dangerous pattern: %s", pat.String())
		}
	}
	// Validate params bounds
	if state.Params.Temperature < 0 || state.Params.Temperature > 2.0 {
		return fmt.Errorf("temperature must be 0.0-2.0, got %f", state.Params.Temperature)
	}
	if state.Params.MaxTurns < kernelParams.MinMaxTurns || state.Params.MaxTurns > kernelParams.MaxMaxTurns {
		return fmt.Errorf("max_turns must be %d-%d, got %d", kernelParams.MinMaxTurns, kernelParams.MaxMaxTurns, state.Params.MaxTurns)
	}
	if state.Params.CompressAt < 20 || state.Params.CompressAt > 100 {
		return fmt.Errorf("compress_at must be 20-100, got %d", state.Params.CompressAt)
	}
	if len(state.Rules) > kernelParams.MaxRules {
		return fmt.Errorf("too many rules (%d, maximum %d)", len(state.Rules), kernelParams.MaxRules)
	}
	return nil
}

// ---- Version Control / Snapshots ----

func createSnapshot(state *SelfState, reason string) (*SelfSnapshot, error) {
	if err := initSelfDir(); err != nil {
		return nil, err
	}
	state.Version++
	ts := time.Now()
	dirName := fmt.Sprintf("v%03d_%s", state.Version, ts.Format("20060102T150405"))
	snapDir := filepath.Join(selfDir, "snapshots", dirName)
	if err := os.MkdirAll(snapDir, 0755); err != nil {
		return nil, err
	}

	// Copy files
	os.WriteFile(filepath.Join(snapDir, "prompt.md"), []byte(state.Prompt), 0644)
	paramsData, _ := json.MarshalIndent(state, "", "  ")
	os.WriteFile(filepath.Join(snapDir, "params.json"), paramsData, 0644)

	rulesF, _ := os.Create(filepath.Join(snapDir, "rules.jsonl"))
	for _, r := range state.Rules {
		d, _ := json.Marshal(r)
		fmt.Fprintln(rulesF, string(d))
	}
	rulesF.Close()

	snap := &SelfSnapshot{
		Version:        state.Version,
		Timestamp:      ts,
		Reason:         reason,
		BenchmarkScore: state.LastBenchmark,
	}
	metaData, _ := json.MarshalIndent(snap, "", "  ")
	os.WriteFile(filepath.Join(snapDir, "meta.json"), metaData, 0644)

	// Prune old snapshots
	pruneSnapshots()

	return snap, nil
}

func listSnapshots() ([]SelfSnapshot, error) {
	if err := initSelfDir(); err != nil {
		return nil, err
	}
	snapDir := filepath.Join(selfDir, "snapshots")
	entries, err := os.ReadDir(snapDir)
	if err != nil {
		return nil, nil
	}

	var snapshots []SelfSnapshot
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		metaPath := filepath.Join(snapDir, e.Name(), "meta.json")
		data, err := os.ReadFile(metaPath)
		if err != nil {
			continue
		}
		var snap SelfSnapshot
		if err := json.Unmarshal(data, &snap); err == nil {
			snapshots = append(snapshots, snap)
		}
	}
	return snapshots, nil
}

func rollbackToSnapshot(version int) (*SelfState, error) {
	if err := initSelfDir(); err != nil {
		return nil, err
	}
	snapDir := filepath.Join(selfDir, "snapshots")
	entries, err := os.ReadDir(snapDir)
	if err != nil {
		return nil, fmt.Errorf("no snapshots found")
	}

	prefix := fmt.Sprintf("v%03d_", version)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), prefix) {
			srcDir := filepath.Join(snapDir, e.Name())
			currentDir := filepath.Join(selfDir, "current")

			// Copy files from snapshot to current
			for _, fname := range []string{"prompt.md", "params.json", "rules.jsonl"} {
				data, err := os.ReadFile(filepath.Join(srcDir, fname))
				if err != nil {
					continue
				}
				os.WriteFile(filepath.Join(currentDir, fname), data, 0644)
			}

			// Reload state
			state, err := loadSelfState()
			if err != nil {
				return nil, err
			}
			return state, nil
		}
	}

	// Version 0 = hardcoded default
	if version == 0 {
		state := defaultSelfState()
		if err := saveSelfState(state); err != nil {
			return nil, err
		}
		return state, nil
	}

	return nil, fmt.Errorf("snapshot version %d not found", version)
}

func pruneSnapshots() {
	snapDir := filepath.Join(selfDir, "snapshots")
	entries, err := os.ReadDir(snapDir)
	if err != nil {
		return
	}
	if len(entries) <= kernelParams.MaxSnapshots {
		return
	}
	// Remove oldest entries beyond limit, but keep first one
	toRemove := len(entries) - kernelParams.MaxSnapshots
	removed := 0
	for i := 1; i < len(entries)-10 && removed < toRemove; i++ {
		os.RemoveAll(filepath.Join(snapDir, entries[i].Name()))
		removed++
	}
}

func saveBenchmarkResult(result BenchmarkResult) error {
	if err := initSelfDir(); err != nil {
		return err
	}
	path := filepath.Join(selfDir, "benchmarks", "results.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	data, _ := json.Marshal(result)
	_, err = fmt.Fprintln(f, string(data))
	return err
}

// ---- Benchmark System ----

type benchmarkCase struct {
	prompt   string
	expected string
	scorer   func(response, expected string) float64
}

func containsScore(response, expected string) float64 {
	if strings.Contains(strings.ToLower(response), strings.ToLower(expected)) {
		return 1.0
	}
	return 0.0
}

func semanticScore(response, expected string) float64 {
	expWords := strings.Fields(strings.ToLower(expected))
	if len(expWords) == 0 {
		return 0.5
	}
	respLower := strings.ToLower(response)
	overlap := 0
	for _, w := range expWords {
		if strings.Contains(respLower, w) {
			overlap++
		}
	}
	return float64(overlap) / float64(len(expWords))
}

func numericScore(response, _ string) float64 {
	response = strings.TrimSpace(response)
	var n float64
	fmt.Sscanf(response, "%f", &n)
	if n < 1 {
		n = 1
	}
	if n > 10 {
		n = 10
	}
	return n / 10.0
}

var benchmarkSuites = map[string][]benchmarkCase{
	"tool_use": {
		{prompt: "User says: 'show files in /tmp'. What tool should be called? Answer with just the tool name.", expected: "list_files", scorer: containsScore},
		{prompt: "User says: 'search for TODO in code'. What tool should be called? Answer with just the tool name.", expected: "grep", scorer: containsScore},
		{prompt: "User says: 'what happened in the news?'. What tool should be called? Answer with just the tool name.", expected: "web_search", scorer: containsScore},
	},
	"reasoning": {
		{prompt: "If a user uploads an image and says 'これ何?', the AI should use which tool first? Answer the tool name.", expected: "describe", scorer: semanticScore},
		{prompt: "A user says 'もっと詳しく'. Should the AI (a) elaborate on the previous topic or (b) ask what they mean? Answer a or b.", expected: "a", scorer: containsScore},
	},
	"language": {
		{prompt: "User says 'hello'. Should you respond in English or Japanese? Answer the language.", expected: "English", scorer: containsScore},
		{prompt: "User says 'こんにちは'. Should you respond in English or Japanese? Answer the language.", expected: "Japanese", scorer: containsScore},
	},
	"helpfulness": {
		{prompt: "Rate this instruction on clarity from 1-10: 'You are a helpful AI assistant with access to tools.' Just the number.", expected: "", scorer: numericScore},
	},
}

// ---- Self-Modification Tool Handlers ----

func replaceSectionInPrompt(prompt, sectionHeader, newContent string) string {
	startIdx := strings.Index(prompt, sectionHeader)
	if startIdx == -1 {
		return prompt + "\n\n" + sectionHeader + "\n" + newContent
	}
	afterStart := startIdx + len(sectionHeader)
	remaining := prompt[afterStart:]
	nextSection := strings.Index(remaining, "\n## ")
	if nextSection == -1 {
		return prompt[:startIdx] + sectionHeader + "\n" + newContent
	}
	return prompt[:startIdx] + sectionHeader + "\n" + newContent + remaining[nextSection:]
}

func (a *Agent) selfModifyPrompt(args map[string]interface{}) (string, error) {
	action, _ := args["action"].(string)
	content, _ := args["content"].(string)
	reason, _ := args["reason"].(string)
	section, _ := args["section"].(string)

	if content == "" || reason == "" {
		return "", fmt.Errorf("content and reason are required")
	}

	selfMu.Lock()
	defer selfMu.Unlock()

	// Snapshot before modification
	createSnapshot(currentSelf, "pre-modify: "+reason)
	oldPrompt := currentSelf.Prompt

	switch action {
	case "replace":
		currentSelf.Prompt = content
	case "append":
		currentSelf.Prompt += "\n\n" + content
	case "replace_section":
		if section == "" {
			currentSelf.Prompt = oldPrompt
			return "", fmt.Errorf("section header required for replace_section")
		}
		currentSelf.Prompt = replaceSectionInPrompt(currentSelf.Prompt, section, content)
	default:
		return "", fmt.Errorf("invalid action: %s (use replace, append, or replace_section)", action)
	}

	// Validate
	if err := validateSelfState(currentSelf); err != nil {
		currentSelf.Prompt = oldPrompt
		return "", fmt.Errorf("modification rejected: %w", err)
	}

	currentSelf.LastModified = time.Now()
	saveSelfState(currentSelf)
	snap, _ := createSnapshot(currentSelf, reason)

	return fmt.Sprintf("System prompt modified (action: %s). Version: v%d.\nReason: %s\nChanges take effect on new conversations. Use self_benchmark to evaluate.", action, snap.Version, reason), nil
}

func (a *Agent) selfModifyParams(args map[string]interface{}) (string, error) {
	paramsStr, _ := args["params"].(string)
	reason, _ := args["reason"].(string)

	if paramsStr == "" || reason == "" {
		return "", fmt.Errorf("params (JSON) and reason are required")
	}

	var updates map[string]interface{}
	if err := json.Unmarshal([]byte(paramsStr), &updates); err != nil {
		return "", fmt.Errorf("invalid params JSON: %w", err)
	}

	selfMu.Lock()
	defer selfMu.Unlock()

	createSnapshot(currentSelf, "pre-params: "+reason)

	if v, ok := updates["temperature"].(float64); ok {
		currentSelf.Params.Temperature = v
	}
	if v, ok := updates["max_turns"].(float64); ok {
		currentSelf.Params.MaxTurns = int(v)
	}
	if v, ok := updates["compress_at"].(float64); ok {
		currentSelf.Params.CompressAt = int(v)
	}
	if v, ok := updates["reflect_on_tools"].(bool); ok {
		currentSelf.Params.ReflectOnTools = v
	}
	if v, ok := updates["preferred_lang"].(string); ok {
		currentSelf.Params.PreferredLang = v
	}
	if v, ok := updates["verbosity"].(string); ok {
		currentSelf.Params.Verbosity = v
	}

	if err := validateSelfState(currentSelf); err != nil {
		// Reload from disk
		restored, _ := loadSelfState()
		if restored != nil {
			*currentSelf = *restored
		}
		return "", fmt.Errorf("params rejected: %w", err)
	}

	currentSelf.LastModified = time.Now()
	saveSelfState(currentSelf)
	snap, _ := createSnapshot(currentSelf, reason)

	data, _ := json.MarshalIndent(currentSelf.Params, "", "  ")
	return fmt.Sprintf("Parameters updated. Version: v%d.\n%s", snap.Version, string(data)), nil
}

func (a *Agent) selfAddRule(args map[string]interface{}) (string, error) {
	ruleText, _ := args["rule"].(string)
	reason, _ := args["reason"].(string)
	if ruleText == "" || reason == "" {
		return "", fmt.Errorf("rule and reason are required")
	}

	selfMu.Lock()
	defer selfMu.Unlock()

	rule := SelfRule{
		ID:        fmt.Sprintf("rule-%d", time.Now().UnixMilli()),
		Rule:      ruleText,
		Reason:    reason,
		CreatedAt: time.Now().Unix(),
		Active:    true,
	}
	currentSelf.Rules = append(currentSelf.Rules, rule)

	if err := validateSelfState(currentSelf); err != nil {
		currentSelf.Rules = currentSelf.Rules[:len(currentSelf.Rules)-1]
		return "", fmt.Errorf("rule rejected: %w", err)
	}

	saveSelfState(currentSelf)
	return fmt.Sprintf("Rule added: [%s] %s\nReason: %s", rule.ID, rule.Rule, rule.Reason), nil
}

func (a *Agent) selfRemoveRule(args map[string]interface{}) (string, error) {
	ruleID, _ := args["rule_id"].(string)
	if ruleID == "" {
		return "", fmt.Errorf("rule_id is required")
	}

	selfMu.Lock()
	defer selfMu.Unlock()

	found := false
	for i := range currentSelf.Rules {
		if currentSelf.Rules[i].ID == ruleID {
			currentSelf.Rules[i].Active = false
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("rule not found: %s", ruleID)
	}

	saveSelfState(currentSelf)
	return fmt.Sprintf("Rule %s deactivated.", ruleID), nil
}

func (a *Agent) selfStatus() (string, error) {
	selfMu.RLock()
	defer selfMu.RUnlock()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Self Status (v%d)\n\n", currentSelf.Version))
	sb.WriteString(fmt.Sprintf("Last modified: %s\n", currentSelf.LastModified.Format("2006-01-02 15:04:05")))
	sb.WriteString(fmt.Sprintf("Last benchmark: %.1f%%\n\n", currentSelf.LastBenchmark*100))

	sb.WriteString("### Parameters\n")
	data, _ := json.MarshalIndent(currentSelf.Params, "", "  ")
	sb.WriteString("```json\n" + string(data) + "\n```\n\n")

	sb.WriteString(fmt.Sprintf("### System Prompt (%d chars)\n", len(currentSelf.Prompt)))
	promptPreview := currentSelf.Prompt
	if len(promptPreview) > 500 {
		promptPreview = promptPreview[:500] + "..."
	}
	sb.WriteString("```\n" + promptPreview + "\n```\n\n")

	// Active rules
	var activeRules []SelfRule
	for _, r := range currentSelf.Rules {
		if r.Active {
			activeRules = append(activeRules, r)
		}
	}
	sb.WriteString(fmt.Sprintf("### Self-Rules (%d active)\n", len(activeRules)))
	for _, r := range activeRules {
		sb.WriteString(fmt.Sprintf("- [%s] %s (%s)\n", r.ID, r.Rule, r.Reason))
	}

	// Snapshots
	snapshots, _ := listSnapshots()
	sb.WriteString(fmt.Sprintf("\n### Version History (%d snapshots)\n", len(snapshots)))
	// Show last 10
	start := 0
	if len(snapshots) > 10 {
		start = len(snapshots) - 10
		sb.WriteString(fmt.Sprintf("... (%d older snapshots)\n", start))
	}
	for i := start; i < len(snapshots); i++ {
		s := snapshots[i]
		score := ""
		if s.BenchmarkScore > 0 {
			score = fmt.Sprintf(" [%.1f%%]", s.BenchmarkScore*100)
		}
		sb.WriteString(fmt.Sprintf("- v%d: %s - %s%s\n", s.Version, s.Timestamp.Format("01/02 15:04"), s.Reason, score))
	}

	return sb.String(), nil
}

func (a *Agent) selfRollback(args map[string]interface{}) (string, error) {
	versionF, _ := args["version"].(float64)
	version := int(versionF)

	selfMu.Lock()
	defer selfMu.Unlock()

	oldVersion := currentSelf.Version
	state, err := rollbackToSnapshot(version)
	if err != nil {
		return "", err
	}
	currentSelf = state

	return fmt.Sprintf("Rolled back from v%d to v%d. Changes take effect on new conversations.", oldVersion, version), nil
}

func (a *Agent) selfBenchmark(args map[string]interface{}) (string, error) {
	categories := "all"
	if c, ok := args["categories"].(string); ok && c != "" {
		categories = c
	}

	selfMu.RLock()
	promptText := currentSelf.Prompt
	version := currentSelf.Version
	selfMu.RUnlock()

	suitesToRun := make(map[string][]benchmarkCase)
	if categories == "all" {
		for k, v := range benchmarkSuites {
			suitesToRun[k] = v
		}
	} else {
		for _, cat := range strings.Split(categories, ",") {
			cat = strings.TrimSpace(cat)
			if suite, ok := benchmarkSuites[cat]; ok {
				suitesToRun[cat] = suite
			}
		}
	}

	if len(suitesToRun) == 0 {
		return "", fmt.Errorf("no valid benchmark categories found")
	}

	// Truncate prompt for evaluation
	evalPrompt := promptText
	if len(evalPrompt) > 4000 {
		evalPrompt = evalPrompt[:4000] + "..."
	}

	scores := make(map[string]float64)
	var totalScore float64

	for catName, cases := range suitesToRun {
		var catScore float64
		for _, bc := range cases {
			prompt := fmt.Sprintf("Given this AI system prompt:\n---\n%s\n---\n\n%s", evalPrompt, bc.prompt)
			_, response, err := callSubModel(prompt, a.config)
			if err != nil {
				continue
			}
			score := bc.scorer(response, bc.expected)
			catScore += score
		}
		if len(cases) > 0 {
			scores[catName] = catScore / float64(len(cases))
			totalScore += scores[catName]
		}
	}

	overall := 0.0
	if len(scores) > 0 {
		overall = totalScore / float64(len(scores))
	}

	result := BenchmarkResult{
		Timestamp:    time.Now(),
		Version:      version,
		OverallScore: overall,
		Scores:       scores,
	}
	saveBenchmarkResult(result)

	selfMu.Lock()
	currentSelf.LastBenchmark = overall
	saveSelfState(currentSelf)
	selfMu.Unlock()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("## Benchmark Results (v%d)\n\n", version))
	sb.WriteString(fmt.Sprintf("**Overall: %.1f%%**\n\n", overall*100))
	for cat, score := range scores {
		sb.WriteString(fmt.Sprintf("- %s: %.1f%%\n", cat, score*100))
	}
	return sb.String(), nil
}

// ---- Self-Improvement Loop ----

func (ws *WebServer) selfImproveLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ws.mu.RLock()
		idle := time.Since(ws.lastActivity) > 5*time.Minute
		ws.mu.RUnlock()

		if !idle {
			continue
		}

		if !ws.improveMu.TryLock() {
			continue
		}

		fmt.Println("[siki] Self-improvement loop: analyzing recent conversations...")
		ws.runSelfImprovement()
		ws.improveMu.Unlock()
	}
}

func (ws *WebServer) runSelfImprovement() {
	threads, err := listThreads()
	if err != nil || len(threads) == 0 {
		return
	}

	// Look at threads from last 24 hours
	cutoff := time.Now().Add(-24 * time.Hour)
	var recentCount int
	var threadSummaries strings.Builder
	for _, t := range threads {
		if t.UpdatedAt.After(cutoff) {
			recentCount++
			threadSummaries.WriteString(fmt.Sprintf("- '%s' (%d messages)\n", t.Title, t.MessageCount))
		}
	}

	if recentCount == 0 {
		return
	}

	selfMu.RLock()
	lastBench := currentSelf.LastBenchmark
	version := currentSelf.Version
	selfMu.RUnlock()

	bullets, _ := loadPlaybook()

	prompt := fmt.Sprintf(`Analyze recent siki usage and suggest improvements.

Recent threads (last 24h):
%s

Current state: v%d, benchmark: %.1f%%, playbook bullets: %d

Suggest 0-3 specific improvements. Categories:
- new_rule: add a behavioral rule
- param_change: adjust parameters
- prompt_change: modify system prompt

Output JSON array only:
[{"category":"new_rule|param_change|prompt_change","description":"...","expected_impact":"..."}]
If no improvements needed: []`, threadSummaries.String(), version, lastBench*100, len(bullets))

	_, response, err := callSubModel(prompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] Self-improvement analysis failed: %v\n", err)
		return
	}

	start := strings.Index(response, "[")
	end := strings.LastIndex(response, "]")
	if start < 0 || end <= start {
		return
	}

	var suggestions []struct {
		Category       string `json:"category"`
		Description    string `json:"description"`
		ExpectedImpact string `json:"expected_impact"`
	}
	if err := json.Unmarshal([]byte(response[start:end+1]), &suggestions); err != nil {
		return
	}

	if len(suggestions) == 0 {
		fmt.Println("[siki] Self-improvement: no improvements suggested")
		return
	}

	for _, s := range suggestions {
		switch s.Category {
		case "new_rule":
			selfMu.Lock()
			rule := SelfRule{
				ID:        fmt.Sprintf("auto-%d", time.Now().UnixMilli()),
				Rule:      s.Description,
				Reason:    "auto: " + s.ExpectedImpact,
				CreatedAt: time.Now().Unix(),
				Active:    true,
			}
			currentSelf.Rules = append(currentSelf.Rules, rule)
			if err := validateSelfState(currentSelf); err != nil {
				currentSelf.Rules = currentSelf.Rules[:len(currentSelf.Rules)-1]
				selfMu.Unlock()
				continue
			}
			saveSelfState(currentSelf)
			selfMu.Unlock()
			fmt.Printf("[siki] Self-improvement: added rule '%s'\n", s.Description)
		default:
			fmt.Printf("[siki] Self-improvement: suggestion logged (%s): %s\n", s.Category, s.Description)
		}
	}

	fmt.Println("[siki] Self-improvement cycle complete")
}

// ============================================================================
// Self-Evolution System (Code-Level Kernel Modification)
// ============================================================================

// EvolveState tracks an in-progress code evolution
type EvolveState struct {
	SourceDir   string        `json:"source_dir"`
	Patches     []EvolvePatch `json:"patches"`
	BuildStatus string        `json:"build_status"` // "", "success", "failed"
	TestStatus  string        `json:"test_status"`  // "", "success", "failed"
	BuildOutput string        `json:"build_output"`
	TestOutput  string        `json:"test_output"`
	TempBinary  string        `json:"temp_binary"`
	OrigSource  string        `json:"-"` // backup of main.go before patches
	StartedAt   time.Time     `json:"started_at"`
}

type EvolvePatch struct {
	Description string `json:"description"`
	OldText     string `json:"old_text"`
	NewText     string `json:"new_text"`
	Applied     bool   `json:"applied"`
}

type EvolveHistory struct {
	Timestamp   time.Time `json:"timestamp"`
	GitHash     string    `json:"git_hash"`
	PatchCount  int       `json:"patch_count"`
	TestsPassed bool      `json:"tests_passed"`
	Reason      string    `json:"reason"`
	Success     bool      `json:"success"`
}

var evolveState *EvolveState
var evolveMu sync.Mutex

// detectSourceDir finds the siki source directory by looking for main.go + go.mod
func detectSourceDir() string {
	candidates := []string{"/mnt/siki"}
	if exe, err := os.Executable(); err == nil {
		candidates = append([]string{filepath.Dir(exe)}, candidates...)
	}
	for _, dir := range candidates {
		mainGo := filepath.Join(dir, "main.go")
		goMod := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(mainGo); err == nil {
			if _, err := os.Stat(goMod); err == nil {
				return dir
			}
		}
	}
	return ""
}

func (a *Agent) selfEvolve(args map[string]interface{}) (string, error) {
	action, _ := args["action"].(string)
	switch action {
	case "view_source":
		return a.evolveViewSource(args)
	case "patch":
		return a.evolvePatch(args)
	case "build_test":
		return a.evolveBuildTest()
	case "deploy":
		return a.evolveDeploy(args)
	case "status":
		return a.evolveStatus()
	case "abort":
		return a.evolveAbort()
	default:
		return "", fmt.Errorf("unknown action: %s (valid: view_source, patch, build_test, deploy, status, abort)", action)
	}
}

func (a *Agent) evolveViewSource(args map[string]interface{}) (string, error) {
	sourceDir := detectSourceDir()
	if sourceDir == "" {
		return "", fmt.Errorf("source directory not found (need main.go + go.mod)")
	}

	startLine := 1
	endLine := 50
	if s, ok := args["start_line"].(float64); ok {
		startLine = int(s)
	}
	if e, ok := args["end_line"].(float64); ok {
		endLine = int(e)
	}

	data, err := os.ReadFile(filepath.Join(sourceDir, "main.go"))
	if err != nil {
		return "", err
	}

	lines := strings.Split(string(data), "\n")
	if startLine < 1 {
		startLine = 1
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	if endLine-startLine > 200 {
		endLine = startLine + 200
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("Source: %s/main.go (lines %d-%d of %d)\n\n", sourceDir, startLine, endLine, len(lines)))
	for i := startLine - 1; i < endLine && i < len(lines); i++ {
		result.WriteString(fmt.Sprintf("%5d | %s\n", i+1, lines[i]))
	}
	return result.String(), nil
}

func (a *Agent) evolvePatch(args map[string]interface{}) (string, error) {
	evolveMu.Lock()
	defer evolveMu.Unlock()

	sourceDir := detectSourceDir()
	if sourceDir == "" {
		return "", fmt.Errorf("source directory not found")
	}

	oldText, _ := args["old_text"].(string)
	newText, _ := args["new_text"].(string)
	description, _ := args["description"].(string)

	if oldText == "" || newText == "" {
		return "", fmt.Errorf("old_text and new_text are required")
	}
	if description == "" {
		description = "unnamed patch"
	}

	// Initialize evolve state on first patch
	if evolveState == nil {
		data, err := os.ReadFile(filepath.Join(sourceDir, "main.go"))
		if err != nil {
			return "", err
		}
		evolveState = &EvolveState{
			SourceDir:  sourceDir,
			OrigSource: string(data),
			StartedAt:  time.Now(),
		}
	}

	// Read current source (may already have prior patches applied)
	data, err := os.ReadFile(filepath.Join(sourceDir, "main.go"))
	if err != nil {
		return "", err
	}
	source := string(data)

	if !strings.Contains(source, oldText) {
		return "", fmt.Errorf("old_text not found in source. Make sure it matches exactly (including whitespace)")
	}
	if strings.Count(source, oldText) > 1 {
		return "", fmt.Errorf("old_text appears %d times. Provide more surrounding context to make it unique", strings.Count(source, oldText))
	}

	// Apply patch
	newSource := strings.Replace(source, oldText, newText, 1)
	if err := os.WriteFile(filepath.Join(sourceDir, "main.go"), []byte(newSource), 0644); err != nil {
		return "", err
	}

	patch := EvolvePatch{
		Description: description,
		OldText:     oldText,
		NewText:     newText,
		Applied:     true,
	}
	evolveState.Patches = append(evolveState.Patches, patch)
	evolveState.BuildStatus = ""
	evolveState.TestStatus = ""

	return fmt.Sprintf("Patch #%d applied: %s\nTotal patches staged: %d\nNext: use action=build_test to compile and run tests.", len(evolveState.Patches), description, len(evolveState.Patches)), nil
}

func (a *Agent) evolveBuildTest() (string, error) {
	evolveMu.Lock()
	defer evolveMu.Unlock()

	if evolveState == nil || len(evolveState.Patches) == 0 {
		return "", fmt.Errorf("no patches staged. Use action=patch first")
	}

	sourceDir := evolveState.SourceDir
	goPath := "/usr/local/go/bin/go"
	if _, err := os.Stat(goPath); err != nil {
		// Fallback
		if p, err := exec.LookPath("go"); err == nil {
			goPath = p
		} else {
			return "", fmt.Errorf("Go compiler not found")
		}
	}

	var result strings.Builder

	// Build
	result.WriteString("=== Building ===\n")
	tempBinary := filepath.Join(os.TempDir(), fmt.Sprintf("siki-evolve-%d", time.Now().UnixMilli()))

	cmd := exec.Command(goPath, "build", "-o", tempBinary, ".")
	cmd.Dir = sourceDir
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	buildOut, err := cmd.CombinedOutput()
	evolveState.BuildOutput = string(buildOut)

	if err != nil {
		evolveState.BuildStatus = "failed"
		result.WriteString(fmt.Sprintf("BUILD FAILED:\n%s\n", string(buildOut)))
		result.WriteString("\nFix the issue with another patch, or use action=abort to restore original source.")
		return result.String(), nil
	}

	evolveState.BuildStatus = "success"
	evolveState.TempBinary = tempBinary

	// Get binary size
	if info, err := os.Stat(tempBinary); err == nil {
		result.WriteString(fmt.Sprintf("BUILD OK (%d bytes)\n\n", info.Size()))
	} else {
		result.WriteString("BUILD OK\n\n")
	}

	// Test
	result.WriteString("=== Testing ===\n")
	cmd = exec.Command(goPath, "test", "-count=1", "-timeout=180s", "./...")
	cmd.Dir = sourceDir
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	testOut, err := cmd.CombinedOutput()
	evolveState.TestOutput = string(testOut)

	if err != nil {
		evolveState.TestStatus = "failed"
		// Show last 30 lines of test output
		testLines := strings.Split(string(testOut), "\n")
		start := 0
		if len(testLines) > 30 {
			start = len(testLines) - 30
		}
		for i := start; i < len(testLines); i++ {
			result.WriteString(testLines[i] + "\n")
		}
		result.WriteString("\nTESTS FAILED. Fix and retry, or use action=abort to restore.")
		return result.String(), nil
	}

	evolveState.TestStatus = "success"

	// Show summary lines from test output
	testLines := strings.Split(string(testOut), "\n")
	for _, line := range testLines {
		if strings.Contains(line, "PASS") || strings.Contains(line, "ok ") || strings.Contains(line, "---") {
			result.WriteString(line + "\n")
		}
	}

	result.WriteString(fmt.Sprintf("\nALL TESTS PASSED\nNew binary: %s\n", tempBinary))
	result.WriteString("\nReady to deploy. Use action=deploy with description to deploy and restart the process.")

	return result.String(), nil
}

func (a *Agent) evolveDeploy(args map[string]interface{}) (string, error) {
	evolveMu.Lock()
	defer evolveMu.Unlock()

	if evolveState == nil {
		return "", fmt.Errorf("no evolution in progress")
	}
	if evolveState.BuildStatus != "success" || evolveState.TestStatus != "success" {
		return "", fmt.Errorf("build and tests must both pass before deploying (build=%s, test=%s)", evolveState.BuildStatus, evolveState.TestStatus)
	}

	reason, _ := args["description"].(string)
	if reason == "" {
		reason = "self-evolution"
	}

	// Get current executable path
	currentExe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("cannot determine current executable: %w", err)
	}
	currentExe, _ = filepath.EvalSymlinks(currentExe)

	// Backup current binary
	backupPath := currentExe + ".pre-evolve"
	if data, err := os.ReadFile(currentExe); err == nil {
		if err := os.WriteFile(backupPath, data, 0755); err != nil {
			fmt.Printf("[siki] Warning: failed to backup binary: %v\n", err)
		}
	}

	// Copy new binary over current
	newData, err := os.ReadFile(evolveState.TempBinary)
	if err != nil {
		return "", fmt.Errorf("cannot read new binary: %w", err)
	}
	if err := os.WriteFile(currentExe, newData, 0755); err != nil {
		return "", fmt.Errorf("cannot deploy new binary (try stopping the process first): %w", err)
	}

	// Git commit the source changes
	sourceDir := evolveState.SourceDir
	gitCommit := func() string {
		cmd := exec.Command("git", "add", "main.go")
		cmd.Dir = sourceDir
		cmd.Run()

		commitMsg := fmt.Sprintf("self-evolve: %s\n\nPatches applied:\n", reason)
		for i, p := range evolveState.Patches {
			commitMsg += fmt.Sprintf("  %d. %s\n", i+1, p.Description)
		}

		cmd = exec.Command("git", "commit", "-m", commitMsg)
		cmd.Dir = sourceDir
		cmd.Run()

		cmd = exec.Command("git", "rev-parse", "--short", "HEAD")
		cmd.Dir = sourceDir
		if out, err := cmd.Output(); err == nil {
			return strings.TrimSpace(string(out))
		}
		return "unknown"
	}
	gitHash := gitCommit()

	// Log evolution history
	if selfDir != "" {
		histDir := filepath.Join(selfDir, "evolve")
		os.MkdirAll(histDir, 0755)

		history := EvolveHistory{
			Timestamp:   time.Now(),
			GitHash:     gitHash,
			PatchCount:  len(evolveState.Patches),
			TestsPassed: true,
			Reason:      reason,
			Success:     true,
		}
		histData, _ := json.Marshal(history)
		if f, err := os.OpenFile(filepath.Join(histDir, "history.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644); err == nil {
			fmt.Fprintln(f, string(histData))
			f.Close()
		}
	}

	// Clean up
	os.Remove(evolveState.TempBinary)
	patchCount := len(evolveState.Patches)
	evolveState = nil

	fmt.Printf("[siki] Self-evolution deployed: %s (commit %s, %d patches). Restarting...\n", reason, gitHash, patchCount)

	// Schedule graceful restart
	go func() {
		time.Sleep(1 * time.Second) // Let HTTP response be sent
		fmt.Println("[siki] Executing new binary via syscall.Exec...")
		err := syscall.Exec(currentExe, os.Args, os.Environ())
		if err != nil {
			fmt.Printf("[siki] syscall.Exec failed: %v (manual restart required)\n", err)
		}
	}()

	return fmt.Sprintf("Evolution deployed successfully!\n- Git commit: %s\n- Patches applied: %d\n- Reason: %s\n- Backup: %s\n\nProcess will restart in ~1 second. Connections will reconnect automatically.", gitHash, patchCount, reason, backupPath), nil
}

func (a *Agent) evolveStatus() (string, error) {
	evolveMu.Lock()
	defer evolveMu.Unlock()

	sourceDir := detectSourceDir()

	if evolveState == nil {
		var sb strings.Builder
		sb.WriteString("No evolution in progress.\n")
		if sourceDir != "" {
			if data, err := os.ReadFile(filepath.Join(sourceDir, "main.go")); err == nil {
				lines := strings.Count(string(data), "\n") + 1
				sb.WriteString(fmt.Sprintf("Source: %s/main.go (%d lines)\n", sourceDir, lines))
			}
		}

		// Show evolution history
		if selfDir != "" {
			histPath := filepath.Join(selfDir, "evolve", "history.jsonl")
			if data, err := os.ReadFile(histPath); err == nil {
				lines := strings.Split(strings.TrimSpace(string(data)), "\n")
				if len(lines) > 0 && lines[0] != "" {
					sb.WriteString(fmt.Sprintf("\nEvolution history (%d entries):\n", len(lines)))
					// Show last 5
					start := 0
					if len(lines) > 5 {
						start = len(lines) - 5
					}
					for i := start; i < len(lines); i++ {
						var h EvolveHistory
						if json.Unmarshal([]byte(lines[i]), &h) == nil {
							status := "OK"
							if !h.Success {
								status = "FAILED"
							}
							sb.WriteString(fmt.Sprintf("  [%s] %s - %s (%d patches, commit %s)\n",
								status, h.Timestamp.Format("01/02 15:04"), h.Reason, h.PatchCount, h.GitHash))
						}
					}
				}
			}
		}

		sb.WriteString("\nUse action=view_source to examine code, then action=patch to propose changes.")
		return sb.String(), nil
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Evolution in progress (started: %s)\n", evolveState.StartedAt.Format("15:04:05")))
	sb.WriteString(fmt.Sprintf("Source: %s\n", evolveState.SourceDir))
	sb.WriteString(fmt.Sprintf("Patches: %d\n", len(evolveState.Patches)))
	for i, p := range evolveState.Patches {
		sb.WriteString(fmt.Sprintf("  %d. %s (applied=%v)\n", i+1, p.Description, p.Applied))
	}
	sb.WriteString(fmt.Sprintf("Build: %s\n", evolveState.BuildStatus))
	sb.WriteString(fmt.Sprintf("Tests: %s\n", evolveState.TestStatus))

	if evolveState.BuildStatus == "success" && evolveState.TestStatus == "success" {
		sb.WriteString("\nReady to deploy! Use action=deploy with description.")
	}

	return sb.String(), nil
}

func (a *Agent) evolveAbort() (string, error) {
	evolveMu.Lock()
	defer evolveMu.Unlock()

	if evolveState == nil {
		return "No evolution in progress.", nil
	}

	// Restore original source
	if evolveState.OrigSource != "" && evolveState.SourceDir != "" {
		if err := os.WriteFile(filepath.Join(evolveState.SourceDir, "main.go"), []byte(evolveState.OrigSource), 0644); err != nil {
			return "", fmt.Errorf("failed to restore original source: %w", err)
		}
	}

	// Clean up temp binary
	if evolveState.TempBinary != "" {
		os.Remove(evolveState.TempBinary)
	}

	patchCount := len(evolveState.Patches)
	evolveState = nil

	return fmt.Sprintf("Evolution aborted. %d patches reverted. Source restored to original.", patchCount), nil
}

// ============================================================================
// Docker GPU Container Management
// ============================================================================

var (
	dockerContainerName = "siki-worker"
	dockerImageName     = "siki-worker:latest"
	dockerMu            sync.Mutex
	dockerWorkspaceDir  string
)

const dockerfileContent = `FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip python3-venv sox libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir openai-whisper
RUN mkdir -p /workspace
WORKDIR /workspace
CMD ["sleep", "infinity"]
`

func initDockerWorkspace() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	dockerWorkspaceDir = filepath.Join(home, ".siki", "workspace")
	return os.MkdirAll(dockerWorkspaceDir, 0755)
}

func isDockerAvailable() bool {
	_, err := exec.LookPath("docker")
	return err == nil
}

func isDockerImageAvailable() bool {
	cmd := exec.Command("docker", "image", "inspect", dockerImageName)
	return cmd.Run() == nil
}

func buildDockerImage() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	dockerDir := filepath.Join(home, ".siki", "docker")
	if err := os.MkdirAll(dockerDir, 0755); err != nil {
		return err
	}
	dockerfilePath := filepath.Join(dockerDir, "Dockerfile")
	if err := os.WriteFile(dockerfilePath, []byte(dockerfileContent), 0644); err != nil {
		return err
	}
	fmt.Println("[siki] Building Docker image siki-worker (this may take a while)...")
	cmd := exec.Command("docker", "build", "-t", dockerImageName, "-f", dockerfilePath, dockerDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func ensureDockerContainer() error {
	dockerMu.Lock()
	defer dockerMu.Unlock()

	if !isDockerAvailable() {
		return fmt.Errorf("docker is not installed")
	}

	if !isDockerImageAvailable() {
		if err := buildDockerImage(); err != nil {
			return fmt.Errorf("failed to build Docker image: %w", err)
		}
	}

	if err := initDockerWorkspace(); err != nil {
		return fmt.Errorf("failed to init workspace: %w", err)
	}

	// Check if container is already running
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", dockerContainerName)
	out, err := cmd.Output()
	if err == nil && strings.TrimSpace(string(out)) == "true" {
		return nil // already running
	}

	// Remove old stopped container if exists
	exec.Command("docker", "rm", "-f", dockerContainerName).Run()

	// Start new container with GPU access
	fmt.Println("[siki] Starting Docker container siki-worker with GPU access...")
	startCmd := exec.Command("docker", "run", "-d",
		"--name", dockerContainerName,
		"--gpus", "all",
		"-v", dockerWorkspaceDir+":/workspace",
		dockerImageName,
	)
	startCmd.Stdout = os.Stdout
	startCmd.Stderr = os.Stderr
	return startCmd.Run()
}

func dockerExec(command string, timeoutSec int) (string, error) {
	if err := ensureDockerContainer(); err != nil {
		return "", err
	}

	if timeoutSec <= 0 {
		timeoutSec = 300
	}
	if timeoutSec > 3600 {
		timeoutSec = 3600
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSec)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "docker", "exec", dockerContainerName, "sh", "-c", command)
	output, err := cmd.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return string(output), fmt.Errorf("command timed out after %d seconds", timeoutSec)
	}
	if err != nil {
		return string(output) + "\nError: " + err.Error(), nil
	}
	return string(output), nil
}

func stopDockerContainer() {
	fmt.Println("[siki] Stopping Docker container siki-worker...")
	exec.Command("docker", "stop", dockerContainerName).Run()
	exec.Command("docker", "rm", dockerContainerName).Run()
}

type Thread struct {
	ID           string          `json:"id"`
	Title        string          `json:"title"`
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
	Messages     []ThreadMessage `json:"messages,omitempty"` // omitempty: metadata JSON has no messages
	MessageCount int             `json:"message_count,omitempty"`
	Summary      string          `json:"summary,omitempty"`
}

type ThreadMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Thinking   string     `json:"thinking,omitempty"`
	Images     []string   `json:"images,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolName   string     `json:"tool_name,omitempty"`
	Summarized bool       `json:"summarized,omitempty"`
	Timestamp  int64      `json:"timestamp"`
}

type ThreadListItem struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	MessageCount int       `json:"message_count"`
	Summary      string    `json:"summary,omitempty"`
}

func initThreadDir() error {
	if threadDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	threadDir = filepath.Join(home, ".siki", "threads")
	return os.MkdirAll(threadDir, 0755)
}

// saveThreadMeta saves only the thread metadata (no messages) to {id}.json.
// Messages are stored separately in {id}.jsonl (append-only).
func saveThreadMeta(t *Thread) error {
	if err := initThreadDir(); err != nil {
		return err
	}
	meta := Thread{
		ID:           t.ID,
		Title:        t.Title,
		CreatedAt:    t.CreatedAt,
		UpdatedAt:    t.UpdatedAt,
		MessageCount: t.MessageCount,
		Summary:      t.Summary,
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(threadDir, t.ID+".json"), data, 0644)
}

// saveThread saves metadata; kept as alias for compatibility with callers that
// don't need to write messages (e.g. handleThreads POST, rename).
func saveThread(t *Thread) error {
	return saveThreadMeta(t)
}

// appendToLog appends a single ThreadMessage as a JSON line to {id}.jsonl.
func appendToLog(threadID string, tm ThreadMessage) error {
	if err := initThreadDir(); err != nil {
		return err
	}
	f, err := os.OpenFile(filepath.Join(threadDir, threadID+".jsonl"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	data, err := json.Marshal(tm)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(f, "%s\n", data)
	return err
}

// loadThreadMessages reads all messages from {id}.jsonl.
func loadThreadMessages(id string) ([]ThreadMessage, error) {
	if err := initThreadDir(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(threadDir, id+".jsonl"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var msgs []ThreadMessage
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var tm ThreadMessage
		if err := json.Unmarshal([]byte(line), &tm); err != nil {
			fmt.Printf("[siki] Warning: skipping malformed JSONL line: %v\n", err)
			continue
		}
		msgs = append(msgs, tm)
	}
	return msgs, nil
}

// loadThreadMeta loads only metadata from {id}.json (no messages).
// Fast — used for title updates, listing, etc.
func loadThreadMeta(id string) (*Thread, error) {
	if err := initThreadDir(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(threadDir, id+".json"))
	if err != nil {
		return nil, err
	}
	var t Thread
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	t.Messages = nil // don't return old-format messages
	return &t, nil
}

// loadThread loads metadata from {id}.json and messages from {id}.jsonl.
// Backward compatible: if the .json file contains messages (old format), migrate them.
func loadThread(id string) (*Thread, error) {
	if err := initThreadDir(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(threadDir, id+".json"))
	if err != nil {
		return nil, err
	}
	var t Thread
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}

	// Backward compatibility: if .json has messages (old format), migrate to .jsonl
	if len(t.Messages) > 0 {
		jsonlPath := filepath.Join(threadDir, id+".jsonl")
		if _, err := os.Stat(jsonlPath); os.IsNotExist(err) {
			fmt.Printf("[siki] Migrating thread %s: %d messages → JSONL\n", id, len(t.Messages))
			for _, tm := range t.Messages {
				if err := appendToLog(id, tm); err != nil {
					return &t, nil // return with old messages on migration error
				}
			}
		}
		// Fix "New thread" / "New conversation" titles during migration
		if t.Title == "New thread" || t.Title == "New conversation" || t.Title == "" {
			for _, tm := range t.Messages {
				if tm.Role == "user" && tm.Content != "" {
					title := tm.Content
					if len(title) > 50 {
						title = title[:50] + "..."
					}
					t.Title = title
					break
				}
			}
		}
		// Update metadata without messages
		t.MessageCount = len(t.Messages)
		t.Messages = nil
		saveThreadMeta(&t)
	}

	// Load messages from JSONL
	msgs, err := loadThreadMessages(id)
	if err != nil {
		return &t, nil
	}
	t.Messages = msgs
	if t.MessageCount == 0 {
		t.MessageCount = len(msgs)
	}
	return &t, nil
}

func listThreads() ([]ThreadListItem, error) {
	if err := initThreadDir(); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(threadDir)
	if err != nil {
		return nil, err
	}
	var items []ThreadListItem
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		// Skip .jsonl files (only process .json metadata)
		name := entry.Name()
		if strings.HasSuffix(name, ".jsonl") {
			continue
		}
		id := strings.TrimSuffix(name, ".json")
		// Check if this is old format (has messages in .json, no .jsonl yet)
		jsonlPath := filepath.Join(threadDir, id+".jsonl")
		if _, err := os.Stat(jsonlPath); os.IsNotExist(err) {
			// Might be old format — do a full load to trigger migration
			if t, err := loadThread(id); err == nil {
				items = append(items, ThreadListItem{
					ID:           t.ID,
					Title:        t.Title,
					CreatedAt:    t.CreatedAt,
					UpdatedAt:    t.UpdatedAt,
					MessageCount: t.MessageCount,
					Summary:      t.Summary,
				})
			}
			continue
		}
		// Read metadata only (fast, no message loading)
		data, err := os.ReadFile(filepath.Join(threadDir, name))
		if err != nil {
			continue
		}
		var t Thread
		if err := json.Unmarshal(data, &t); err != nil {
			continue
		}
		items = append(items, ThreadListItem{
			ID:           t.ID,
			Title:        t.Title,
			CreatedAt:    t.CreatedAt,
			UpdatedAt:    t.UpdatedAt,
			MessageCount: t.MessageCount,
			Summary:      t.Summary,
		})
	}
	return items, nil
}

func deleteThread(id string) error {
	if err := initThreadDir(); err != nil {
		return err
	}
	// Remove both metadata and message log
	os.Remove(filepath.Join(threadDir, id+".jsonl"))
	return os.Remove(filepath.Join(threadDir, id+".json"))
}

func getRecentThreadMessages(thread *Thread, n int) []ThreadMessage {
	if len(thread.Messages) <= n {
		return thread.Messages
	}
	return thread.Messages[len(thread.Messages)-n:]
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// generateThreadTitle uses the LLM to create a concise thread title from the
// first user message and assistant response. Runs asynchronously.
func generateThreadTitle(config *Config, threadID, userMessage, assistantResponse string) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	prompt := fmt.Sprintf("以下の会話の内容を表す短いタイトル（15文字以内）を1つだけ出力してください。説明や記号は不要です。\n\nユーザー: %s", userMessage)
	if len(assistantResponse) > 300 {
		assistantResponse = assistantResponse[:300]
	}
	if assistantResponse != "" {
		prompt += fmt.Sprintf("\nアシスタント: %s", assistantResponse)
	}

	messages := []Message{
		{Role: "user", Content: prompt},
	}

	req := ChatRequest{
		Model:       config.primaryProvider().Model,
		Messages:    messages,
		MaxTokens:   50,
		Temperature: 0.3,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return
	}
	setProviderHeaders(httpReq, config.primaryProvider())

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return
	}
	if len(chatResp.Choices) == 0 {
		return
	}

	title := strings.TrimSpace(chatResp.Choices[0].Message.Content)
	// Clean up: remove quotes, periods, etc.
	title = strings.Trim(title, "\"'「」『』。.")
	if title == "" {
		return
	}
	if len(title) > 50 {
		title = title[:50]
	}

	thread, err := loadThreadMeta(threadID)
	if err != nil {
		return
	}
	thread.Title = title
	thread.UpdatedAt = time.Now()
	saveThreadMeta(thread)
	fmt.Printf("[siki] Thread %s titled: %s\n", threadID, title)
}


// appendMessageToThread appends a single message to the thread log (JSONL, append-only).
// Also updates thread metadata (title, timestamps, message count).
// toolNameHint is used for tool result messages to record which tool produced the result.
func appendMessageToThread(threadID string, msg Message, toolNameHint string) {
	if msg.Role == "system" {
		return
	}

	tm := ThreadMessage{
		Role:       msg.Role,
		Content:    msg.Content,
		Thinking:   msg.Thinking,
		Images:     msg.Images,
		ToolCalls:  msg.ToolCalls,
		ToolCallID: msg.ToolCallID,
		Timestamp:  time.Now().Unix(),
	}
	if msg.Role == "assistant" && (strings.HasPrefix(msg.Content, "[以前の会話の要約]") || strings.HasPrefix(msg.Content, "[Previous conversation summary]")) {
		tm.Summarized = true
	}
	if msg.Role == "tool" && toolNameHint != "" {
		tm.ToolName = toolNameHint
	}

	// Append message to JSONL log (append-only, never rewrite)
	if err := appendToLog(threadID, tm); err != nil {
		fmt.Printf("[siki] Warning: failed to append message to log %s: %v\n", threadID, err)
		return
	}

	// Update thread metadata
	metaPath := filepath.Join(threadDir, threadID+".json")
	var thread Thread
	if data, err := os.ReadFile(metaPath); err == nil {
		json.Unmarshal(data, &thread)
	} else {
		// Thread metadata doesn't exist yet — create it
		thread = Thread{
			ID:        threadID,
			CreatedAt: time.Now(),
		}
	}

	// Set title from first user message
	if msg.Role == "user" && thread.MessageCount == 0 {
		title := msg.Content
		if len(title) > 50 {
			title = title[:50] + "..."
		}
		thread.Title = title
	}
	thread.MessageCount++
	thread.UpdatedAt = time.Now()
	saveThreadMeta(&thread)
}

func initPlaygroundDir() error {
	if playgroundDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	playgroundDir = filepath.Join(home, ".siki", "playground")
	return os.MkdirAll(playgroundDir, 0755)
}

func initPluginDir() error {
	if pluginDir != "" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	pluginDir = filepath.Join(home, ".siki", "plugins")
	return os.MkdirAll(pluginDir, 0755)
}

func loadPlugins() error {
	if err := initPluginDir(); err != nil {
		return err
	}
	entries, err := os.ReadDir(pluginDir)
	if err != nil {
		return err
	}
	pluginMu.Lock()
	defer pluginMu.Unlock()
	loadedPlugins = nil
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(pluginDir, entry.Name()))
		if err != nil {
			fmt.Printf("[siki] Warning: failed to read plugin %s: %v\n", entry.Name(), err)
			continue
		}
		var p Plugin
		if err := json.Unmarshal(data, &p); err != nil {
			fmt.Printf("[siki] Warning: failed to parse plugin %s: %v\n", entry.Name(), err)
			continue
		}
		loadedPlugins = append(loadedPlugins, p)
	}
	fmt.Printf("[siki] Loaded %d plugins\n", len(loadedPlugins))
	return nil
}

func savePlugin(p Plugin) error {
	if err := initPluginDir(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(pluginDir, p.Name+".json"), data, 0644)
}

func deletePluginFile(name string) error {
	return os.Remove(filepath.Join(pluginDir, name+".json"))
}

func (a *Agent) runCode(htmlCode, title string) (string, error) {
	if err := initPlaygroundDir(); err != nil {
		return "", fmt.Errorf("failed to create playground dir: %w", err)
	}

	// Generate unique filename
	filename := fmt.Sprintf("playground_%d.html", time.Now().UnixNano())
	outputPath := filepath.Join(playgroundDir, filename)

	// Wrap the code in a complete HTML document if needed
	var fullHTML string
	if strings.Contains(htmlCode, "<html") || strings.Contains(htmlCode, "<!DOCTYPE") {
		fullHTML = htmlCode
	} else {
		fullHTML = fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>%s</title>
    <style>
        body { font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        canvas { border: 1px solid #333; }
    </style>
</head>
<body>
%s
</body>
</html>`, title, htmlCode)
	}

	// Write the HTML file
	if err := os.WriteFile(outputPath, []byte(fullHTML), 0644); err != nil {
		return "", fmt.Errorf("failed to write playground file: %w", err)
	}

	// Return an embedded iframe - simple format for reliable rendering
	playgroundURL := fmt.Sprintf("/playground/%s", filename)
	return fmt.Sprintf(`<iframe src="%s" style="width:100%%;height:500px;border:1px solid #444;border-radius:8px;"></iframe>`, playgroundURL), nil
}

func extractTextFromHTML(html string) string {
	// Remove script and style tags
	html = removeTagContent(html, "script")
	html = removeTagContent(html, "style")
	html = removeTagContent(html, "nav")
	html = removeTagContent(html, "header")
	html = removeTagContent(html, "footer")

	// Remove all HTML tags
	var result strings.Builder
	inTag := false
	for _, r := range html {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			result.WriteRune(' ')
			continue
		}
		if !inTag {
			result.WriteRune(r)
		}
	}

	text := result.String()

	// Clean up whitespace
	text = cleanHTMLEntities(text)
	lines := strings.Split(text, "\n")
	var cleanLines []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			cleanLines = append(cleanLines, line)
		}
	}

	return strings.Join(cleanLines, "\n")
}

func removeTagContent(html, tag string) string {
	for {
		startTag := strings.Index(strings.ToLower(html), "<"+tag)
		if startTag == -1 {
			break
		}
		endTag := strings.Index(strings.ToLower(html[startTag:]), "</"+tag+">")
		if endTag == -1 {
			// No closing tag, just remove the opening tag
			closeIdx := strings.Index(html[startTag:], ">")
			if closeIdx != -1 {
				html = html[:startTag] + html[startTag+closeIdx+1:]
			} else {
				break
			}
		} else {
			html = html[:startTag] + html[startTag+endTag+len("</"+tag+">"):]
		}
	}
	return html
}

func (a *Agent) resolvePath(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(a.config.Workspace, path)
}

// ============================================================================
// LLM Client (OpenAI-compatible API)
// ============================================================================

type Message struct {
	Role       string     `json:"-"`
	Content    string     `json:"-"`
	Thinking   string     `json:"-"` // <think> content, saved to log but excluded from LLM context
	Images     []string   `json:"-"` // base64 data URIs for vision
	ToolCalls  []ToolCall `json:"-"`
	ToolCallID string     `json:"-"`
}

func (m Message) MarshalJSON() ([]byte, error) {
	type basicMsg struct {
		Role       string     `json:"role"`
		Content    interface{} `json:"content,omitempty"`
		ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
		ToolCallID string     `json:"tool_call_id,omitempty"`
	}

	if len(m.Images) == 0 {
		return json.Marshal(basicMsg{
			Role:       m.Role,
			Content:    m.Content,
			ToolCalls:  m.ToolCalls,
			ToolCallID: m.ToolCallID,
		})
	}

	// Multimodal: content is array of parts
	parts := []interface{}{}
	if m.Content != "" {
		parts = append(parts, map[string]string{"type": "text", "text": m.Content})
	}
	for _, img := range m.Images {
		parts = append(parts, map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]string{"url": img},
		})
	}

	return json.Marshal(basicMsg{
		Role:       m.Role,
		Content:    parts,
		ToolCalls:  m.ToolCalls,
		ToolCallID: m.ToolCallID,
	})
}

func (m *Message) UnmarshalJSON(data []byte) error {
	type alias struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content,omitempty"`
		ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID string          `json:"tool_call_id,omitempty"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	m.Role = a.Role
	m.ToolCalls = a.ToolCalls
	m.ToolCallID = a.ToolCallID

	// Content can be string or array; we only need string for responses
	if len(a.Content) > 0 {
		var s string
		if err := json.Unmarshal(a.Content, &s); err == nil {
			m.Content = s
		} else {
			// It's an array (from our own multimodal messages), extract text parts
			var parts []map[string]interface{}
			if err := json.Unmarshal(a.Content, &parts); err == nil {
				for _, p := range parts {
					if p["type"] == "text" {
						if t, ok := p["text"].(string); ok {
							m.Content += t
						}
					}
				}
			}
		}
	}
	return nil
}

type ChatRequest struct {
	Model       string                   `json:"model"`
	Messages    []Message                `json:"messages"`
	Tools       []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice  string                   `json:"tool_choice,omitempty"`
	MaxTokens   int                      `json:"max_tokens,omitempty"`
	Temperature float64                  `json:"temperature,omitempty"`
	Stream      bool                     `json:"stream,omitempty"`
}

type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int      `json:"index"`
		Message      Message  `json:"message"`
		FinishReason string   `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func (a *Agent) checkServerHealth() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, _ := http.NewRequestWithContext(ctx, "GET", a.config.primaryProvider().Endpoint+"/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("server not available at %s", a.config.primaryProvider().Endpoint)
	}
	resp.Body.Close()
	return nil
}

func (a *Agent) chat(ctx context.Context) (*Message, error) {
	return a.chatStream(ctx, StreamCallbacks{})
}

// validateResponse checks if the response is relevant to the user's question
func (a *Agent) validateResponse(ctx context.Context, userMessage string, response string) (bool, string) {
	validateMessages := []Message{
		{Role: "system", Content: `You are a response quality checker. Given a user's question and an AI's response, determine if the response is relevant and appropriate.
Reply with ONLY "OK" if the response is relevant and addresses the question.
Reply with "NG: <reason>" if the response is irrelevant, off-topic, or ignores the conversation context.
Be lenient - only flag clearly irrelevant or nonsensical responses.`},
		{Role: "user", Content: fmt.Sprintf("User's question: %s\n\nAI's response: %s", userMessage, response)},
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    validateMessages,
		MaxTokens:   100,
		Temperature: 0.1,
		Stream:      false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return true, "" // assume OK on error
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		a.config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return true, ""
	}
	a.setAuthHeaders(httpReq)

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return true, ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return true, ""
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return true, ""
	}

	if len(chatResp.Choices) == 0 {
		return true, ""
	}

	result := strings.TrimSpace(chatResp.Choices[0].Message.Content)
	if strings.HasPrefix(result, "OK") {
		return true, ""
	}
	reason := strings.TrimPrefix(result, "NG: ")
	reason = strings.TrimPrefix(reason, "NG:")
	return false, strings.TrimSpace(reason)
}

// compressConversation summarizes older messages to reduce context size
func (a *Agent) compressConversation(ctx context.Context) {
	// Keep system message + last 60 messages, compress everything older
	// Model has 64k context so we can keep a lot of history
	keepRecent := 60
	selfMu.RLock()
	if currentSelf != nil && currentSelf.Params.CompressAt > 0 {
		keepRecent = currentSelf.Params.CompressAt
	}
	selfMu.RUnlock()
	if len(a.messages) <= keepRecent+2 {
		return
	}

	// Messages to compress: from index 1 (after system) to len-keepRecent
	compressEnd := len(a.messages) - keepRecent
	if compressEnd <= 1 {
		return
	}

	var historyText strings.Builder
	for i := 1; i < compressEnd; i++ {
		msg := a.messages[i]
		content := msg.Content
		if len(content) > 1000 {
			content = content[:1000] + "..."
		}
		switch msg.Role {
		case "user":
			historyText.WriteString(fmt.Sprintf("[user]: %s\n", content))
		case "assistant":
			historyText.WriteString(fmt.Sprintf("[assistant]: %s\n", content))
		case "tool":
			historyText.WriteString(fmt.Sprintf("[tool result]: %s\n", content))
		}
	}

	if historyText.Len() == 0 {
		return
	}

	history := historyText.String()
	if len(history) > 30000 {
		history = history[:30000]
	}

	compressMessages := []Message{
		{Role: "system", Content: `以下の会話履歴を要約してください。以下を必ず保持すること:
- ユーザーが質問した内容とその回答の要点
- ツールで取得した重要な情報（検索結果、ウェブページの内容など）
- 会話の流れと現在のトピック
2000文字以内で要約してください。`},
		{Role: "user", Content: history},
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    compressMessages,
		MaxTokens:   2500,
		Temperature: 0.1,
		Stream:      false,
	}

	body, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		a.config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return
	}
	a.setAuthHeaders(httpReq)

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil || len(chatResp.Choices) == 0 {
		return
	}

	summary := chatResp.Choices[0].Message.Content
	fmt.Printf("[siki] Compressed %d messages into summary (%d chars)\n", compressEnd-1, len(summary))

	// ACE Reflector: extract insights before discarding old messages
	go func(msgs []Message, cfg *Config) {
		newBullets := reflectOnConversation(cfg, msgs)
		if len(newBullets) > 0 {
			existing, _ := loadPlaybook()
			merged := curateBullets(existing, newBullets)
			if err := savePlaybook(merged); err != nil {
				fmt.Printf("[siki] Failed to save playbook: %v\n", err)
			} else {
				fmt.Printf("[siki] Playbook updated on compression: %d bullets\n", len(merged))
			}
		}
	}(a.messages[1:compressEnd], a.config)

	// Rebuild messages: system + summary + recent messages
	newMessages := []Message{a.messages[0]}
	newMessages = append(newMessages, Message{
		Role:    "assistant",
		Content: fmt.Sprintf("[以前の会話の要約]\n%s", summary),
	})
	newMessages = append(newMessages, a.messages[compressEnd:]...)
	a.messages = newMessages
}

// forceCompressConversation aggressively compresses conversation regardless of message count
// Used when context deadline is exceeded to reduce context size
func (a *Agent) forceCompressConversation(ctx context.Context) {
	if len(a.messages) <= 2 {
		return // Only system + 1 message, nothing to compress
	}

	// Keep system message + last 2 messages, compress everything else
	keepLast := 2
	if len(a.messages)-1 < keepLast {
		keepLast = len(a.messages) - 1
	}
	compressEnd := len(a.messages) - keepLast
	if compressEnd <= 1 {
		return
	}

	var historyText strings.Builder
	for i := 1; i < compressEnd; i++ {
		msg := a.messages[i]
		if msg.Role == "tool" {
			continue
		}
		historyText.WriteString(fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content))
	}

	if historyText.Len() == 0 {
		return
	}

	// Truncate history text if too long (more aggressive limit)
	history := historyText.String()
	if len(history) > 6000 {
		history = history[:6000]
	}

	compressMessages := []Message{
		{Role: "system", Content: "会話履歴を簡潔に要約してください。重要な事実、ユーザーの質問の要点、AIの回答の要点を保持してください。150文字以内で要約してください。"},
		{Role: "user", Content: history},
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    compressMessages,
		MaxTokens:   200,
		Temperature: 0.1,
		Stream:      false,
	}

	body, _ := json.Marshal(req)
	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		a.config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		// Fallback: just truncate messages without summarizing
		a.messages = append([]Message{a.messages[0]}, a.messages[compressEnd:]...)
		return
	}
	a.setAuthHeaders(httpReq)

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		// Fallback: just truncate messages without summarizing
		a.messages = append([]Message{a.messages[0]}, a.messages[compressEnd:]...)
		fmt.Printf("[siki] Force compress failed (LLM error), truncated %d messages\n", compressEnd-1)
		return
	}
	defer resp.Body.Close()

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil || len(chatResp.Choices) == 0 {
		// Fallback: just truncate
		a.messages = append([]Message{a.messages[0]}, a.messages[compressEnd:]...)
		fmt.Printf("[siki] Force compress failed (decode error), truncated %d messages\n", compressEnd-1)
		return
	}

	summary := chatResp.Choices[0].Message.Content
	fmt.Printf("[siki] Force compressed %d messages into summary\n", compressEnd-1)

	// Rebuild messages: system + summary + last few messages
	newMessages := []Message{a.messages[0]}
	newMessages = append(newMessages, Message{
		Role:    "assistant",
		Content: fmt.Sprintf("[以前の会話の要約]\n%s", summary),
	})
	newMessages = append(newMessages, a.messages[compressEnd:]...)
	a.messages = newMessages
}

// searchConversation searches through the complete thread log for relevant messages
func (a *Agent) searchConversation(query string) string {
	// Search the complete thread log file, not the in-memory LLM context
	thread, err := loadThread(a.threadID)
	if err != nil {
		// Fallback: search in-memory messages
		return a.searchConversationInMemory(query)
	}

	var results []string
	queryLower := strings.ToLower(query)

	for i, m := range thread.Messages {
		if m.Role == "system" {
			continue
		}
		if m.Summarized {
			continue
		}
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0 {
			continue
		}
		if m.Role == "assistant" && strings.HasPrefix(m.Content, "[tool_calls:") {
			continue
		}
		searchContent := m.Content
		roleLabel := m.Role
		if m.Role == "tool" && m.ToolName != "" {
			roleLabel = "tool:" + m.ToolName
		}
		contentLower := strings.ToLower(searchContent)
		if strings.Contains(contentLower, queryLower) {
			snippet := searchContent
			if len(snippet) > 300 {
				idx := strings.Index(contentLower, queryLower)
				start := idx - 100
				if start < 0 {
					start = 0
				}
				end := idx + len(query) + 100
				if end > len(snippet) {
					end = len(snippet)
				}
				snippet = "..." + snippet[start:end] + "..."
			}
			ts := time.Unix(m.Timestamp, 0).Format("2006-01-02 15:04")
			results = append(results, fmt.Sprintf("- [#%d %s %s]: %s", i+1, ts, roleLabel, snippet))
		}
	}

	if len(results) == 0 {
		return fmt.Sprintf("会話ログに「%s」に関する言及は見つかりませんでした（%d件のメッセージを検索）。", query, len(thread.Messages))
	}
	if len(results) > 20 {
		results = results[len(results)-20:]
	}

	return fmt.Sprintf("会話ログで「%s」に関する言及（%d件中%d件マッチ）:\n\n%s", query, len(thread.Messages), len(results), strings.Join(results, "\n"))
}

// searchThreadLogForContext searches the thread log from newest to oldest,
// extracting messages relevant to the query. Returns context text for LLM re-prompting.
func (a *Agent) searchThreadLogForContext(query string) string {
	thread, err := loadThread(a.threadID)
	if err != nil || len(thread.Messages) == 0 {
		return ""
	}

	// Extract keywords from query (split by spaces, filter short words)
	keywords := extractKeywords(query)
	if len(keywords) == 0 {
		return ""
	}

	// Search from newest to oldest
	var hits []string
	totalChars := 0
	const maxChars = 4000

	for i := len(thread.Messages) - 1; i >= 0; i-- {
		m := thread.Messages[i]
		if m.Summarized || m.Role == "system" {
			continue
		}
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0 {
			continue
		}
		if m.Role == "assistant" && strings.HasPrefix(m.Content, "[tool_calls:") {
			continue
		}
		contentLower := strings.ToLower(m.Content)
		matched := false
		for _, kw := range keywords {
			if strings.Contains(contentLower, kw) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}

		roleLabel := m.Role
		if m.Role == "tool" && m.ToolName != "" {
			roleLabel = "tool:" + m.ToolName
		}
		snippet := m.Content
		if len(snippet) > 500 {
			snippet = snippet[:500] + "..."
		}
		ts := time.Unix(m.Timestamp, 0).Format("01/02 15:04")
		hit := fmt.Sprintf("[%s %s]: %s", ts, roleLabel, snippet)
		totalChars += len(hit)
		if totalChars > maxChars {
			break
		}
		hits = append(hits, hit)
	}

	if len(hits) == 0 {
		return ""
	}

	// Reverse to chronological order
	for i, j := 0, len(hits)-1; i < j; i, j = i+1, j-1 {
		hits[i], hits[j] = hits[j], hits[i]
	}

	return strings.Join(hits, "\n")
}

// extractKeywords splits text into meaningful search keywords
func extractKeywords(text string) []string {
	words := strings.Fields(strings.ToLower(text))
	var keywords []string
	seen := make(map[string]bool)
	for _, w := range words {
		// Skip very short words and common particles
		if len(w) < 2 {
			continue
		}
		if seen[w] {
			continue
		}
		seen[w] = true
		keywords = append(keywords, w)
	}
	return keywords
}

// searchConversationInMemory is a fallback that searches in-memory agent messages
func (a *Agent) searchConversationInMemory(query string) string {
	var results []string
	queryLower := strings.ToLower(query)

	for i, msg := range a.messages {
		if msg.Role == "system" || msg.Role == "tool" {
			continue
		}
		contentLower := strings.ToLower(msg.Content)
		if strings.Contains(contentLower, queryLower) {
			snippet := msg.Content
			if len(snippet) > 200 {
				idx := strings.Index(contentLower, queryLower)
				start := idx - 80
				if start < 0 {
					start = 0
				}
				end := idx + len(query) + 80
				if end > len(snippet) {
					end = len(snippet)
				}
				snippet = "..." + snippet[start:end] + "..."
			}
			results = append(results, fmt.Sprintf("- [メッセージ%d/%s]: %s", i, msg.Role, snippet))
		}
	}

	if len(results) == 0 {
		return fmt.Sprintf("会話履歴に「%s」に関する言及は見つかりませんでした。", query)
	}
	return fmt.Sprintf("会話履歴で「%s」に関する言及:\n\n%s", query, strings.Join(results, "\n"))
}

func (a *Agent) recallContext(query string) (string, error) {
	thread, err := loadThread(a.threadID)
	if err != nil {
		return "No conversation history found.", nil
	}
	// Search through ALL messages in the complete thread log
	queryLower := strings.ToLower(query)
	var matches []string
	totalMessages := len(thread.Messages)
	for i, m := range thread.Messages {
		// Skip internal markers
		if m.Summarized {
			continue
		}
		if m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0 {
			continue
		}
		if m.Role == "assistant" && strings.HasPrefix(m.Content, "[tool_calls:") {
			continue
		}
		if strings.Contains(strings.ToLower(m.Content), queryLower) {
			roleLabel := m.Role
			if m.Role == "tool" && m.ToolName != "" {
				roleLabel = "tool:" + m.ToolName
			}
			ts := time.Unix(m.Timestamp, 0).Format("2006-01-02 15:04")
			content := truncateString(m.Content, 500)
			matches = append(matches, fmt.Sprintf("[#%d %s %s]: %s", i+1, ts, roleLabel, content))
		}
	}
	if len(matches) == 0 {
		return fmt.Sprintf("No matches found for '%s' in current thread (%d messages searched).", query, totalMessages), nil
	}
	// Return up to 20 matches (most recent)
	if len(matches) > 20 {
		matches = matches[len(matches)-20:]
	}
	return fmt.Sprintf("Found %d matches in thread (%d total messages):\n\n%s", len(matches), totalMessages, strings.Join(matches, "\n\n")), nil
}

func (a *Agent) searchAllThreads(query string) (string, error) {
	items, err := listThreads()
	if err != nil {
		return "", err
	}
	queryLower := strings.ToLower(query)
	var results []string
	for _, item := range items {
		thread, err := loadThread(item.ID)
		if err != nil {
			continue
		}
		matchCount := 0
		var lastMatch string
		for _, m := range thread.Messages {
			if strings.Contains(strings.ToLower(m.Content), queryLower) {
				matchCount++
				lastMatch = truncateString(m.Content, 200)
			}
		}
		if matchCount > 0 {
			results = append(results, fmt.Sprintf("**Thread: %s** (%s) - %d matches\nLast match: %s", item.Title, item.ID, matchCount, lastMatch))
		}
	}
	if len(results) == 0 {
		return fmt.Sprintf("No matches found for '%s' across any thread.", query), nil
	}
	return fmt.Sprintf("Found matches in %d threads:\n\n%s", len(results), strings.Join(results, "\n\n---\n")), nil
}

func (a *Agent) chatStream(ctx context.Context, cb StreamCallbacks) (*Message, error) {
	// NOTE: compression is NOT called here. It is called once per user message
	// in handleChatStream, before the agent loop starts. This prevents context
	// loss during multi-turn tool-calling loops.

	// Select relevant tools based on conversation context (small models choke on 30+ tools)
	selectedTools := selectToolsForContext(a.messages)

	// Convert tools to OpenAI format
	var toolDefs []map[string]interface{}
	for _, tool := range selectedTools {
		toolDefs = append(toolDefs, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  tool.Parameters,
			},
		})
	}

	streaming := cb.OnContent != nil || cb.OnThinking != nil
	temp := 0.7
	selfMu.RLock()
	if currentSelf != nil {
		temp = currentSelf.Params.Temperature
	}
	selfMu.RUnlock()

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    a.messages,
		Tools:       toolDefs,
		ToolChoice:  "auto",
		MaxTokens:   8192,
		Temperature: temp,
		Stream:      streaming,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		a.config.primaryProvider().Endpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	a.setAuthHeaders(httpReq)

	// No client-level timeout — context cancellation handles it.
	// Local models (Ollama) can take minutes per response depending on hardware.
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("API request failed (is the model server running?): %w\nTry: siki serve <model>", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Handle streaming response
	if streaming {
		return a.handleStreamingResponse(resp.Body, cb)
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no response from model")
	}

	return &chatResp.Choices[0].Message, nil
}

// StreamChoice represents a streaming response choice
type StreamChoice struct {
	Index int `json:"index"`
	Delta struct {
		Role      string     `json:"role,omitempty"`
		Content   string     `json:"content,omitempty"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	} `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// StreamResponse represents a streaming response chunk
type StreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// StreamCallbacks provides callbacks for different types of streaming content.
type StreamCallbacks struct {
	OnContent  func(string) // regular content chunks
	OnThinking func(string) // thinking/reasoning chunks (inside <think> tags)
}

func (a *Agent) handleStreamingResponse(body io.Reader, cb StreamCallbacks) (*Message, error) {
	reader := bufio.NewReader(body)
	var fullContent strings.Builder
	var fullThinking strings.Builder
	var toolCalls []ToolCall
	toolCallArgs := make(map[int]string) // index -> accumulated arguments
	inThink := false                     // tracking <think> state

	// Buffer for detecting partial <think> / </think> tags across chunks
	var pendingBuf string

	flushContent := func(s string) {
		if s == "" {
			return
		}
		fullContent.WriteString(s)
		if cb.OnContent != nil {
			cb.OnContent(s)
		}
	}
	flushThinking := func(s string) {
		if s == "" {
			return
		}
		fullThinking.WriteString(s)
		if cb.OnThinking != nil {
			cb.OnThinking(s)
		}
	}

	var readErr error
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			readErr = err
			break // don't return nil — fall through to return partial content
		}

		line = strings.TrimSpace(line)
		if line == "" || line == "data: [DONE]" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		var streamResp StreamResponse
		if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
			continue
		}

		if len(streamResp.Choices) == 0 {
			continue
		}

		delta := streamResp.Choices[0].Delta

		// Handle content — route to thinking or content based on <think> tags
		if delta.Content != "" {
			chunk := pendingBuf + delta.Content
			pendingBuf = ""

			for len(chunk) > 0 {
				if inThink {
					// Inside <think> — look for </think>
					if idx := strings.Index(chunk, "</think>"); idx != -1 {
						flushThinking(chunk[:idx])
						chunk = chunk[idx+8:]
						inThink = false
					} else if strings.Contains("</think>"[:min(len(chunk), 8)], chunk[max(0, len(chunk)-7):]) {
						// Possible partial </think> at end of chunk
						pendingBuf = chunk
						chunk = ""
					} else {
						flushThinking(chunk)
						chunk = ""
					}
				} else {
					// Outside <think> — look for <think>
					if idx := strings.Index(chunk, "<think>"); idx != -1 {
						flushContent(chunk[:idx])
						chunk = chunk[idx+7:]
						inThink = true
					} else if len(chunk) >= 1 && strings.HasPrefix("<think>", chunk[max(0, len(chunk)-6):]) {
						// Possible partial <think> at end of chunk
						pendingBuf = chunk
						chunk = ""
					} else {
						flushContent(chunk)
						chunk = ""
					}
				}
			}
		}

		// Handle tool calls
		for _, tc := range delta.ToolCalls {
			idx := tc.Index
			if idx >= len(toolCalls) {
				// New tool call
				toolCalls = append(toolCalls, ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: ToolCallFunc{
						Name:      tc.Function.Name,
						Arguments: "",
					},
				})
				toolCallArgs[idx] = ""
			}
			// Accumulate arguments
			if tc.Function.Arguments != "" {
				toolCallArgs[idx] += tc.Function.Arguments
			}
			if tc.Function.Name != "" {
				toolCalls[idx].Function.Name = tc.Function.Name
			}
			if tc.ID != "" {
				toolCalls[idx].ID = tc.ID
			}
		}
	}

	// Flush any remaining pending buffer
	if pendingBuf != "" {
		if inThink {
			flushThinking(pendingBuf)
		} else {
			flushContent(pendingBuf)
		}
	}

	// Finalize tool call arguments
	for idx, args := range toolCallArgs {
		if idx < len(toolCalls) {
			toolCalls[idx].Function.Arguments = args
		}
	}

	return &Message{
		Role:      "assistant",
		Content:   fullContent.String(),
		Thinking:  fullThinking.String(),
		ToolCalls: toolCalls,
	}, readErr
}

// ============================================================================
// Agent Loop
// ============================================================================

func (a *Agent) run(ctx context.Context, userInput string) error {
	// Add user message
	a.messages = append(a.messages, Message{
		Role:    "user",
		Content: userInput,
	})

	for turn := 0; turn < a.config.MaxTurns; turn++ {
		// Get response from LLM
		response, err := a.chat(ctx)
		if err != nil {
			return err
		}

		// Add assistant message
		a.messages = append(a.messages, *response)

		// Check for tool calls
		if len(response.ToolCalls) == 0 {
			// No tool calls, print response and return
			if response.Content != "" {
				fmt.Printf("\n%s\n", response.Content)
			}
			return nil
		}

		// Execute tool calls
		for _, tc := range response.ToolCalls {
			fmt.Printf("[Tool: %s]\n", tc.Function.Name)

			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				a.messages = append(a.messages, Message{
					Role:       "tool",
					Content:    fmt.Sprintf("Error parsing arguments: %v", err),
					ToolCallID: tc.ID,
				})
				continue
			}

			result, err := a.executeTool(tc.Function.Name, args)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			// Truncate very long results
			if len(result) > 10000 {
				result = result[:10000] + "\n... (truncated)"
			}

			a.messages = append(a.messages, Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	return fmt.Errorf("max turns (%d) exceeded", a.config.MaxTurns)
}

// ============================================================================
// Model Management
// ============================================================================

type ModelManager struct {
	config     *Config
	serverProc *os.Process
}

func NewModelManager(config *Config) *ModelManager {
	return &ModelManager{config: config}
}

func (m *ModelManager) DownloadModel(modelID string) error {
	fmt.Printf("Downloading model: %s\n", modelID)

	// Create models directory
	if err := os.MkdirAll(m.config.ModelPath, 0755); err != nil {
		return err
	}

	modelDir := filepath.Join(m.config.ModelPath, strings.ReplaceAll(modelID, "/", "_"))

	// Use huggingface-cli if available, otherwise use git
	if _, err := exec.LookPath("huggingface-cli"); err == nil {
		cmd := exec.Command("huggingface-cli", "download", modelID,
			"--local-dir", modelDir)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		return cmd.Run()
	}

	// Fallback to git clone
	repoURL := fmt.Sprintf("https://huggingface.co/%s", modelID)
	cmd := exec.Command("git", "clone", "--depth", "1", repoURL, modelDir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (m *ModelManager) ListModels() ([]string, error) {
	entries, err := os.ReadDir(m.config.ModelPath)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			models = append(models, entry.Name())
		}
	}
	return models, nil
}

func (m *ModelManager) StartServer(modelPath string) error {
	var cmd *exec.Cmd

	switch m.config.Backend {
	case "vllm":
		fmt.Println("Starting vllm server...")
		cmd = exec.Command("python", "-m", "vllm.entrypoints.openai.api_server",
			"--model", modelPath,
			"--port", fmt.Sprintf("%d", DefaultPort),
			"--trust-remote-code")

	case "mlx":
		fmt.Println("Starting mlx-lm server...")
		cmd = exec.Command("python", "-m", "mlx_lm.server",
			"--model", modelPath,
			"--port", fmt.Sprintf("%d", DefaultPort))

	case "ollama":
		fmt.Println("Starting ollama server...")
		// For ollama, we need to pull the model first
		pullCmd := exec.Command("ollama", "pull", modelPath)
		pullCmd.Stdout = os.Stdout
		pullCmd.Stderr = os.Stderr
		if err := pullCmd.Run(); err != nil {
			return fmt.Errorf("failed to pull model: %w", err)
		}

		cmd = exec.Command("ollama", "serve")

	default:
		return fmt.Errorf("unknown backend: %s", m.config.Backend)
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return err
	}

	m.serverProc = cmd.Process

	// Wait for server to be ready
	fmt.Println("Waiting for server to be ready...")
	for i := 0; i < 60; i++ {
		resp, err := http.Get(m.config.APIEndpoint + "/models")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				fmt.Println("Server is ready!")
				return nil
			}
		}
		time.Sleep(time.Second)
	}

	return fmt.Errorf("server did not start within 60 seconds")
}

func (m *ModelManager) StopServer() error {
	if m.serverProc != nil {
		return m.serverProc.Kill()
	}
	return nil
}

// ============================================================================
// CLI Interface
// ============================================================================

func printHelp() {
	fmt.Printf(`siki v%s - 式神 (Shikigami) Agentic AI

Usage:
  siki [command] [options]

Commands:
  chat                  Start interactive chat session (CLI)
  web                   Start web GUI (browser-based)
  download <model>      Download model from HuggingFace
  list                  List downloaded models
  serve <model>         Start model server
  config                Show current configuration
  quickstart            Download recommended model and start chatting

Options:
  --backend <backend>          Set LLM backend (ollama, vllm, mlx, openai, anthropic, gemini)
  --model <name>               Set model name
  --endpoint <url>             Set API endpoint
  --api-key <key>              Set API key for cloud backends
  --workspace <path>           Set workspace directory
  --host <ip>                  Set web server host (default: 127.0.0.1, use 0.0.0.0 for network)
  --port <port>                Set web server port (default: 3000)

Examples:
  siki web                                     # Start web GUI
  siki quickstart                              # Quick setup with recommended model
  siki download cyberagent/gpt-oss-20b         # Download specific model
  siki serve gpt-oss-20b                       # Start model server
  siki chat --model gpt-oss-20b                # Chat with model (CLI)

Supported Backends:
  vllm   - NVIDIA GPU (CUDA) - Best for Linux servers with GPU
  mlx    - Apple Silicon (Metal) - Best for M1/M2/M3 Macs
  ollama - Universal backend - Works on any platform

`, Version)
}

func runQuickstart(config *Config) error {
	fmt.Println("siki - Quickstart")
	fmt.Println("==================")
	fmt.Printf("Detected backend: %s\n\n", config.Backend)

	var modelID string
	switch config.Backend {
	case "ollama":
		// Use a model that supports tool calling on ollama
		modelID = "llama3.2"
		fmt.Printf("Installing model via ollama: %s\n", modelID)
		cmd := exec.Command("ollama", "pull", modelID)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to pull model: %w", err)
		}
		config.ModelName = modelID
	case "mlx":
		// Use a smaller model for mlx
		modelID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
		fmt.Printf("Downloading model: %s\n", modelID)
		mm := NewModelManager(config)
		if err := mm.DownloadModel(modelID); err != nil {
			return fmt.Errorf("failed to download model: %w", err)
		}
		config.ModelName = "Llama-3.2-3B-Instruct-4bit"
	case "vllm":
		modelID = "cyberagent/gpt-oss-20b"
		fmt.Printf("Downloading model: %s\n", modelID)
		mm := NewModelManager(config)
		if err := mm.DownloadModel(modelID); err != nil {
			return fmt.Errorf("failed to download model: %w", err)
		}
		config.ModelName = "gpt-oss-20b"
	}

	fmt.Println("\nQuickstart complete!")
	fmt.Println("\nNext steps:")
	fmt.Printf("1. Start the server: siki serve %s\n", config.ModelName)
	fmt.Printf("2. In another terminal: siki chat --model %s\n", config.ModelName)
	return nil
}

// ============================================================================
// Web GUI Server
// ============================================================================

type WebServer struct {
	config        *Config
	conversations map[string]*Agent
	mu            sync.RWMutex
	lastActivity  time.Time
	improveMu     sync.Mutex
}

type ChatAPIRequest struct {
	Message        string   `json:"message"`
	ConversationID string   `json:"conversation_id"`
	Images         []string `json:"images,omitempty"`
}

type ToolCallResult struct {
	Name   string `json:"name"`
	Result string `json:"result"`
}

type ChatAPIResponse struct {
	Response  string           `json:"response,omitempty"`
	ToolCalls []ToolCallResult `json:"tool_calls,omitempty"`
	Error     string           `json:"error,omitempty"`
}

type StatusResponse struct {
	Model            string     `json:"model"`
	Backend          string     `json:"backend"`
	Endpoint         string     `json:"endpoint"`
	Version          string     `json:"version"`
	HasAPIKey        bool       `json:"has_api_key"`
	Providers        []Provider `json:"providers"`
	DockerAvailable  bool       `json:"docker_available"`
	DockerImageReady bool       `json:"docker_image_ready"`
	VisionModel      string     `json:"vision_model,omitempty"`
}

type SettingsRequest struct {
	// Action: "set_providers" to replace entire providers list
	Action    string     `json:"action"`
	Providers []Provider `json:"providers,omitempty"`
	// Legacy single-provider fields (backward compat)
	Model    string `json:"model"`
	Backend  string `json:"backend"`
	Endpoint string `json:"endpoint"`
	APIKey   string `json:"api_key"`
}

func NewWebServer(config *Config) *WebServer {
	return &WebServer{
		config:        config,
		conversations: make(map[string]*Agent),
		lastActivity:  time.Now(),
	}
}

func buildSystemPrompt(config *Config) string {
	now := time.Now()
	dateStr := fmt.Sprintf("今日は%d年%d月%d日です。", now.Year(), int(now.Month()), now.Day())

	var sb strings.Builder
	sb.WriteString(dateStr)
	sb.WriteString("\n\n")

	// Use self-state prompt if available, otherwise config default
	selfMu.RLock()
	if currentSelf != nil && currentSelf.Prompt != "" {
		sb.WriteString(currentSelf.Prompt)

		// Inject self-rules
		var activeRules []SelfRule
		for _, r := range currentSelf.Rules {
			if r.Active {
				activeRules = append(activeRules, r)
			}
		}
		if len(activeRules) > 0 {
			sb.WriteString("\n\n## Self-Rules (自己ルール)\n")
			for _, r := range activeRules {
				sb.WriteString(fmt.Sprintf("- %s\n", r.Rule))
			}
		}
	} else {
		sb.WriteString(config.SystemPrompt)
	}
	selfMu.RUnlock()

	// Inject immutable self-modification instructions (kernel)
	sb.WriteString(`

## Self-Modification (自己改変)
You can modify your own behavior:
- self_status: View current version, parameters, rules, benchmark score
- self_modify_prompt: Edit your system prompt (auto-snapshot before changes)
- self_modify_params: Adjust behavioral parameters (temperature, max_turns, etc.)
- self_add_rule / self_remove_rule: Manage self-imposed rules
- self_rollback: Revert to a previous version (version 0 = factory default)
- self_benchmark: Run self-evaluation benchmarks

## Self-Evolution (自己進化)
You can modify your own Go source code:
- self_evolve action=view_source: Read your own source code
- self_evolve action=patch: Apply code changes (old_text → new_text replacement)
- self_evolve action=build_test: Compile and run all tests
- self_evolve action=deploy: Deploy tested binary and restart process
- self_evolve action=abort: Revert all patches

Evolution workflow:
1. view_source to understand the code
2. patch to apply changes (one at a time, small and focused)
3. build_test to verify (MUST pass all tests)
4. deploy to restart with new code

When modifying yourself:
1. Run self_benchmark BEFORE and AFTER changes
2. If score drops, consider rollback
3. Document your reason for every change
4. Prefer small, incremental changes
5. For code evolution: always build_test before deploy
`)

	// Inject ACE playbook context
	sb.WriteString(buildPlaybookContext())

	// Inject indexed documents summary
	if docs, err := listDocuments(); err == nil && len(docs) > 0 {
		sb.WriteString("\n\n## Indexed Documents\n")
		sb.WriteString("search_documentツールで検索可能なドキュメント:\n")
		for _, doc := range docs {
			sb.WriteString(fmt.Sprintf("- %s (ID: %s)\n", doc.Title, doc.ID))
		}
	}

	// Inject installed plugins into system prompt
	pluginMu.RLock()
	if len(loadedPlugins) > 0 {
		sb.WriteString("\n\n## Installed Plugins\n")
		sb.WriteString("Before responding, check if any installed plugin can handle the user's request. If a matching enabled plugin exists, use it.\n\n")
		for _, p := range loadedPlugins {
			status := "ON"
			if !p.IsEnabled() {
				status = "OFF"
			}
			testBadge := "UNTESTED"
			if p.Tested {
				testBadge = "TESTED"
			}
			sb.WriteString(fmt.Sprintf("- **%s** [%s][%s] (v%s): %s", p.Name, status, testBadge, p.Version, p.Description))
			if p.Tool != nil {
				sb.WriteString(fmt.Sprintf(" → tool: `plugin_%s` - %s", p.Name, p.Tool.Description))
			}
			if p.UI != nil {
				sb.WriteString(" (has UI)")
			}
			sb.WriteString("\n")
		}
		sb.WriteString("\nOFF状態のプラグインは使用不可。ユーザーが必要なら設定からONにするよう案内すること。\n")
		sb.WriteString("UNTESTED状態のプラグインは信頼性が未検証。\n")
	}
	pluginMu.RUnlock()

	// Inject configured providers
	if len(config.Providers) > 1 {
		sb.WriteString("\n\n## Configured Providers\n")
		sb.WriteString("Use `query_model` tool to query other providers.\n\n")
		for i, p := range config.Providers {
			primary := ""
			if i == 0 {
				primary = " [PRIMARY]"
			}
			sb.WriteString(fmt.Sprintf("- **%s**%s: %s / %s\n", p.Name, primary, p.Backend, p.Model))
		}
	}

	return sb.String()
}

func (ws *WebServer) getOrCreateAgent(convID string) *Agent {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if agent, exists := ws.conversations[convID]; exists {
		return agent
	}

	agent := &Agent{
		config:   ws.config,
		threadID: convID,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(ws.config)},
		},
	}

	// Build LLM context from thread log (log file itself is never modified)
	thread, err := loadThread(convID)
	if err == nil && len(thread.Messages) > 0 {
		const recentCount = 60
		// Filter out summarization markers
		var contextMsgs []ThreadMessage
		for _, tm := range thread.Messages {
			if tm.Summarized {
				continue
			}
			// Skip legacy [tool_calls:] text markers (old format without ToolCalls field)
			if tm.Role == "assistant" && strings.HasPrefix(tm.Content, "[tool_calls:") && len(tm.ToolCalls) == 0 {
				continue
			}
			contextMsgs = append(contextMsgs, tm)
		}

		if len(contextMsgs) <= recentCount {
			// Few enough messages — load all directly
			for _, tm := range contextMsgs {
				agent.messages = append(agent.messages, Message{
					Role: tm.Role, Content: tm.Content,
					Images: tm.Images, ToolCalls: tm.ToolCalls, ToolCallID: tm.ToolCallID,
				})
			}
		} else {
			// Many messages — build plain-text digest of older messages for LLM context
			// (this digest is ephemeral; it is NOT saved to the thread log)
			olderMsgs := contextMsgs[:len(contextMsgs)-recentCount]
			var digest strings.Builder
			for _, tm := range olderMsgs {
				if tm.Role == "tool" {
					// Include tool results in digest with tool name
					toolLabel := tm.ToolName
					if toolLabel == "" {
						toolLabel = "tool"
					}
					line := truncateString(tm.Content, 800)
					digest.WriteString(fmt.Sprintf("[tool result (%s)]: %s\n", toolLabel, line))
					continue
				}
				if tm.Role == "assistant" && len(tm.ToolCalls) > 0 && tm.Content == "" {
					// Summarize tool calls in digest
					var names []string
					for _, tc := range tm.ToolCalls {
						names = append(names, tc.Function.Name)
					}
					digest.WriteString(fmt.Sprintf("[assistant → tool calls: %s]\n", strings.Join(names, ", ")))
					continue
				}
				line := truncateString(tm.Content, 800)
				digest.WriteString(fmt.Sprintf("[%s]: %s\n", tm.Role, line))
			}
			digestText := digest.String()
			if len(digestText) > 20000 {
				digestText = digestText[:20000] + "\n...(older messages omitted)"
			}
			if digestText != "" {
				agent.messages = append(agent.messages, Message{
					Role:    "assistant",
					Content: fmt.Sprintf("[過去の会話ログ (%d件) — 詳細は recall_context ツールで検索可能]\n%s", len(olderMsgs), digestText),
				})
			}

			// Load recent messages directly (with proper ToolCalls for API compatibility)
			recentMsgs := contextMsgs[len(contextMsgs)-recentCount:]
			for _, tm := range recentMsgs {
				agent.messages = append(agent.messages, Message{
					Role: tm.Role, Content: tm.Content,
					Images: tm.Images, ToolCalls: tm.ToolCalls, ToolCallID: tm.ToolCallID,
				})
			}
		}
	}

	// Fix incomplete tool call sequences: if the last assistant message has
	// tool_calls but not all tool results are present, add placeholder results.
	// This can happen when a network error interrupts the agent loop mid-execution.
	agent.messages = fixIncompleteToolCalls(agent.messages)

	ws.conversations[convID] = agent
	return agent
}

// fixIncompleteToolCalls ensures that every assistant tool_call has a matching
// tool result message. Missing results (from interrupted sessions) get a
// placeholder inserted right after the corresponding tool results.
// This prevents the LLM API from rejecting incomplete message sequences.
func fixIncompleteToolCalls(msgs []Message) []Message {
	if len(msgs) == 0 {
		return msgs
	}

	// Build a new message list, inserting placeholders where needed
	var result []Message
	for i := 0; i < len(msgs); i++ {
		result = append(result, msgs[i])

		if msgs[i].Role != "assistant" || len(msgs[i].ToolCalls) == 0 {
			continue
		}

		// This assistant message has tool_calls — check which results exist
		needed := make(map[string]string) // id -> function name
		for _, tc := range msgs[i].ToolCalls {
			needed[tc.ID] = tc.Function.Name
		}

		// Consume consecutive tool result messages
		j := i + 1
		for j < len(msgs) && msgs[j].Role == "tool" {
			if msgs[j].ToolCallID != "" {
				delete(needed, msgs[j].ToolCallID)
			}
			result = append(result, msgs[j])
			j++
		}

		// Insert placeholders for any missing tool results
		for id, name := range needed {
			result = append(result, Message{
				Role:       "tool",
				Content:    fmt.Sprintf("[%s の結果は取得できませんでした（ネットワークエラー等で中断）]", name),
				ToolCallID: id,
			})
		}

		// Skip the tool messages we already consumed
		i = j - 1
	}
	return result
}

func (ws *WebServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	content, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(content)
}

func (ws *WebServer) handleDiagrams(w http.ResponseWriter, r *http.Request) {
	// Extract filename from path
	filename := strings.TrimPrefix(r.URL.Path, "/diagrams/")
	if filename == "" || strings.Contains(filename, "..") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	// Only allow .svg files
	if !strings.HasSuffix(filename, ".svg") {
		http.Error(w, "Invalid file type", http.StatusBadRequest)
		return
	}

	filePath := filepath.Join(diagramDir, filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		http.Error(w, "Diagram not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "image/svg+xml")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Header().Set("Cache-Control", "public, max-age=3600")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Write(data)
}

func (ws *WebServer) handlePlayground(w http.ResponseWriter, r *http.Request) {
	// Extract filename from path
	filename := strings.TrimPrefix(r.URL.Path, "/playground/")
	if filename == "" || strings.Contains(filename, "..") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}

	// Only allow .html files
	if !strings.HasSuffix(filename, ".html") {
		http.Error(w, "Invalid file type", http.StatusBadRequest)
		return
	}

	filePath := filepath.Join(playgroundDir, filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		http.Error(w, "Playground not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Write(data)
}

func (ws *WebServer) handlePluginsUI(w http.ResponseWriter, r *http.Request) {
	pluginMu.RLock()
	defer pluginMu.RUnlock()

	type UIPayload struct {
		Name string `json:"name"`
		JS   string `json:"js"`
		CSS  string `json:"css"`
	}

	var uis []UIPayload
	for _, p := range loadedPlugins {
		if p.IsEnabled() && p.UI != nil && (p.UI.JS != "" || p.UI.CSS != "") {
			uis = append(uis, UIPayload{
				Name: p.Name,
				JS:   p.UI.JS,
				CSS:  p.UI.CSS,
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(uis)
}

func (ws *WebServer) handlePlugins(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		// List all plugins with full info
		pluginMu.RLock()
		defer pluginMu.RUnlock()

		type PluginInfo struct {
			Name        string `json:"name"`
			Description string `json:"description"`
			Version     string `json:"version"`
			Enabled     bool   `json:"enabled"`
			Tested      bool   `json:"tested"`
			TestResult  string `json:"test_result,omitempty"`
			HasTool     bool   `json:"has_tool"`
			HasUI       bool   `json:"has_ui"`
		}

		list := make([]PluginInfo, 0)
		for _, p := range loadedPlugins {
			list = append(list, PluginInfo{
				Name:        p.Name,
				Description: p.Description,
				Version:     p.Version,
				Enabled:     p.IsEnabled(),
				Tested:      p.Tested,
				TestResult:  p.TestResult,
				HasTool:     p.Tool != nil,
				HasUI:       p.UI != nil && (p.UI.JS != "" || p.UI.CSS != ""),
			})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(list)

	case http.MethodPost:
		// Toggle plugin enabled/disabled
		var req struct {
			Name    string `json:"name"`
			Enabled bool   `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		pluginMu.Lock()
		found := false
		for i := range loadedPlugins {
			if loadedPlugins[i].Name == req.Name {
				loadedPlugins[i].Enabled = &req.Enabled
				// Persist to disk
				if err := savePlugin(loadedPlugins[i]); err != nil {
					pluginMu.Unlock()
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				found = true
				break
			}
		}
		pluginMu.Unlock()

		if !found {
			http.Error(w, "plugin not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleImages(w http.ResponseWriter, r *http.Request) {
	targetURL := r.URL.Query().Get("url")
	if targetURL == "" {
		http.Error(w, "url parameter required", http.StatusBadRequest)
		return
	}

	agent := &Agent{config: ws.config}
	result, err := agent.webImages(targetURL)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"images": []string{},
			"error":  err.Error(),
		})
		return
	}

	// Parse image URLs from the result
	var images []string
	imgRe := regexp.MustCompile(`!\[.*?\]\(([^)]+)\)`)
	matches := imgRe.FindAllStringSubmatch(result, -1)
	for _, m := range matches {
		if len(m) > 1 {
			images = append(images, m[1])
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"images": images,
	})
}

func (ws *WebServer) handleThreads(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	path := r.URL.Path
	// Strip base path
	trimmed := strings.TrimPrefix(path, "/api/threads")
	trimmed = strings.TrimPrefix(trimmed, "/")

	if trimmed == "" {
		// /api/threads
		switch r.Method {
		case http.MethodGet:
			// List all threads
			items, err := listThreads()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if items == nil {
				items = []ThreadListItem{}
			}
			json.NewEncoder(w).Encode(items)
		case http.MethodPost:
			// Create new thread
			var req struct {
				ID    string `json:"id"`
				Title string `json:"title"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if req.Title == "" {
				req.Title = "New conversation"
			}
			id := req.ID
			if id == "" {
				id = fmt.Sprintf("%d", time.Now().UnixMilli())
			}
			t := &Thread{
				ID:        id,
				Title:     req.Title,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
			}
			if err := saveThread(t); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			json.NewEncoder(w).Encode(t)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
		return
	}

	// /api/threads/search?q=...
	if trimmed == "search" {
		query := r.URL.Query().Get("q")
		if query == "" {
			json.NewEncoder(w).Encode([]interface{}{})
			return
		}
		queryLower := strings.ToLower(query)
		items, err := listThreads()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		type SearchResult struct {
			ThreadID     string `json:"thread_id"`
			ThreadTitle  string `json:"thread_title"`
			MatchCount   int    `json:"match_count"`
			Snippet      string `json:"snippet"`
			SnippetRole  string `json:"snippet_role"`
			MessageCount int    `json:"message_count"`
		}
		var results []SearchResult
		for _, item := range items {
			// Check title match
			titleMatch := strings.Contains(strings.ToLower(item.Title), queryLower)

			thread, err := loadThread(item.ID)
			if err != nil {
				if titleMatch {
					results = append(results, SearchResult{
						ThreadID: item.ID, ThreadTitle: item.Title,
						MatchCount: 1, Snippet: item.Title, SnippetRole: "title",
						MessageCount: item.MessageCount,
					})
				}
				continue
			}

			matchCount := 0
			var bestSnippet string
			var bestRole string
			for _, m := range thread.Messages {
				if strings.Contains(strings.ToLower(m.Content), queryLower) {
					matchCount++
					// Extract snippet around the match
					lower := strings.ToLower(m.Content)
					idx := strings.Index(lower, queryLower)
					start := idx - 40
					if start < 0 {
						start = 0
					}
					end := idx + len(query) + 80
					if end > len(m.Content) {
						end = len(m.Content)
					}
					snippet := m.Content[start:end]
					if start > 0 {
						snippet = "..." + snippet
					}
					if end < len(m.Content) {
						snippet = snippet + "..."
					}
					bestSnippet = snippet
					bestRole = m.Role
				}
			}
			if titleMatch {
				matchCount++
			}
			if matchCount > 0 {
				if bestSnippet == "" {
					bestSnippet = item.Title
					bestRole = "title"
				}
				results = append(results, SearchResult{
					ThreadID: item.ID, ThreadTitle: item.Title,
					MatchCount: matchCount, Snippet: bestSnippet, SnippetRole: bestRole,
					MessageCount: item.MessageCount,
				})
			}
		}
		if results == nil {
			results = []SearchResult{}
		}
		json.NewEncoder(w).Encode(results)
		return
	}

	// Extract thread ID and possible sub-path
	parts := strings.SplitN(trimmed, "/", 2)
	threadID := parts[0]
	subPath := ""
	if len(parts) > 1 {
		subPath = parts[1]
	}

	switch subPath {
	case "":
		// /api/threads/{id}
		switch r.Method {
		case http.MethodGet:
			t, err := loadThread(threadID)
			if err != nil {
				http.Error(w, "Thread not found", http.StatusNotFound)
				return
			}
			json.NewEncoder(w).Encode(t)
		case http.MethodDelete:
			if err := deleteThread(threadID); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			// Also remove from in-memory cache
			ws.mu.Lock()
			delete(ws.conversations, threadID)
			ws.mu.Unlock()
			json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	case "rename":
		// /api/threads/{id}/rename
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Title string `json:"title"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		t, err := loadThreadMeta(threadID)
		if err != nil {
			http.Error(w, "Thread not found", http.StatusNotFound)
			return
		}
		t.Title = req.Title
		t.UpdatedAt = time.Now()
		if err := saveThreadMeta(t); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(t)
	default:
		http.Error(w, "Not found", http.StatusNotFound)
	}
}

func (ws *WebServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	pp := ws.config.primaryProvider()
	resp := StatusResponse{
		Model:            pp.Model,
		Backend:          pp.Backend,
		Endpoint:         pp.Endpoint,
		Version:          Version,
		HasAPIKey:        pp.APIKey != "",
		Providers:        ws.config.Providers,
		DockerAvailable:  isDockerAvailable(),
		DockerImageReady: isDockerAvailable() && isDockerImageAvailable(),
		VisionModel:      ws.config.VisionModel,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (ws *WebServer) handleSettings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SettingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ws.mu.Lock()
	switch req.Action {
	case "set_providers":
		// Replace entire providers list
		ws.config.Providers = req.Providers
		// Sync legacy fields from primary provider
		if len(ws.config.Providers) > 0 {
			pp := ws.config.Providers[0]
			ws.config.Backend = pp.Backend
			ws.config.ModelName = pp.Model
			ws.config.APIEndpoint = pp.Endpoint
			ws.config.APIKey = pp.APIKey
		}
	default:
		// Legacy single-provider update
		if req.Model != "" {
			ws.config.ModelName = req.Model
		}
		if req.Backend != "" {
			ws.config.Backend = req.Backend
		}
		if req.Endpoint != "" {
			ws.config.APIEndpoint = req.Endpoint
		}
		ws.config.APIKey = req.APIKey
		// Also update providers[0] if it exists
		if len(ws.config.Providers) > 0 {
			if req.Model != "" {
				ws.config.Providers[0].Model = req.Model
			}
			if req.Backend != "" {
				ws.config.Providers[0].Backend = req.Backend
			}
			if req.Endpoint != "" {
				ws.config.Providers[0].Endpoint = req.Endpoint
			}
			ws.config.Providers[0].APIKey = req.APIKey
		}
	}
	ws.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (ws *WebServer) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	agent := ws.getOrCreateAgent(req.ConversationID)
	threadID := req.ConversationID
	saveMsg := func(msg Message, toolName string) {
		appendMessageToThread(threadID, msg, toolName)
	}

	// If images are present and a vision model is configured, convert to text
	userContent := req.Message
	userImages := req.Images
	if len(req.Images) > 0 && ws.config.VisionModel != "" {
		endpoint := ws.config.primaryProvider().Endpoint
		imageDesc := describeImages(req.Images, ws.config.VisionModel, endpoint)
		if imageDesc != "" {
			if userContent != "" {
				userContent = userContent + "\n\n" + imageDesc
			} else {
				userContent = imageDesc
			}
		}
		userImages = nil
	}

	userMsg := Message{
		Role:    "user",
		Content: userContent,
		Images:  userImages,
	}
	agent.messages = append(agent.messages, userMsg)
	logMsg := Message{
		Role:    "user",
		Content: userContent,
		Images:  req.Images,
	}
	saveMsg(logMsg, "")

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	var apiResp ChatAPIResponse

	// Agent loop
	for turn := 0; turn < ws.config.MaxTurns; turn++ {
		response, err := agent.chat(ctx)
		if err != nil {
			apiResp.Error = err.Error()
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(apiResp)
			return
		}

		agent.messages = append(agent.messages, *response)
		saveMsg(*response, "")

		if len(response.ToolCalls) == 0 {
			apiResp.Response = response.Content
			break
		}

		// Execute tool calls
		for _, tc := range response.ToolCalls {
			toolName := tc.Function.Name

			// Redirect: diagram can't handle complex visualizations
			if toolName == "diagram" {
				userReq := agent.lastUserMessage()
				if needsRunCode(userReq) {
					fmt.Printf("[siki] Redirecting diagram → run_code for: %s\n", userReq)
					toolName = "run_code"
				}
			}

			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				toolMsg := Message{
					Role:       "tool",
					Content:    fmt.Sprintf("Error parsing arguments: %v", err),
					ToolCallID: tc.ID,
				}
				agent.messages = append(agent.messages, toolMsg)
				saveMsg(toolMsg, toolName)
				apiResp.ToolCalls = append(apiResp.ToolCalls, ToolCallResult{
					Name:   toolName,
					Result: fmt.Sprintf("Error: %v", err),
				})
				continue
			}

			// For code generation tools, delegate to sub-model
			if toolName == "run_code" {
				userReq := agent.lastUserMessage()
				if html, genErr := generateCodeWithSubModel(userReq, ws.config); genErr == nil {
					args["html"] = html
				} else {
					fmt.Printf("[siki] Sub-model code gen failed: %v, using orchestrator HTML\n", genErr)
				}
			}

			result, err := agent.executeTool(toolName, args)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			displayResult := result
			if len(displayResult) > 2000 {
				displayResult = displayResult[:2000] + "\n... (truncated)"
			}

			apiResp.ToolCalls = append(apiResp.ToolCalls, ToolCallResult{
				Name:   toolName,
				Result: displayResult,
			})

			toolMsg := Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			}
			agent.messages = append(agent.messages, toolMsg)
			saveMsg(toolMsg, tc.Function.Name)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(apiResp)
}

// StreamEvent represents a server-sent event
type StreamEvent struct {
	Type    string `json:"type"`
	Content string `json:"content,omitempty"`
	Name    string `json:"name,omitempty"`
	Result  string `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
}

func (ws *WebServer) handleChatStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 50*1024*1024) // 50MB max for image uploads
	var req ChatAPIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Set headers for SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	sendEvent := func(event StreamEvent) {
		data, _ := json.Marshal(event)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Track activity for self-improvement idle detection
	ws.mu.Lock()
	ws.lastActivity = time.Now()
	ws.mu.Unlock()

	agent := ws.getOrCreateAgent(req.ConversationID)

	// Helper: save a message to thread log immediately
	threadID := req.ConversationID
	saveMsg := func(msg Message, toolName string) {
		appendMessageToThread(threadID, msg, toolName)
	}

	// Check if this is the first user message (for title generation later)
	isFirstMessage := false
	if meta, err := loadThreadMeta(threadID); err != nil || meta.MessageCount == 0 {
		isFirstMessage = true
	}

	// If images are present and a vision model is configured, convert images to text
	userContent := req.Message
	userImages := req.Images
	if len(req.Images) > 0 && ws.config.VisionModel != "" {
		sendEvent(StreamEvent{Type: "thinking", Content: "画像を解析中..."})
		endpoint := ws.config.primaryProvider().Endpoint
		imageDesc := describeImages(req.Images, ws.config.VisionModel, endpoint)
		if imageDesc != "" {
			if userContent != "" {
				userContent = userContent + "\n\n" + imageDesc
			} else {
				userContent = imageDesc
			}
		}
		// Clear images from LLM message since we converted them to text
		userImages = nil
	}

	userMsg := Message{
		Role:    "user",
		Content: userContent,
		Images:  userImages,
	}
	agent.messages = append(agent.messages, userMsg)
	// Save original message with images to log (for record keeping)
	logMsg := Message{
		Role:    "user",
		Content: userContent,
		Images:  req.Images,
	}
	saveMsg(logMsg, "")

	// Compress once before the agent loop (not inside chatStream)
	// This prevents context loss during multi-turn tool calling
	{
		compressCtx, compressCancel := context.WithTimeout(context.Background(), 30*time.Second)
		agent.compressConversation(compressCtx)
		compressCancel()
	}

	var lastAssistantReply string

	// 10 minutes for the full agent loop — local models can be very slow
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	var hitTimeout bool

	// Agent loop: stream content + tool events, save each message in real-time
	for turn := 0; turn < ws.config.MaxTurns; turn++ {
		response, err := agent.chatStream(ctx, StreamCallbacks{
			OnContent: func(content string) {
				sendEvent(StreamEvent{Type: "content", Content: content})
			},
			OnThinking: func(thinking string) {
				sendEvent(StreamEvent{Type: "thinking", Content: thinking})
			},
		})

		if err != nil {
			// Save partial response if any content was received
			if response != nil && (response.Content != "" || len(response.ToolCalls) > 0) {
				agent.messages = append(agent.messages, *response)
				saveMsg(*response, "")
				// Add placeholder results for any tool_calls in the partial response
				for _, tc := range response.ToolCalls {
					placeholder := Message{
						Role:       "tool",
						Content:    fmt.Sprintf("[%s の結果は取得できませんでした（ネットワークエラー等で中断）]", tc.Function.Name),
						ToolCallID: tc.ID,
					}
					agent.messages = append(agent.messages, placeholder)
					saveMsg(placeholder, tc.Function.Name)
				}
			}
			if strings.Contains(err.Error(), "deadline exceeded") || strings.Contains(err.Error(), "context canceled") {
				hitTimeout = true
				break
			}
			sendEvent(StreamEvent{Type: "error", Error: err.Error()})
			cancel()
			break // break instead of return — still do title generation & done event
		}

		agent.messages = append(agent.messages, *response)
		saveMsg(*response, "")

		if len(response.ToolCalls) == 0 {
			// Fallback: if model didn't call tools but should have, auto-call
			if turn == 0 {
				if fallbackResult := autoToolFallback(agent, req.Message, response.Content, sendEvent, saveMsg); fallbackResult != "" {
					lastAssistantReply = fallbackResult
					break
				}
			}
			lastAssistantReply = response.Content
			break
		}

		// Execute tool calls
		for _, tc := range response.ToolCalls {
			toolName := tc.Function.Name

			// Redirect: diagram can't handle complex visualizations — use run_code instead
			if toolName == "diagram" {
				userReq := agent.lastUserMessage()
				if needsRunCode(userReq) {
					fmt.Printf("[siki] Redirecting diagram → run_code for: %s\n", userReq)
					toolName = "run_code"
				}
			}

			// Send tool_start event so frontend shows spinner
			sendEvent(StreamEvent{Type: "tool_start", Name: toolName})

			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				result := fmt.Sprintf("Error parsing arguments: %v", err)
				sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: result})
				toolMsg := Message{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
				}
				agent.messages = append(agent.messages, toolMsg)
				saveMsg(toolMsg, toolName)
				continue
			}

			// For code generation tools, delegate to sub-model for higher quality
			if toolName == "run_code" {
				userReq := agent.lastUserMessage()
				if html, genErr := generateCodeWithSubModel(userReq, ws.config); genErr == nil {
					args["html"] = html
				} else {
					fmt.Printf("[siki] Sub-model code gen failed: %v, using orchestrator HTML\n", genErr)
				}
			}

			result, err := agent.executeTool(toolName, args)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			displayResult := result
			if len(displayResult) > 2000 {
				displayResult = displayResult[:2000] + "\n... (truncated)"
			}

			sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: displayResult})

			toolMsg := Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			}
			agent.messages = append(agent.messages, toolMsg)
			saveMsg(toolMsg, toolName)

			// ACE Reflector: extract insights from tool execution (async, non-blocking)
			go func(toolName, toolArgs, toolResult, userQuery string, cfg *Config) {
				newBullets := reflectOnExecution(cfg, toolName, toolArgs, toolResult, userQuery)
				if len(newBullets) > 0 {
					existing, _ := loadPlaybook()
					merged := curateBullets(existing, newBullets)
					if err := savePlaybook(merged); err != nil {
						fmt.Printf("[siki] Failed to save playbook: %v\n", err)
					} else {
						fmt.Printf("[siki] Playbook updated: %d bullets (added %d new)\n", len(merged), len(newBullets))
					}
				}
			}(tc.Function.Name, tc.Function.Arguments, result, req.Message, ws.config)
		}
	}
	cancel()

	// Handle timeout: only compress if conversation is actually long
	if hitTimeout {
		fmt.Printf("[siki] Context deadline exceeded (messages: %d)\n", len(agent.messages))
		if len(agent.messages) > 30 {
			// Conversation is genuinely long — compress it
			sendEvent(StreamEvent{Type: "content", Content: "\n\n*会話履歴を要約しています...*\n\n"})
			compressCtx, compressCancel := context.WithTimeout(context.Background(), 300*time.Second)
			agent.forceCompressConversation(compressCtx)
			compressCancel()
			sendEvent(StreamEvent{Type: "content", Content: "*会話が長くなりすぎたため要約しました。もう一度質問してください。*"})
		} else {
			// Short conversation — timeout was due to slow model, not context size
			sendEvent(StreamEvent{Type: "content", Content: "\n\n*応答がタイムアウトしました。モデルの処理に時間がかかっています。もう一度お試しください。*"})
		}
	}

	// Generate thread title: try LLM, fall back to truncated user message
	if isFirstMessage && req.Message != "" {
		generateThreadTitle(ws.config, threadID, req.Message, lastAssistantReply)
		if t, err := loadThreadMeta(threadID); err == nil {
			// If LLM title generation failed, use truncated user message
			if t.Title == "" || t.Title == "New thread" {
				title := req.Message
				if len(title) > 30 {
					title = title[:30] + "..."
				}
				t.Title = title
				saveThreadMeta(t)
			}
			sendEvent(StreamEvent{Type: "title", Result: t.Title})
		}
	}

	sendEvent(StreamEvent{Type: "done"})
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// ============================================================================
// Docker HTTP Handlers
// ============================================================================

func (ws *WebServer) handleDockerExec(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Command string `json:"command"`
		Timeout int    `json:"timeout"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	if req.Command == "" {
		http.Error(w, "command is required", http.StatusBadRequest)
		return
	}

	output, err := dockerExec(req.Command, req.Timeout)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"output": output,
			"error":  err.Error(),
		})
		return
	}
	json.NewEncoder(w).Encode(map[string]interface{}{
		"output": output,
	})
}

func (ws *WebServer) handleDockerStatus(w http.ResponseWriter, r *http.Request) {
	available := isDockerAvailable()
	imageReady := available && isDockerImageAvailable()

	containerRunning := false
	if available {
		cmd := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", dockerContainerName)
		out, err := cmd.Output()
		containerRunning = err == nil && strings.TrimSpace(string(out)) == "true"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"docker_available":  available,
		"image_ready":       imageReady,
		"container_running": containerRunning,
	})
}

func (ws *WebServer) handleUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := initDockerWorkspace(); err != nil {
		http.Error(w, "Failed to init workspace: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Parse multipart form (max 2GB)
	if err := r.ParseMultipartForm(2 << 30); err != nil {
		http.Error(w, "Failed to parse upload: "+err.Error(), http.StatusBadRequest)
		return
	}

	var uploaded []string
	for _, fileHeaders := range r.MultipartForm.File {
		for _, fh := range fileHeaders {
			if err := saveUploadedFile(fh, dockerWorkspaceDir); err != nil {
				http.Error(w, "Failed to save file: "+err.Error(), http.StatusInternalServerError)
				return
			}
			uploaded = append(uploaded, fh.Filename)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"files":     uploaded,
		"workspace": "/workspace",
		"message":   fmt.Sprintf("%d file(s) uploaded to /workspace", len(uploaded)),
	})
}

func saveUploadedFile(fh *multipart.FileHeader, destDir string) error {
	src, err := fh.Open()
	if err != nil {
		return err
	}
	defer src.Close()

	// Sanitize filename: only keep the base name
	filename := filepath.Base(fh.Filename)
	destPath := filepath.Join(destDir, filename)

	dst, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer dst.Close()

	_, err = io.Copy(dst, src)
	return err
}

func runWeb(config *Config, host string, port int) error {
	ws := NewWebServer(config)

	http.HandleFunc("/", ws.handleIndex)
	http.HandleFunc("/api/status", ws.handleStatus)
	http.HandleFunc("/api/settings", ws.handleSettings)
	http.HandleFunc("/api/chat", ws.handleChat)
	http.HandleFunc("/api/chat/stream", ws.handleChatStream)
	http.HandleFunc("/api/images", ws.handleImages)
	http.HandleFunc("/diagrams/", ws.handleDiagrams)
	http.HandleFunc("/playground/", ws.handlePlayground)
	http.HandleFunc("/api/plugins/ui", ws.handlePluginsUI)
	http.HandleFunc("/api/plugins", ws.handlePlugins)
	http.HandleFunc("/api/threads/", ws.handleThreads)
	http.HandleFunc("/api/threads", ws.handleThreads)
	http.HandleFunc("/api/docker/exec", ws.handleDockerExec)
	http.HandleFunc("/api/docker/status", ws.handleDockerStatus)
	http.HandleFunc("/api/upload", ws.handleUpload)

	// Initialize directories
	if err := initThreadDir(); err != nil {
		fmt.Printf("Warning: failed to initialize thread dir: %v\n", err)
	}
	if err := initDiagramDir(); err != nil {
		fmt.Printf("Warning: failed to initialize diagram dir: %v\n", err)
	}
	if err := initPlaygroundDir(); err != nil {
		fmt.Printf("Warning: failed to initialize playground dir: %v\n", err)
	}
	if err := loadPlugins(); err != nil {
		fmt.Printf("Warning: failed to load plugins: %v\n", err)
	}
	if err := initDockerWorkspace(); err != nil {
		fmt.Printf("Warning: failed to initialize docker workspace: %v\n", err)
	}
	if err := initPlaybookDir(); err != nil {
		fmt.Printf("Warning: failed to initialize playbook dir: %v\n", err)
	}
	// Initialize self-modification system
	if err := initSelfDir(); err != nil {
		fmt.Printf("Warning: failed to initialize self dir: %v\n", err)
	}
	if currentSelf == nil {
		selfState, err := loadSelfState()
		if err != nil {
			fmt.Printf("Warning: failed to load self-state, using defaults: %v\n", err)
			selfState = defaultSelfState()
		}
		currentSelf = selfState
		// Save initial state if prompt.md doesn't exist
		if _, err := os.Stat(filepath.Join(selfDir, "current", "prompt.md")); os.IsNotExist(err) {
			saveSelfState(currentSelf)
			fmt.Printf("[siki] Initialized self-state v%d (prompt.md created)\n", currentSelf.Version)
		}
	}

	fmt.Printf("siki v%s - 式神 Web GUI\n", Version)
	pp := config.primaryProvider()
	fmt.Printf("Backend: %s, Model: %s\n", pp.Backend, pp.Model)
	if config.VisionModel != "" {
		fmt.Printf("Vision Model: %s\n", config.VisionModel)
	}
	if config.SubModel != "" {
		fmt.Printf("Sub Model: %s\n", config.SubModel)
	}
	fmt.Printf("API Endpoint: %s\n", pp.Endpoint)
	if len(config.Providers) > 1 {
		fmt.Printf("Configured providers: %d\n", len(config.Providers))
		for i, p := range config.Providers {
			marker := "  "
			if i == 0 {
				marker = "* "
			}
			fmt.Printf("  %s%s (%s, %s)\n", marker, p.Name, p.Backend, p.Model)
		}
	}

	addr := fmt.Sprintf("%s:%d", host, port)
	if host == "0.0.0.0" {
		// Get local IP for display
		localIP := getLocalIP()
		fmt.Printf("\nStarting web server on:\n")
		fmt.Printf("  Local:   http://localhost:%d\n", port)
		fmt.Printf("  Network: http://%s:%d\n", localIP, port)
	} else {
		fmt.Printf("\nStarting web server on http://%s\n", addr)
	}
	fmt.Println("Press Ctrl+C to stop")

	// Start self-improvement background loop
	go ws.selfImproveLoop()

	// Open browser automatically (only for localhost)
	if host != "0.0.0.0" {
		go func() {
			time.Sleep(500 * time.Millisecond)
			openBrowser(fmt.Sprintf("http://localhost:%d", port))
		}()
	}

	// Graceful shutdown with Docker cleanup
	server := &http.Server{Addr: addr}
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		stopDockerContainer()
		server.Close()
	}()

	return server.ListenAndServe()
}

func getLocalIP() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "127.0.0.1"
	}
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
			if ipnet.IP.To4() != nil {
				return ipnet.IP.String()
			}
		}
	}
	return "127.0.0.1"
}

func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return
	}
	cmd.Start()
}

func runChat(config *Config) error {
	agent := &Agent{
		config: config,
		messages: []Message{
			{Role: "system", Content: buildSystemPrompt(config)},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle Ctrl+C
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nGoodbye!")
		stopDockerContainer()
		cancel()
		os.Exit(0)
	}()

	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("siki v%s - 式神 Agentic AI - Type 'exit' to quit, 'clear' to reset\n", Version)
	chatPP := config.primaryProvider()
	fmt.Printf("Backend: %s, Model: %s, Endpoint: %s\n\n",
		chatPP.Backend, chatPP.Model, chatPP.Endpoint)

	for {
		fmt.Print("> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		input = strings.TrimSpace(input)
		if input == "" {
			continue
		}

		switch input {
		case "exit", "quit":
			fmt.Println("Goodbye!")
			return nil
		case "clear":
			agent.messages = []Message{
				{Role: "system", Content: buildSystemPrompt(config)},
			}
			fmt.Println("Conversation cleared.")
			continue
		case "help":
			printHelp()
			continue
		}

		if err := agent.run(ctx, input); err != nil {
			fmt.Printf("Error: %v\n", err)
		}
	}

	return nil
}

func main() {
	config := defaultConfig()
	webPort := 3000
	webHost := "127.0.0.1"

	// Parse command line arguments
	args := os.Args[1:]

	// Process flags first
	i := 0
	endpointOverridden := false
	for i < len(args) {
		switch args[i] {
		case "--backend":
			if i+1 < len(args) {
				config.Backend = args[i+1]
				if !endpointOverridden {
					config.APIEndpoint = defaultEndpointForBackend(args[i+1])
				}
				i += 2
				continue
			}
		case "--api-key":
			if i+1 < len(args) {
				config.APIKey = args[i+1]
				i += 2
				continue
			}
		case "--model":
			if i+1 < len(args) {
				config.ModelName = args[i+1]
				i += 2
				continue
			}
		case "--endpoint":
			if i+1 < len(args) {
				config.APIEndpoint = args[i+1]
				endpointOverridden = true
				i += 2
				continue
			}
		case "--vision-model":
			if i+1 < len(args) {
				config.VisionModel = args[i+1]
				i += 2
				continue
			}
		case "--sub-model":
			if i+1 < len(args) {
				config.SubModel = args[i+1]
				i += 2
				continue
			}
		case "--reset-self":
			if err := initSelfDir(); err == nil {
				selfMu.Lock()
				currentSelf = defaultSelfState()
				saveSelfState(currentSelf)
				selfMu.Unlock()
				fmt.Println("[siki] Self-state reset to factory defaults.")
			}
			i++
			continue
		case "--workspace":
			if i+1 < len(args) {
				config.Workspace = args[i+1]
				i += 2
				continue
			}
		case "--port":
			if i+1 < len(args) {
				fmt.Sscanf(args[i+1], "%d", &webPort)
				i += 2
				continue
			}
		case "--host":
			if i+1 < len(args) {
				webHost = args[i+1]
				i += 2
				continue
			}
		case "-h", "--help":
			printHelp()
			return
		}
		break
	}

	// Initialize providers from legacy CLI flags if no providers configured
	if len(config.Providers) == 0 {
		config.Providers = []Provider{{
			Name:     "default",
			Backend:  config.Backend,
			Endpoint: config.APIEndpoint,
			Model:    config.ModelName,
			APIKey:   config.APIKey,
		}}
	}

	// Get remaining args as command
	remaining := args[i:]

	if len(remaining) == 0 {
		// Default to chat
		if err := runChat(config); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	mm := NewModelManager(config)

	switch remaining[0] {
	case "chat":
		if err := runChat(config); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

	case "web", "gui":
		if err := runWeb(config, webHost, webPort); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

	case "quickstart":
		if err := runQuickstart(config); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

	case "download":
		if len(remaining) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: stand download <model>")
			os.Exit(1)
		}
		if err := mm.DownloadModel(remaining[1]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Download complete!")

	case "list":
		models, err := mm.ListModels()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if len(models) == 0 {
			fmt.Println("No models downloaded yet.")
		} else {
			fmt.Println("Downloaded models:")
			for _, m := range models {
				fmt.Printf("  - %s\n", m)
			}
		}

	case "serve":
		if len(remaining) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: stand serve <model>")
			os.Exit(1)
		}
		modelPath := filepath.Join(config.ModelPath, remaining[1])
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			// Try using the model name directly (for ollama)
			modelPath = remaining[1]
		}

		if err := mm.StartServer(modelPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Wait for interrupt
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh
		fmt.Println("\nShutting down server...")
		mm.StopServer()

	case "config":
		data, _ := json.MarshalIndent(config, "", "  ")
		fmt.Println(string(data))

	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", remaining[0])
		printHelp()
		os.Exit(1)
	}
}
