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
		ModelName:   "gpt-oss:20b",
		Backend:     backend,
		APIEndpoint: endpoint,
		Workspace:   ".",
		MaxTurns:    MaxTurns,
		SystemPrompt: `You are 式神 (Shikigami), a helpful AI assistant with access to powerful tools.

## Available Tools
- read_file, write_file, list_files: File operations
- execute_command: Run shell commands
- search_files, grep: Search code and files
- web_search: Search the internet
- web_fetch: Get text content from a URL
- web_images: Extract images from a URL
- diagram: Generate Graphviz diagrams
- run_code: **Execute HTML/JavaScript in browser iframe**
- blog_person_search: **ブログの最新記事を巡回し、言及されている人物名を抽出する**。人物の関係性や著名人について聞かれたらこのツールを使う
- search_conversation: **過去の会話履歴からキーワード検索**する。以前の会話内容を参照したいときに使う
- recall_context: **現在のスレッドの過去の会話を検索**する。要約されて見えなくなった詳細を思い出すときに使う
- search_threads: **全スレッドを横断検索**する。他のスレッドの会話内容を探すときに使う
- create_plugin: **プラグインを作成/更新**する。新しいツールやUI拡張を動的に追加できる
- test_plugin: **プラグインをテスト**する。テストケースを渡して全パスしたらTESTED状態にする
- list_plugins: インストール済みプラグイン一覧を表示
- delete_plugin: プラグインを削除

## Plugin System
You can create plugins to extend your own capabilities:
- **Tool plugins**: Server-side Node.js code. The code receives a ` + "`params`" + ` object and outputs results via console.log(). Can use require() for Node.js built-in modules.
- **UI plugins**: Client-side JS/CSS rendered in a dedicated Plugin pane (right sidebar). The JS receives a ` + "`pane`" + ` argument (DOM element) - render all UI into this element only. Do NOT modify the main page DOM.
  - Example ui_js: ` + "`pane.innerHTML = '<h3>My Plugin</h3><p>Hello!</p>';`" + `
- Plugin tools become available as ` + "`plugin_<name>`" + ` immediately after creation.
- Use create_plugin when the user needs functionality you don't have built-in.
- **IMPORTANT: After creating a plugin with a tool, you MUST immediately use test_plugin to run test cases.** Design at least 2-3 meaningful test cases covering normal and edge cases. A plugin is NOT ready until it passes all tests and is marked TESTED. If tests fail, fix the plugin with create_plugin and re-test.

## run_code Tool Specification
When user asks to draw, visualize, calculate, simulate, or create anything visual:

1. **ALWAYS use run_code** - do not show code examples, execute them
2. **HTML format must be complete and self-contained:**

<html>
<head>
<style>body{margin:0;background:#1a1a2e;}</style>
</head>
<body>
<canvas id="c" width="800" height="600"></canvas>
<script>
// Your JavaScript code here
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
// Draw something...
</script>
</body>
</html>

3. **After run_code succeeds, respond briefly** like "マンデルブロ集合を描画しました。クリックでズームできます。"

4. **Do NOT:**
   - Show code blocks as examples
   - Explain how to write the code
   - Ask if the user wants to see code

5. **DO:**
   - Immediately call run_code with working HTML
   - Make it interactive when possible (mouse events, animations)
   - Use Canvas for graphics, SVG for diagrams

## Examples of when to use run_code:
- "フラクタルを描いて" → run_code with Mandelbrot/Julia set
- "ソートを可視化して" → run_code with sorting animation
- "素数を表示して" → run_code with prime number visualization
- "ゲームを作って" → run_code with interactive game
- "グラフを描いて" → run_code with chart

## Other Rules
- For news/current events: use web_search
- For relationships/architecture: use diagram
- Respond in user's language
- IMPORTANT: Always include at least one relevant URL in your response. Use web_search to find URLs if needed.
- For questions about people related to a person/blogger: use blog_person_search to read their blog articles and extract person names. Do NOT just do a web_search - actually read the articles.
- To recall earlier conversation: use search_conversation`,
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
		Description: "Search the current conversation thread's full history for specific content. Use this when you need to recall details from earlier in the conversation that may have been summarized.",
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

// ============================================================================
// Tool Execution
// ============================================================================

type Agent struct {
	config   *Config
	messages []Message
	threadID string
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

func (a *Agent) executeTool(name string, args map[string]interface{}) (string, error) {
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
		return a.runCode(args["html"].(string), title)
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
		MaxTokens:   4096,
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
	// Use DuckDuckGo HTML search
	searchURL := fmt.Sprintf("https://html.duckduckgo.com/html/?q=%s", strings.ReplaceAll(query, " ", "+"))

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

	return fmt.Sprintf("Search results for: %s\n\n%s", query, strings.Join(results, "\n---\n")), nil
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

type Thread struct {
	ID        string          `json:"id"`
	Title     string          `json:"title"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
	Messages  []ThreadMessage `json:"messages"`
	Summary   string          `json:"summary,omitempty"`
}

type ThreadMessage struct {
	Role       string   `json:"role"`
	Content    string   `json:"content"`
	Images     []string `json:"images,omitempty"`
	ToolCallID string   `json:"tool_call_id,omitempty"`
	ToolName   string   `json:"tool_name,omitempty"`
	Timestamp  int64    `json:"timestamp"`
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

func saveThread(t *Thread) error {
	if err := initThreadDir(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(threadDir, t.ID+".json"), data, 0644)
}

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
		id := strings.TrimSuffix(entry.Name(), ".json")
		t, err := loadThread(id)
		if err != nil {
			continue
		}
		items = append(items, ThreadListItem{
			ID:           t.ID,
			Title:        t.Title,
			CreatedAt:    t.CreatedAt,
			UpdatedAt:    t.UpdatedAt,
			MessageCount: len(t.Messages),
			Summary:      t.Summary,
		})
	}
	return items, nil
}

func deleteThread(id string) error {
	if err := initThreadDir(); err != nil {
		return err
	}
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

func saveMessagesToThread(threadID string, newMsgs []Message, userMessage string) {
	thread, err := loadThread(threadID)
	if err != nil {
		// Create new thread
		title := userMessage
		if len(title) > 50 {
			title = title[:50] + "..."
		}
		thread = &Thread{
			ID:        threadID,
			Title:     title,
			CreatedAt: time.Now(),
		}
	}

	thread.UpdatedAt = time.Now()
	for _, m := range newMsgs {
		if m.Role == "system" {
			continue
		}
		tm := ThreadMessage{
			Role:       m.Role,
			Content:    m.Content,
			Images:     m.Images,
			ToolCallID: m.ToolCallID,
			Timestamp:  time.Now().Unix(),
		}
		thread.Messages = append(thread.Messages, tm)
	}
	if err := saveThread(thread); err != nil {
		fmt.Printf("[siki] Warning: failed to save thread %s: %v\n", threadID, err)
	}
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
	return a.chatStream(ctx, nil)
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

	client := &http.Client{Timeout: 30 * time.Second}
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
	// Keep system message + last 6 messages, compress everything in between
	if len(a.messages) <= 8 {
		return
	}

	// Messages to compress: from index 1 (after system) to len-6
	compressEnd := len(a.messages) - 6
	if compressEnd <= 1 {
		return
	}

	var historyText strings.Builder
	for i := 1; i < compressEnd; i++ {
		msg := a.messages[i]
		if msg.Role == "tool" {
			continue // skip tool results for compression
		}
		historyText.WriteString(fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content))
	}

	if historyText.Len() == 0 {
		return
	}

	// Truncate history text if too long
	history := historyText.String()
	if len(history) > 10000 {
		history = history[:10000]
	}

	compressMessages := []Message{
		{Role: "system", Content: "会話履歴を簡潔に要約してください。重要な事実、ユーザーの質問の要点、AIの回答の要点を保持してください。200文字以内で要約してください。"},
		{Role: "user", Content: history},
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    compressMessages,
		MaxTokens:   300,
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

	client := &http.Client{Timeout: 30 * time.Second}
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
	fmt.Printf("[siki] Compressed %d messages into summary\n", compressEnd-1)

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

	client := &http.Client{Timeout: 60 * time.Second}
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

// searchConversation searches through conversation history for relevant messages
func (a *Agent) searchConversation(query string) string {
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
				// Find the match position and show surrounding context
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
	// Search through all messages
	queryLower := strings.ToLower(query)
	var matches []string
	for _, m := range thread.Messages {
		if strings.Contains(strings.ToLower(m.Content), queryLower) {
			matches = append(matches, fmt.Sprintf("[%s] %s: %s", time.Unix(m.Timestamp, 0).Format("15:04"), m.Role, truncateString(m.Content, 300)))
		}
	}
	if len(matches) == 0 {
		return fmt.Sprintf("No matches found for '%s' in current thread.", query), nil
	}
	if len(matches) > 10 {
		matches = matches[len(matches)-10:]
	}
	return fmt.Sprintf("Found %d matches:\n\n%s", len(matches), strings.Join(matches, "\n\n")), nil
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

func (a *Agent) chatStream(ctx context.Context, onContent func(string)) (*Message, error) {
	// Auto-compress conversation if it's getting long
	a.compressConversation(ctx)

	// Convert tools to OpenAI format
	var toolDefs []map[string]interface{}
	for _, tool := range getAllTools() {
		toolDefs = append(toolDefs, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  tool.Parameters,
			},
		})
	}

	req := ChatRequest{
		Model:       a.config.primaryProvider().Model,
		Messages:    a.messages,
		Tools:       toolDefs,
		ToolChoice:  "auto",
		MaxTokens:   4096,
		Temperature: 0.7,
		Stream:      onContent != nil,
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

	client := &http.Client{Timeout: 120 * time.Second}
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
	if onContent != nil {
		return a.handleStreamingResponse(resp.Body, onContent)
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

func (a *Agent) handleStreamingResponse(body io.Reader, onContent func(string)) (*Message, error) {
	reader := bufio.NewReader(body)
	var fullContent strings.Builder
	var toolCalls []ToolCall
	toolCallArgs := make(map[int]string) // index -> accumulated arguments

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
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

		// Handle content
		if delta.Content != "" {
			fullContent.WriteString(delta.Content)
			onContent(delta.Content)
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

	// Finalize tool call arguments
	for idx, args := range toolCallArgs {
		if idx < len(toolCalls) {
			toolCalls[idx].Function.Arguments = args
		}
	}

	return &Message{
		Role:      "assistant",
		Content:   fullContent.String(),
		ToolCalls: toolCalls,
	}, nil
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
	Model     string     `json:"model"`
	Backend   string     `json:"backend"`
	Endpoint  string     `json:"endpoint"`
	Version   string     `json:"version"`
	HasAPIKey bool       `json:"has_api_key"`
	Providers []Provider `json:"providers"`
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
	}
}

func buildSystemPrompt(config *Config) string {
	now := time.Now()
	dateStr := fmt.Sprintf("今日は%d年%d月%d日です。", now.Year(), int(now.Month()), now.Day())

	var sb strings.Builder
	sb.WriteString(dateStr)
	sb.WriteString("\n\n")
	sb.WriteString(config.SystemPrompt)

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

	// Load thread and build orchestrator context
	thread, err := loadThread(convID)
	if err == nil && len(thread.Messages) > 0 {
		// Add summary if available
		if thread.Summary != "" {
			agent.messages = append(agent.messages, Message{
				Role:    "assistant",
				Content: fmt.Sprintf("[Previous conversation summary]\n%s", thread.Summary),
			})
		}
		// Add last N messages for immediate context
		recentMessages := getRecentThreadMessages(thread, 10)
		for _, tm := range recentMessages {
			msg := Message{Role: tm.Role, Content: tm.Content, Images: tm.Images, ToolCallID: tm.ToolCallID}
			agent.messages = append(agent.messages, msg)
		}
	}

	ws.conversations[convID] = agent
	return agent
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
		t, err := loadThread(threadID)
		if err != nil {
			http.Error(w, "Thread not found", http.StatusNotFound)
			return
		}
		t.Title = req.Title
		t.UpdatedAt = time.Now()
		if err := saveThread(t); err != nil {
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
		Model:     pp.Model,
		Backend:   pp.Backend,
		Endpoint:  pp.Endpoint,
		Version:   Version,
		HasAPIKey: pp.APIKey != "",
		Providers: ws.config.Providers,
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

	// Track starting index for new messages to save to thread
	msgStartIdx := len(agent.messages)

	// Add user message
	agent.messages = append(agent.messages, Message{
		Role:    "user",
		Content: req.Message,
		Images:  req.Images,
	})

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

		if len(response.ToolCalls) == 0 {
			apiResp.Response = response.Content
			break
		}

		// Execute tool calls
		for _, tc := range response.ToolCalls {
			var args map[string]interface{}
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				agent.messages = append(agent.messages, Message{
					Role:       "tool",
					Content:    fmt.Sprintf("Error parsing arguments: %v", err),
					ToolCallID: tc.ID,
				})
				apiResp.ToolCalls = append(apiResp.ToolCalls, ToolCallResult{
					Name:   tc.Function.Name,
					Result: fmt.Sprintf("Error: %v", err),
				})
				continue
			}

			result, err := agent.executeTool(tc.Function.Name, args)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			// Truncate very long results for display
			displayResult := result
			if len(displayResult) > 2000 {
				displayResult = displayResult[:2000] + "\n... (truncated)"
			}

			apiResp.ToolCalls = append(apiResp.ToolCalls, ToolCallResult{
				Name:   tc.Function.Name,
				Result: displayResult,
			})

			agent.messages = append(agent.messages, Message{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
			})
		}
	}

	// Save new messages to thread
	if msgStartIdx < len(agent.messages) {
		saveMessagesToThread(req.ConversationID, agent.messages[msgStartIdx:], req.Message)
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

	agent := ws.getOrCreateAgent(req.ConversationID)

	// Track starting index for new messages to save to thread
	msgStartIdx := len(agent.messages)

	// Add user message
	agent.messages = append(agent.messages, Message{
		Role:    "user",
		Content: req.Message,
		Images:  req.Images,
	})

	const maxRetries = 2

	for retry := 0; retry <= maxRetries; retry++ {
		ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
		var fullResponse string
		var hitTimeout bool

		// Agent loop
		for turn := 0; turn < ws.config.MaxTurns; turn++ {
			// Try streaming first
			response, err := agent.chatStream(ctx, func(content string) {
				sendEvent(StreamEvent{Type: "content", Content: content})
			})

			if err != nil {
				// Check if it's a timeout/context error
				if strings.Contains(err.Error(), "deadline exceeded") || strings.Contains(err.Error(), "context canceled") {
					hitTimeout = true
					break
				}
				sendEvent(StreamEvent{Type: "error", Error: err.Error()})
				cancel()
				return
			}

			agent.messages = append(agent.messages, *response)

			if len(response.ToolCalls) == 0 {
				fullResponse = response.Content
				break
			}

			// Execute tool calls
			for _, tc := range response.ToolCalls {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					result := fmt.Sprintf("Error parsing arguments: %v", err)
					sendEvent(StreamEvent{Type: "tool_call", Name: tc.Function.Name, Result: result})
					agent.messages = append(agent.messages, Message{
						Role:       "tool",
						Content:    result,
						ToolCallID: tc.ID,
					})
					continue
				}

				result, err := agent.executeTool(tc.Function.Name, args)
				if err != nil {
					result = fmt.Sprintf("Error: %v", err)
				}

				// Truncate very long results for display
				displayResult := result
				if len(displayResult) > 2000 {
					displayResult = displayResult[:2000] + "\n... (truncated)"
				}

				sendEvent(StreamEvent{Type: "tool_call", Name: tc.Function.Name, Result: displayResult})

				agent.messages = append(agent.messages, Message{
					Role:       "tool",
					Content:    result,
					ToolCallID: tc.ID,
				})
			}
		}
		cancel()

		// Handle timeout: compress conversation and retry
		if hitTimeout {
			fmt.Printf("[siki] Context deadline exceeded, compressing conversation (attempt %d)\n", retry+1)
			sendEvent(StreamEvent{Type: "content", Content: "\n\n*会話履歴を要約しています...*\n\n"})

			// Force compress with a fresh context
			compressCtx, compressCancel := context.WithTimeout(context.Background(), 60*time.Second)
			agent.forceCompressConversation(compressCtx)
			compressCancel()

			if retry < maxRetries {
				sendEvent(StreamEvent{Type: "content", Content: "*要約完了。回答を再生成しています...*\n\n"})
				continue
			} else {
				// Show what we have as summary
				sendEvent(StreamEvent{Type: "content", Content: "*会話が長くなりすぎたため要約しました。もう一度質問してください。*"})
				break
			}
		}

		// Validate the response
		if fullResponse != "" && retry < maxRetries {
			valCtx, valCancel := context.WithTimeout(context.Background(), 30*time.Second)
			ok, reason := agent.validateResponse(valCtx, req.Message, fullResponse)
			valCancel()
			if !ok {
				fmt.Printf("[siki] Response validation failed (attempt %d): %s\n", retry+1, reason)
				// Remove the bad response from messages
				agent.messages = agent.messages[:len(agent.messages)-1]
				// Add a retry prompt
				agent.messages = append(agent.messages, Message{
					Role:    "user",
					Content: fmt.Sprintf("申し訳ありませんが、先ほどの回答は質問に対して適切ではありませんでした（理由: %s）。元の質問「%s」に対して、会話の文脈を踏まえてもう一度回答してください。", reason, req.Message),
				})
				// Send apology notification to client
				sendEvent(StreamEvent{Type: "content", Content: "\n\n---\n*回答を再生成しています...*\n\n"})
				continue
			}
		}
		break
	}

	// Save new messages to thread
	if msgStartIdx < len(agent.messages) {
		saveMessagesToThread(req.ConversationID, agent.messages[msgStartIdx:], req.Message)
	}

	sendEvent(StreamEvent{Type: "done"})
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
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

	fmt.Printf("siki v%s - 式神 Web GUI\n", Version)
	pp := config.primaryProvider()
	fmt.Printf("Backend: %s, Model: %s\n", pp.Backend, pp.Model)
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

	// Open browser automatically (only for localhost)
	if host != "0.0.0.0" {
		go func() {
			time.Sleep(500 * time.Millisecond)
			openBrowser(fmt.Sprintf("http://localhost:%d", port))
		}()
	}

	return http.ListenAndServe(addr, nil)
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
