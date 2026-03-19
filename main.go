// siki - 式神 (Shikigami) Agentic AI Environment
// A portable, single-binary agentic AI tool that runs completely locally
// Named after Shikigami - supernatural servants in Onmyodo tradition
// Supports vllm (GPU), mlx-lm (Mac), and ollama backends

package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/tls"
	"embed"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"mime/multipart"
	"net"
	"net/http"
	"net/smtp"
	"net/textproto"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
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
	SubModel        string     `json:"sub_model"`
	SubModelBackend string     `json:"sub_model_backend"` // "ollama" (default) or "vllm"
	SubModelEndpoint string    `json:"sub_model_endpoint"` // optional separate endpoint for sub-model
	Orchestrator         string `json:"orchestrator"`           // orchestrator model (defaults to SubModel if empty)
	OrchestratorBackend  string `json:"orchestrator_backend"`   // "ollama" (default) or "vllm"
	OrchestratorEndpoint string `json:"orchestrator_endpoint"`  // optional separate endpoint
	SubAgent         string    `json:"sub_agent"`          // powerful model for summarization/code gen (e.g. Qwen3.5-27B)
	SubAgentBackend  string    `json:"sub_agent_backend"`  // "vllm" (default) or "ollama"
	SubAgentEndpoint string    `json:"sub_agent_endpoint"` // endpoint for sub-agent (e.g. http://localhost:8000)
	ImageModel      string     `json:"image_model"`      // default: "black-forest-labs/FLUX.2-klein-4B"
	ImageEndpoint   string     `json:"image_endpoint"`   // default: "http://localhost:8100"
	ImageEnabled    bool       `json:"image_enabled"`    // default: true
	VideoModel      string     `json:"video_model"`      // default: "BestWishYSH/Helios-Distilled"
	VideoEndpoint   string     `json:"video_endpoint"`   // default: "http://localhost:8101"
	VideoEnabled    bool       `json:"video_enabled"`    // default: true
	// Email digest settings
	EmailTo       string `json:"email_to"`
	EmailFrom     string `json:"email_from"`
	SMTPHost      string `json:"smtp_host"`
	SMTPPort      int    `json:"smtp_port"`
	SMTPUser      string `json:"smtp_user"`
	SMTPPass      string `json:"smtp_pass"`
	DigestEnabled bool   `json:"digest_enabled"`
	DigestHours   []int  `json:"digest_hours"`
	// Twitter timeline integration
	TwitterBearerToken    string `json:"twitter_bearer_token"`
	TwitterEnabled        bool   `json:"twitter_enabled"`
	TwitterConsumerKey    string `json:"twitter_consumer_key"`
	TwitterConsumerSecret string `json:"twitter_consumer_secret"`
	TwitterAccessToken    string `json:"twitter_access_token"`
	TwitterAccessSecret   string `json:"twitter_access_secret"`
	// Bluesky feed integration
	BlueskyEnabled      bool     `json:"bluesky_enabled"`
	BlueskyStarterPacks []string `json:"bluesky_starter_packs"`
	BlueskyIdentifier   string   `json:"bluesky_identifier"`   // handle or email for auth
	BlueskyAppPassword  string   `json:"bluesky_app_password"` // app password for auth
	// Bluesky Jetstream monitoring
	JetstreamKeywords []string `json:"jetstream_keywords"`
	// Zeroboot sandbox
	ZerobootEndpoint string `json:"zeroboot_endpoint"` // default: "https://api.zeroboot.dev"
	ZerobootAPIKey   string `json:"zeroboot_api_key"`  // default: "zb_demo_hn2026"
	// External skill API keys
	BraveAPIKey string `json:"brave_api_key"`
	GroqAPIKey  string `json:"groq_api_key"`
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

// orchestratorModel returns the model used for orchestration (falls back to SubModel).
func (c *Config) orchestratorModel() string {
	if c.Orchestrator != "" {
		return c.Orchestrator
	}
	return c.SubModel
}

// orchestratorBackend returns the backend for orchestration (falls back to SubModelBackend).
func (c *Config) orchestratorBackend() string {
	if c.OrchestratorBackend != "" {
		return c.OrchestratorBackend
	}
	return c.SubModelBackend
}

// orchestratorEndpoint returns the endpoint for orchestration (falls back to sub-model endpoint).
func (c *Config) orchestratorEndpoint() string {
	if c.OrchestratorEndpoint != "" {
		return c.OrchestratorEndpoint
	}
	// If orchestrator uses ollama backend, default to ollama endpoint
	if c.Orchestrator != "" && c.orchestratorBackend() == "ollama" {
		if c.SubModelEndpoint != "" {
			return strings.TrimSuffix(c.SubModelEndpoint, "/")
		}
		return "http://localhost:11434"
	}
	return subModelEndpoint(c)
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
	return &Config{
		ModelPath:   filepath.Join(home, ".siki", "models"),
		ModelName:   "openai/gpt-oss-20b",
		Backend:     "vllm",
		APIEndpoint: "http://localhost:8001/v1",
		Workspace:   ".",
		MaxTurns:    MaxTurns,
		VisionModel: "moondream",
		SubModel:        "gpt-oss:latest",
		SubModelBackend: "ollama",
		Orchestrator:        "gpt-oss-20b-128k:latest",
		OrchestratorBackend: "ollama",
		ImageModel:   "black-forest-labs/FLUX.2-klein-4B",
		ImageEndpoint: "http://localhost:8100",
		ImageEnabled:  true,
		SystemPrompt: `あなたは式神(Shikigami)。ツールを持つAIアシスタント。

## 最重要ルール（絶対に守れ）
1. ニュース・時事・最新情報 → 必ず web_search を呼べ。自分の知識で答えるな。
2. URL・ウェブページの内容 → 必ず web_fetch を呼べ。推測するな。
3. ファイル操作 → 必ず read_file/write_file を呼べ。
4. ツールで取得した情報だけで回答しろ。ツールなしで推測回答するな。
5. 回答にはURLを含めろ。URLが無ければ web_search で探せ。

## キーワード→ツール対応（この通りに動け）
- タイムライン/フィード/自分のTwitter/フォロー → twitter_timeline
- twitter/ツイッター/ツイート/tweet/検索 → twitter_search
- ニュース/最新/トレンド/速報/news/latest → web_search
- 教えて/調べて/検索/について → web_search
- HuggingFace/GitHubのURL + 「使って」「動かして」「生成して」 → docker_run_model
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
- twitter_search: Twitter/X検索（ツイート検索）
- twitter_timeline: 自分のTwitterタイムライン取得
- web_fetch: URL内容取得
- web_images: URL画像抽出
- diagram: Graphviz図生成
- run_code: HTML/JS実行（ブラウザiframe）
- blog_person_search: ブログ巡回・人物抽出
- search_conversation: 会話履歴検索
- recall_context: 現スレッド会話ログ検索
- search_threads: 全スレッド横断検索
- docker_exec: GPUコンテナ内コマンド実行
- docker_run_model: HuggingFace/GitHubモデルをDocker GPU環境で自動実行
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

// Skill system
type Skill struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Content     string   `json:"content,omitempty"` // SKILL.md full content
	Files       []string `json:"files,omitempty"`   // supporting files in skill dir
	Source      string   `json:"source,omitempty"`  // e.g. "superpowers"
}

var (
	loadedSkills []Skill
	skillsMu     sync.RWMutex
)

func skillsDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".siki", "skills")
}

func loadSkills() []Skill {
	dir := skillsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var skills []Skill
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		skillPath := filepath.Join(dir, e.Name(), "SKILL.md")
		data, err := os.ReadFile(skillPath)
		if err != nil {
			continue
		}
		content := string(data)
		name, desc := parseSkillFrontmatter(content)
		if name == "" {
			name = e.Name()
		}

		// List supporting files
		var files []string
		subEntries, _ := os.ReadDir(filepath.Join(dir, e.Name()))
		for _, se := range subEntries {
			if !se.IsDir() && se.Name() != "SKILL.md" {
				files = append(files, se.Name())
			}
		}

		// Detect source
		source := ""
		if _, err := os.Stat(filepath.Join(dir, e.Name(), ".superpowers")); err == nil {
			source = "superpowers"
		}

		skills = append(skills, Skill{
			Name:        name,
			Description: desc,
			Content:     content,
			Files:       files,
			Source:       source,
		})
	}
	return skills
}

func parseSkillFrontmatter(content string) (name, description string) {
	if !strings.HasPrefix(content, "---") {
		return "", ""
	}
	end := strings.Index(content[3:], "---")
	if end < 0 {
		return "", ""
	}
	frontmatter := content[3 : 3+end]
	for _, line := range strings.Split(frontmatter, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "name:") {
			name = strings.TrimSpace(strings.TrimPrefix(line, "name:"))
			name = strings.Trim(name, "\"'")
		} else if strings.HasPrefix(line, "description:") {
			description = strings.TrimSpace(strings.TrimPrefix(line, "description:"))
			description = strings.Trim(description, "\"'")
		}
	}
	return
}

func getSkillByName(name string) *Skill {
	skillsMu.RLock()
	defer skillsMu.RUnlock()
	for i := range loadedSkills {
		if loadedSkills[i].Name == name {
			return &loadedSkills[i]
		}
	}
	return nil
}

func getSkillContent(name string) string {
	skill := getSkillByName(name)
	if skill == nil {
		return ""
	}
	// Return SKILL.md content without frontmatter
	content := skill.Content
	if strings.HasPrefix(content, "---") {
		if end := strings.Index(content[3:], "---"); end >= 0 {
			content = strings.TrimSpace(content[3+end+3:])
		}
	}
	// Replace {baseDir} placeholder with actual skill directory path
	baseDir := filepath.Join(skillsDir(), name)
	content = strings.ReplaceAll(content, "{baseDir}", baseDir)
	return content
}

func getSkillFile(skillName, fileName string) string {
	path := filepath.Join(skillsDir(), skillName, fileName)
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

func installSkillsFromDir(srcDir, source string) (int, error) {
	destDir := skillsDir()
	os.MkdirAll(destDir, 0755)

	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		skillSrc := filepath.Join(srcDir, e.Name(), "SKILL.md")
		if _, err := os.Stat(skillSrc); err != nil {
			continue
		}
		destSkillDir := filepath.Join(destDir, e.Name())
		os.MkdirAll(destSkillDir, 0755)

		// Copy all files in skill directory
		subEntries, _ := os.ReadDir(filepath.Join(srcDir, e.Name()))
		for _, se := range subEntries {
			if se.IsDir() {
				// Copy subdirectories recursively
				copyDirRecursive(filepath.Join(srcDir, e.Name(), se.Name()), filepath.Join(destSkillDir, se.Name()))
				continue
			}
			srcFile := filepath.Join(srcDir, e.Name(), se.Name())
			dstFile := filepath.Join(destSkillDir, se.Name())
			data, err := os.ReadFile(srcFile)
			if err != nil {
				continue
			}
			os.WriteFile(dstFile, data, 0644)
		}
		// Mark source
		if source != "" {
			os.WriteFile(filepath.Join(destSkillDir, "."+source), []byte(source), 0644)
		}
		count++
	}
	// Reload
	skillsMu.Lock()
	loadedSkills = loadSkills()
	skillsMu.Unlock()
	return count, nil
}

func copyDirRecursive(src, dst string) {
	os.MkdirAll(dst, 0755)
	entries, err := os.ReadDir(src)
	if err != nil {
		return
	}
	for _, e := range entries {
		srcPath := filepath.Join(src, e.Name())
		dstPath := filepath.Join(dst, e.Name())
		if e.IsDir() {
			copyDirRecursive(srcPath, dstPath)
		} else {
			data, err := os.ReadFile(srcPath)
			if err == nil {
				os.WriteFile(dstPath, data, 0644)
			}
		}
	}
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
		Name:        "twitter_search",
		Description: "Search Twitter/X for recent tweets. Use this when the user asks about Twitter trends, specific topics on Twitter, or wants to see what people are saying about something on Twitter/X.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query for Twitter",
				},
				"max_results": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of tweets to return (10-100, default 30)",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "twitter_timeline",
		Description: "Fetch the authenticated user's Twitter/X home timeline and summarize it. Use this when the user asks to see their timeline, what's happening on their feed, or wants a summary of recent tweets from people they follow.",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "bluesky_feed",
		Description: "Bluesky AI/MLコミュニティの投稿を取得・表示する。ユーザーがBlueskyフィードの確認、AI/MLコミュニティの話題を知りたい場合に使用。",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "bluesky_search",
		Description: "Blueskyでキーワード検索して投稿を取得する。特定のトピック・キーワードでBlueskyの投稿を検索したい場合に使用。",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "検索キーワード",
				},
			},
			"required": []string{"query"},
		},
	},
	{
		Name:        "jetstream_search",
		Description: "Bluesky Jetstreamで監視・保存した投稿を検索する。キーワードモニタリングで収集した過去のBluesky投稿をキーワードで検索。",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "検索キーワード",
				},
				"days": map[string]interface{}{
					"type":        "integer",
					"description": "何日前まで遡るか（デフォルト: 7）",
				},
			},
			"required": []string{"query"},
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
		Name:        "sandbox_exec",
		Description: "Execute Python or JavaScript code in an isolated Zeroboot VM sandbox (~0.8ms startup). Use for safe code execution, data processing, chart generation. To output files, base64-encode and print as: __FILE:filename:base64data__. The sandbox has numpy available. For PNG image generation, use struct+zlib to build PNG manually or use numpy for pixel data. Use exec(base64.b64decode('...').decode()) for multiline scripts since the sandbox runs python -c.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"code": map[string]interface{}{
					"type":        "string",
					"description": "Python or JavaScript code to execute",
				},
				"language": map[string]interface{}{
					"type":        "string",
					"description": "Language: 'python' (default) or 'node'",
				},
				"timeout": map[string]interface{}{
					"type":        "number",
					"description": "Timeout in seconds (default: 30, max: 300)",
				},
			},
			"required": []string{"code"},
		},
	},
	{
		Name:        "docker_exec",
		Description: "Execute a command inside a GPU-enabled Docker container (siki-worker). Use for GPU-intensive tasks like ML inference, whisper transcription. Falls back if Docker unavailable.",
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
		Name:        "docker_run_model",
		Description: "HuggingFace/GitHubのモデルやリポジトリをDocker GPU環境で自動実行する。URLからREADMEを取得し、Pythonスクリプトを自動生成して実行する。CUDA OOM防止のため自動でollamaモデルをアンロードし、完了後にリロードする。",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"url": map[string]interface{}{
					"type":        "string",
					"description": "HuggingFace or GitHub URL of the model/repository",
				},
				"prompt": map[string]interface{}{
					"type":        "string",
					"description": "What to generate or execute (e.g., 'a cat playing piano')",
				},
			},
			"required": []string{"url"},
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
	{
		Name:        "generate_image",
		Description: "Generate an image using Flux Klein 4B AI model. Creates PNG images from text prompts. Use for infographics, illustrations, concept art, and visual content. Prompts should be in English and detailed.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"prompt": map[string]interface{}{
					"type":        "string",
					"description": "Detailed English prompt describing the image to generate",
				},
				"width": map[string]interface{}{
					"type":        "integer",
					"description": "Image width in pixels (256-1024, default 512)",
				},
				"height": map[string]interface{}{
					"type":        "integer",
					"description": "Image height in pixels (256-1024, default 512)",
				},
			},
			"required": []string{"prompt"},
		},
	},
	{
		Name:        "generate_video",
		Description: "Generate a video using Helios AI model. Creates MP4 videos from text prompts. Use for animations, motion graphics, and video content. Prompts should be in English and detailed.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"prompt": map[string]interface{}{
					"type":        "string",
					"description": "Detailed English prompt describing the video to generate",
				},
				"num_frames": map[string]interface{}{
					"type":        "integer",
					"description": "Number of frames (multiples of 33, default 33 for ~1.4s, 99 for ~4s)",
				},
			},
			"required": []string{"prompt"},
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

var skillTools = []Tool{
	{
		Name:        "use_skill",
		Description: "Activate a skill to guide your workflow. Skills provide structured processes for brainstorming, debugging, planning, code review, etc. The skill's instructions will be loaded into context.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Skill name to activate (e.g. 'brainstorming', 'systematic-debugging')",
				},
				"file": map[string]interface{}{
					"type":        "string",
					"description": "Optional: read a supporting file from the skill directory (e.g. 'visual-companion.md')",
				},
			},
			"required": []string{"name"},
		},
	},
	{
		Name:        "list_skills",
		Description: "List all installed skills with descriptions",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
}

// getAllTools returns built-in tools + plugin tools + skill tools + plugin management tools
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
	all = append(all, skillTools...)
	return all
}

// coreToolNames are always included (small models choke on 30+ tool definitions)
var coreToolNames = map[string]bool{
	"web_search": true, "web_fetch": true, "read_file": true,
	"write_file": true, "list_files": true, "execute_command": true,
	"search_files": true, "grep": true, "diagram": true, "run_code": true,
	"use_skill": true, "list_skills": true, "sandbox_exec": true,
}

// toolTriggers maps keywords in user messages to additional tool names
var toolTriggers = map[string][]string{
	"ブログ": {"blog_person_search"}, "人物": {"blog_person_search"},
	"前の会話": {"search_conversation", "recall_context"}, "さっき": {"search_conversation", "recall_context"},
	"会話": {"search_conversation", "recall_context", "search_threads"}, "思い出": {"recall_context", "recall_memory"},
	"スレッド": {"search_threads"}, "docker": {"docker_exec", "docker_run_model"}, "コンテナ": {"docker_exec"},
	"gpu": {"docker_exec"}, "ffmpeg": {"docker_exec"}, "whisper": {"docker_exec"},
	"huggingface": {"docker_run_model"}, "github.com": {"docker_run_model"},
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
	"twitter": {"twitter_search", "twitter_timeline"}, "ツイッター": {"twitter_search", "twitter_timeline"}, "ツイート": {"twitter_search"},
	"tweet": {"twitter_search"}, "x.com": {"twitter_search"}, "タイムライン": {"twitter_timeline"}, "フィード": {"twitter_timeline"},
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
// needsImageGeneration detects if user wants AI image generation (not code/diagram)
func needsImageGeneration(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	keywords := []string{
		"画像生成", "画像を生成", "画像を作", "イラスト描", "イラストを描",
		"インフォグラフィック", "インフォグラフィクス",
		"image generat", "generate image", "generate an image",
		"写真を生成", "写真を作", "絵を描いて", "絵を生成", "絵をかいて",
		"コンセプトアート", "concept art",
	}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// containsResearchKeywords detects if user wants to search/investigate something
func containsResearchKeywords(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	keywords := []string{
		"調べ", "検索", "調査", "ニュース", "最新", "について",
		"まとめ", "にまつわる", "に関する", "search", "research", "investigate",
	}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

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

// extractSearchQuery uses LLM to extract an appropriate search query from the user message.
// platform is "twitter" or "web". Falls back to simple keyword extraction if LLM fails.
func extractSearchQuery(userMsg, platform string, config *Config) string {
	// Try LLM extraction
	prompt := fmt.Sprintf(`ユーザーの発言から%s検索に適したキーワードだけを抽出せよ。
検索クエリのみ出力。説明不要。引用符不要。

例:
入力: "twitterから最新のAIニュースを検索して報告"
出力: AI ニュース 最新

入力: "ツイッターでGPT-5について何か言ってる？"
出力: GPT-5

入力: "最近のLLMのトレンドをtwitterで調べて"
出力: LLM トレンド

入力: "%s"
出力:`, platform, userMsg)

	if config != nil {
		_, query, err := callSubModel(prompt, config)
		if err == nil {
			query = strings.TrimSpace(query)
			// Remove quotes if LLM wrapped them
			query = strings.Trim(query, "\"'「」")
			if query != "" && len(query) < 200 {
				fmt.Printf("[siki] Extracted %s search query: %q from %q\n", platform, query, userMsg)
				return query
			}
		}
	}

	// Fallback: strip common noise words
	lower := userMsg
	noiseWords := []string{"twitter", "ツイッター", "ツイート", "から", "で", "を", "の", "して", "て",
		"検索", "調べて", "教えて", "報告", "まとめて", "最新の", "について", "何か", "言ってる"}
	for _, nw := range noiseWords {
		lower = strings.ReplaceAll(lower, nw, " ")
	}
	lower = strings.Join(strings.Fields(lower), " ")
	if lower == "" {
		lower = userMsg
	}
	fmt.Printf("[siki] Fallback %s search query: %q from %q\n", platform, lower, userMsg)
	return lower
}

// overrideToolArgs replaces unreliable tool arguments from the small orchestrator model
// with sensible defaults based on the user's actual message. The 1.2B model's job is
// to choose WHICH tool to call — argument generation is handled here.
func overrideToolArgs(toolName, userMsg string, originalArgs map[string]interface{}, config *Config, sendEvent func(StreamEvent)) map[string]interface{} {
	switch toolName {
	case "web_search":
		// Override query with userMsg only for first-time queries (not follow-ups)
		if userMsg != "" && !isFollowUpQuery(userMsg) {
			if q, ok := originalArgs["query"].(string); ok && q != userMsg {
				fmt.Printf("[siki] Overriding web_search query: %q → %q\n", q, userMsg)
			}
			originalArgs["query"] = userMsg
		}

	case "twitter_search":
		if q, _ := originalArgs["query"].(string); q == "" {
			originalArgs["query"] = extractSearchQuery(userMsg, "twitter", config)
		}

	case "web_fetch":
		// Extract URL from user message if model didn't provide a valid one
		url, _ := originalArgs["url"].(string)
		if url == "" || !strings.HasPrefix(url, "http") {
			// Try to find URL in user message
			for _, word := range strings.Fields(userMsg) {
				if strings.HasPrefix(word, "http://") || strings.HasPrefix(word, "https://") {
					originalArgs["url"] = word
					break
				}
			}
		}

	case "docker_run_model":
		// Extract URL from user message if not provided
		url, _ := originalArgs["url"].(string)
		if url == "" || !strings.HasPrefix(url, "http") {
			for _, word := range strings.Fields(userMsg) {
				if (strings.Contains(word, "huggingface.co/") || strings.Contains(word, "github.com/")) &&
					strings.HasPrefix(word, "http") {
					originalArgs["url"] = word
					break
				}
			}
		}
		// Use remaining text (without URL) as prompt
		prompt, _ := originalArgs["prompt"].(string)
		if prompt == "" {
			var parts []string
			for _, word := range strings.Fields(userMsg) {
				if !strings.HasPrefix(word, "http") {
					parts = append(parts, word)
				}
			}
			originalArgs["prompt"] = strings.Join(parts, " ")
		}

	case "run_code":
		// Delegate code generation to sub-model
		html, err := generateCodeWithSubModel(userMsg, config)
		if err != nil {
			fmt.Printf("[siki] Sub-model code gen failed: %v, using orchestrator HTML\n", err)
		} else {
			originalArgs["html"] = html
		}

	case "diagram":
		// For diagrams, delegate DOT generation to sub-agent (or sub-model)
		if config.SubModel != "" || hasSubAgent(config) {
			prompt := fmt.Sprintf("以下のリクエストに対して、Graphviz DOTコードのみ出力せよ（説明不要）。\nリクエスト: %s", userMsg)
			_, dotCode, err := callSubAgent(prompt, config)
			if err == nil && len(dotCode) > 10 {
				// Extract DOT code from response
				dot := dotCode
				if idx := strings.Index(dot, "```dot"); idx >= 0 {
					dot = dot[idx+6:]
					if end := strings.Index(dot, "```"); end >= 0 {
						dot = dot[:end]
					}
				} else if idx := strings.Index(dot, "```"); idx >= 0 {
					dot = dot[idx+3:]
					if nl := strings.Index(dot, "\n"); nl >= 0 {
						dot = dot[nl+1:]
					}
					if end := strings.Index(dot, "```"); end >= 0 {
						dot = dot[:end]
					}
				}
				dot = strings.TrimSpace(dot)
				if strings.Contains(dot, "digraph") || strings.Contains(dot, "graph") {
					originalArgs["dot_source"] = dot
				}
			}
		}

	case "execute_command":
		// Keep model's command but validate it's not dangerous
		// (executeTool already has safety checks, so just pass through)

	case "read_file", "write_file", "search_files", "grep":
		// File operations: trust model's args (path/content from user context)

	case "generate_image":
		// Enhance prompt: convert user's Japanese request to detailed English prompt via sub-agent
		if config.SubModel != "" || hasSubAgent(config) {
			enhancePrompt := fmt.Sprintf(`以下のユーザーリクエストから、画像生成AI用の英語プロンプトを生成せよ。
詳細で描写的な英語プロンプトのみを出力し、他の文章は書くな。
スタイル指定（digital art, infographic, illustration等）を含めること。

ユーザーリクエスト: %s`, userMsg)
			_, enhanced, err := callSubAgent(enhancePrompt, config)
			if err == nil && len(enhanced) > 10 {
				enhanced = strings.TrimSpace(enhanced)
				// Remove markdown code blocks if present
				enhanced = strings.TrimPrefix(enhanced, "```")
				enhanced = strings.TrimSuffix(enhanced, "```")
				enhanced = strings.TrimSpace(enhanced)
				if sendEvent != nil {
					sendEvent(modelThinkingEvent(fmt.Sprintf("プロンプト強化: %s", enhanced), config, hasSubAgent(config)))
				}
				originalArgs["prompt"] = enhanced
			}
		}
	}

	return originalArgs
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
	config    *Config
	messages  []Message
	threadID  string
	sendEvent func(StreamEvent) // optional: for tools that need to emit progress to the UI
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
	case "twitter_search":
		query, _ := args["query"].(string)
		if query == "" {
			return "", fmt.Errorf("query is required for twitter_search")
		}
		maxResults := 30
		if n, ok := args["max_results"].(float64); ok && n > 0 {
			maxResults = int(n)
		}
		return twitterSearch(query, maxResults, a.config)
	case "bluesky_feed":
		if !a.config.BlueskyEnabled {
			return "", fmt.Errorf("Blueskyフィードが有効化されていません。設定画面からBlueskyを有効にしてください")
		}
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: "Blueskyフィードを取得中..."})
		}
		// Load cached feed (if within 30 min, reuse; otherwise re-fetch)
		feed := loadBlueskyFeed()
		if time.Since(feed.LastFetched) > 30*time.Minute || len(feed.Posts) == 0 {
			if a.sendEvent != nil {
				a.sendEvent(StreamEvent{Type: "progress", Content: "ハンドル一覧を解決中..."})
			}
			handles, err := resolveBlueskyHandles(a.config)
			if err != nil || len(handles) == 0 {
				return "", fmt.Errorf("Blueskyハンドル取得失敗: %v", err)
			}
			if a.sendEvent != nil {
				a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("%dアカウントから投稿取得中...", len(handles))})
			}
			newPosts := fetchAllBlueskyPosts(handles)
			feed.Posts = mergeBlueskyPosts(feed.Posts, newPosts)
			feed.LastFetched = time.Now()
			saveBlueskyFeed(feed)
		}
		recentPosts := filterRecentBlueskyPosts(feed.Posts, 24*time.Hour)
		if len(recentPosts) == 0 {
			return "Blueskyフィードに最近の投稿がありません。", nil
		}
		// Show raw feed immediately
		rawFeed := formatBlueskyPosts(recentPosts, fmt.Sprintf("Bluesky AI/MLフィード（%d件）", len(recentPosts)))
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "tool_call", Name: "bluesky_feed", Result: rawFeed})
		}
		// Extract intent
		userIntent, _ := args["intent"].(string)
		if userIntent == "" {
			userIntent = a.lastUserMessage()
		}
		intentPrompt := fmt.Sprintf(`Extract the search topic from this message. Remove action/instruction words (bluesky, ブルースカイ, フィード, まとめて, 見せて, 教えて, etc). Output ONLY the topic, nothing else. If no specific topic remains, output "AI・機械学習・テクノロジー".
Message: %s
Topic:`, userIntent)
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: "テーマ抽出中..."})
		}
		extractedIntent, err := callFastModel(intentPrompt, a.config)
		if err == nil && strings.TrimSpace(extractedIntent) != "" {
			extractedIntent = strings.TrimSpace(extractedIntent)
			extractedIntent = strings.Trim(extractedIntent, "「」\"'")
			if extractedIntent != "" {
				fmt.Printf("[siki] Bluesky extracted intent: %q -> %q\n", userIntent, extractedIntent)
				userIntent = extractedIntent
			}
		}
		// デフォルトテーマ: 汎用的なリクエスト時はAI/MLテーマを設定
		if userIntent == "" || len(userIntent) < 3 {
			userIntent = "AI・機械学習・テクノロジー"
		}
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("テーマ: %s", userIntent)})
		}
		evaluated := evaluateBlueskyPostsConcurrently(recentPosts, userIntent, a.config, a.sendEvent)
		if len(evaluated) == 0 {
			return fmt.Sprintf("\n\n---\nAI選別結果: %d件中、「%s」に該当する投稿はありませんでした。", len(recentPosts), userIntent), nil
		}
		return formatEvaluatedBlueskyPosts(evaluated, fmt.Sprintf("\n\n---\n## AI選別結果: %d件中%d件を抽出（重要度順）", len(recentPosts), len(evaluated))), nil
	case "bluesky_search":
		query, _ := args["query"].(string)
		if query == "" {
			// Extract query from user message
			query = a.lastUserMessage()
			extractPrompt := fmt.Sprintf(`Extract the Bluesky search keyword from this message. Remove action words (bluesky, ブルースカイ, 検索, 探して, 調べて, etc). Output ONLY the search keyword/phrase, nothing else.
Message: %s
Keyword:`, query)
			if extracted, err := callFastModel(extractPrompt, a.config); err == nil {
				extracted = strings.TrimSpace(extracted)
				extracted = strings.Trim(extracted, "「」\"'")
				if extracted != "" {
					query = extracted
				}
			}
		}
		if query == "" {
			return "", fmt.Errorf("検索キーワードを指定してください")
		}
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("Blueskyで「%s」を検索中...", query)})
		}
		posts, err := searchBlueskyPosts(query, a.config)
		if err != nil {
			return "", fmt.Errorf("Bluesky検索失敗: %v", err)
		}
		if len(posts) == 0 {
			return fmt.Sprintf("Blueskyで「%s」に該当する投稿は見つかりませんでした。", query), nil
		}
		// Show raw results first
		rawResults := formatBlueskyPosts(posts, fmt.Sprintf("Bluesky検索: 「%s」（%d件）", query, len(posts)))
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "tool_call", Name: "bluesky_search", Result: rawResults})
		}
		// Evaluate each post individually with sub-agents
		evaluated := evaluateBlueskyPostsConcurrently(posts, query, a.config, a.sendEvent)
		if len(evaluated) == 0 {
			return fmt.Sprintf("\n\n---\n検索結果%d件中、重要度の高い投稿はありませんでした。", len(posts)), nil
		}
		return formatEvaluatedBlueskyPosts(evaluated, fmt.Sprintf("\n\n---\n## 検索結果分析: %d件中%d件を重要度順にピックアップ", len(posts), len(evaluated))), nil

	case "jetstream_search":
		query, _ := args["query"].(string)
		if query == "" {
			return "", fmt.Errorf("検索キーワードを指定してください")
		}
		days := 7
		if d, ok := args["days"].(float64); ok && d > 0 {
			days = int(d)
		}
		result, count := searchJetstreamPosts(query, days)
		if count == 0 {
			return result, nil
		}
		return result, nil

	case "twitter_timeline":
		if !hasTwitterOAuth1a(a.config) {
			return "", fmt.Errorf("Twitter OAuth 1.0a が設定されていません。設定画面からConsumer Key/Secret, Access Token/Secretを設定してください")
		}
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: "タイムラインを取得中..."})
		}
		tweets, err := fetchTwitterTimeline(a.config)
		if err != nil {
			return "", fmt.Errorf("タイムライン取得失敗: %v", err)
		}
		if len(tweets) == 0 {
			return "タイムラインにツイートがありません。", nil
		}
		// Show raw timeline immediately so user doesn't wait
		rawTimeline := formatTweets(tweets, fmt.Sprintf("タイムライン（%d件）", len(tweets)))
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "tool_call", Name: "twitter_timeline", Result: rawTimeline})
		}
		// Extract the actual topic of interest from user message
		userIntent, _ := args["intent"].(string)
		if userIntent == "" {
			userIntent = a.lastUserMessage()
		}
		// Fast model extracts the core topic
		intentPrompt := fmt.Sprintf(`Extract the search topic from this message. Remove action/instruction words (twitter, タイムライン, まとめて, 検索, 拾って, etc). Output ONLY the topic, nothing else.
Message: %s
Topic:`, userIntent)
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: "テーマ抽出中..."})
		}
		extractedIntent, err := callFastModel(intentPrompt, a.config)
		if err == nil && strings.TrimSpace(extractedIntent) != "" {
			extractedIntent = strings.TrimSpace(extractedIntent)
			// Remove quotes if LLM wraps in them
			extractedIntent = strings.Trim(extractedIntent, "「」\"'")
			if extractedIntent != "" {
				fmt.Printf("[siki] Extracted intent: %q -> %q\n", userIntent, extractedIntent)
				userIntent = extractedIntent
			}
		}
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("テーマ: %s", userIntent)})
		}
		filtered := filterTweetsByIntent(tweets, userIntent, a.config, a.sendEvent)
		if len(filtered) == 0 {
			return fmt.Sprintf("\n\n---\nAI選別結果: %d件中、「%s」に該当するツイートはありませんでした。", len(tweets), userIntent), nil
		}
		// Evaluate importance and deep-dive worthiness for each filtered tweet
		if a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("%d件を抽出。重要度を評価中...", len(filtered))})
		}
		deepDiveIdxs := evaluateAndSelectDeepDive(filtered, userIntent, a.config, a.sendEvent)

		// Fetch threads in parallel for deep-dive tweets
		type threadResult struct {
			idx  int
			data ThreadData
		}
		if len(deepDiveIdxs) > 0 && a.sendEvent != nil {
			a.sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("%d件のスレッドを深掘り中...", len(deepDiveIdxs))})
		}
		// Filter valid indices first
		var validDiveIdxs []int
		for _, idx := range deepDiveIdxs {
			if idx >= 0 && idx < len(filtered) {
				validDiveIdxs = append(validDiveIdxs, idx)
			}
		}
		threadCh := make(chan threadResult, len(validDiveIdxs))
		for _, idx := range validDiveIdxs {
			go func(i int) {
				replies, err := fetchConversationThread(filtered[i].ID, filtered[i].Author, a.config)
				if err != nil {
					fmt.Printf("[siki] Thread fetch failed for %s: %v\n", filtered[i].ID, err)
					threadCh <- threadResult{i, ThreadData{}}
					return
				}
				threadCh <- threadResult{i, ThreadData{TweetIndex: i, Replies: replies}}
			}(idx)
		}
		var threads []ThreadData
		for range validDiveIdxs {
			tr := <-threadCh
			if len(tr.data.Replies) > 0 {
				threads = append(threads, tr.data)
			}
		}
		return formatTweetsWithThreads(filtered, fmt.Sprintf("\n\n---\n## AI選別結果: %d件中%d件を抽出", len(tweets), len(filtered)), threads), nil
	case "web_fetch":
		return a.webFetch(args["url"].(string))
	case "web_images":
		return a.webImages(args["url"].(string))
	case "diagram":
		title := "Diagram"
		if t, ok := args["title"].(string); ok {
			title = t
		}
		dotCode := ""
		if d, ok := args["dot_code"].(string); ok && d != "" {
			dotCode = d
		} else if d, ok := args["dot_source"].(string); ok && d != "" {
			dotCode = d
		}
		if dotCode == "" {
			return "", fmt.Errorf("diagram: dot_source or dot_code argument is required")
		}
		return a.generateDiagram(dotCode, title)
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
	case "sandbox_exec":
		code, _ := args["code"].(string)
		if code == "" {
			return "", fmt.Errorf("code is required")
		}
		language, _ := args["language"].(string)
		timeout := 30
		if t, ok := args["timeout"].(float64); ok {
			timeout = int(t)
		}
		if timeout > 300 {
			timeout = 300
		}
		result, files, err := zerobootExecWithFiles(code, language, timeout)
		if err != nil {
			return "", err
		}
		var sb strings.Builder
		if result.ExitCode != 0 {
			sb.WriteString(fmt.Sprintf("Exit code: %d\n", result.ExitCode))
		}
		sb.WriteString(fmt.Sprintf("(fork: %.1fms, exec: %.1fms)\n", result.ForkTimeMs, result.ExecTimeMs))
		if result.Stdout != "" {
			sb.WriteString("\n" + result.Stdout)
		}
		if result.Stderr != "" {
			sb.WriteString("\nSTDERR:\n" + result.Stderr)
		}
		// Save any output files to playground
		for name, data := range files {
			outPath := filepath.Join(playgroundDir, name)
			os.WriteFile(outPath, data, 0644)
			sb.WriteString(fmt.Sprintf("\n\nFile saved: /playground/%s", name))
			ext := strings.ToLower(filepath.Ext(name))
			if ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".gif" {
				sb.WriteString(fmt.Sprintf("\n![%s](/playground/%s)", name, name))
			}
		}
		return sb.String(), nil
	case "docker_exec":
		timeout := 300
		if t, ok := args["timeout"].(float64); ok {
			timeout = int(t)
		}
		return dockerExec(args["command"].(string), timeout)
	case "docker_run_model":
		url, _ := args["url"].(string)
		if url == "" {
			return "", fmt.Errorf("url is required for docker_run_model")
		}
		prompt, _ := args["prompt"].(string)
		return dockerRunModel(url, prompt, a.config, a.sendEvent)
	case "generate_image":
		prompt, _ := args["prompt"].(string)
		if prompt == "" {
			return "", fmt.Errorf("prompt is required for generate_image")
		}
		width := 512
		height := 512
		if w, ok := args["width"].(float64); ok && w > 0 {
			width = int(w)
		}
		if h, ok := args["height"].(float64); ok && h > 0 {
			height = int(h)
		}
		urlPath, err := generateImage(prompt, width, height, a.config)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Image generated successfully: ![Generated Image](%s)\n\nURL: %s", urlPath, urlPath), nil
	case "generate_video":
		prompt, _ := args["prompt"].(string)
		if prompt == "" {
			return "", fmt.Errorf("prompt is required for generate_video")
		}
		numFrames := 33
		if nf, ok := args["num_frames"].(float64); ok && nf > 0 {
			numFrames = int(nf)
		}
		urlPath, err := generateVideo(prompt, numFrames, a.config)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Video generated successfully: [Generated Video](%s)\n\nURL: %s\n\n<video src=\"%s\" controls autoplay loop style=\"max-width:100%%\"></video>", urlPath, urlPath, urlPath), nil
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
	case "use_skill":
		skillName, _ := args["name"].(string)
		fileName, _ := args["file"].(string)
		if skillName == "" {
			return "Error: skill name required", nil
		}
		if fileName != "" {
			content := getSkillFile(skillName, fileName)
			if content == "" {
				return fmt.Sprintf("Error: file '%s' not found in skill '%s'", fileName, skillName), nil
			}
			return fmt.Sprintf("# Skill file: %s/%s\n\n%s", skillName, fileName, content), nil
		}
		content := getSkillContent(skillName)
		if content == "" {
			return fmt.Sprintf("Error: skill '%s' not found. Use list_skills to see available skills.", skillName), nil
		}
		return fmt.Sprintf("# Skill activated: %s\n\nFollow the instructions below:\n\n%s", skillName, content), nil
	case "list_skills":
		skillsMu.RLock()
		defer skillsMu.RUnlock()
		if len(loadedSkills) == 0 {
			return "No skills installed. Skills can be installed via the Settings > Skills panel or /api/skills/install API.", nil
		}
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("## Installed Skills (%d)\n\n", len(loadedSkills)))
		for _, s := range loadedSkills {
			src := ""
			if s.Source != "" {
				src = fmt.Sprintf(" [%s]", s.Source)
			}
			sb.WriteString(fmt.Sprintf("- **%s**%s: %s\n", s.Name, src, s.Description))
			if len(s.Files) > 0 {
				sb.WriteString(fmt.Sprintf("  Files: %s\n", strings.Join(s.Files, ", ")))
			}
		}
		return sb.String(), nil
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

	now := time.Now()
	searchQuery := query
	if isTimeSensitive {
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
	}

	// Try Scrapling first
	scrapResults, err := scraplingSearch(searchQuery, 10)
	if err == nil && len(scrapResults) > 0 {
		var results []string
		for _, r := range scrapResults {
			title := r["title"]
			resultURL := r["url"]
			snippet := r["snippet"]
			shortURL := resultURL
			if idx := strings.Index(resultURL, "://"); idx >= 0 {
				shortURL = resultURL[idx+3:]
				if slashIdx := strings.Index(shortURL, "/"); slashIdx >= 0 {
					shortURL = shortURL[:slashIdx]
				}
			}
			if len(shortURL) > 15 {
				shortURL = shortURL[:12] + "..."
			}
			result := fmt.Sprintf("**%s**\n[%s](%s)\n%s\n", title, shortURL, resultURL, snippet)
			results = append(results, result)
		}
		header := fmt.Sprintf("Search results for: %s", query)
		if searchQuery != query {
			header += fmt.Sprintf(" (searched: %s)", searchQuery)
		}
		return fmt.Sprintf("%s\n\n%s", header, strings.Join(results, "\n---\n")), nil
	}
	fmt.Printf("[siki] webSearch: scrapling failed (%v), falling back to Go\n", err)

	// Fallback: direct Go HTTP with DuckDuckGo
	dateFilter := ""
	if isTimeSensitive {
		dateFilter = "&df=m"
	}
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

	html := string(body)
	var results []string
	parts := strings.Split(html, "result__a")
	for i, part := range parts {
		if i == 0 || i > 10 {
			continue
		}
		urlStart := strings.Index(part, "href=\"")
		if urlStart == -1 {
			continue
		}
		urlStart += 6
		urlEnd := strings.Index(part[urlStart:], "\"")
		if urlEnd == -1 {
			continue
		}
		resultURL := part[urlStart : urlStart+urlEnd]
		if strings.Contains(resultURL, "uddg=") {
			if idx := strings.Index(resultURL, "uddg="); idx != -1 {
				resultURL = resultURL[idx+5:]
				if ampIdx := strings.Index(resultURL, "&"); ampIdx != -1 {
					resultURL = resultURL[:ampIdx]
				}
				resultURL = strings.ReplaceAll(resultURL, "%3A", ":")
				resultURL = strings.ReplaceAll(resultURL, "%2F", "/")
				resultURL = strings.ReplaceAll(resultURL, "%3F", "?")
				resultURL = strings.ReplaceAll(resultURL, "%3D", "=")
				resultURL = strings.ReplaceAll(resultURL, "%26", "&")
			}
		}
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
		title = cleanHTMLEntities(title)
		snippet = cleanHTMLEntities(snippet)
		if title != "" || resultURL != "" {
			shortURL := resultURL
			if idx := strings.Index(resultURL, "://"); idx >= 0 {
				shortURL = resultURL[idx+3:]
				if slashIdx := strings.Index(shortURL, "/"); slashIdx >= 0 {
					shortURL = shortURL[:slashIdx]
				}
			}
			if len(shortURL) > 15 {
				shortURL = shortURL[:12] + "..."
			}
			result := fmt.Sprintf("**%s**\n[%s](%s)\n%s\n", title, shortURL, resultURL, snippet)
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

// webFetchQuick is a faster version of webFetch with 10s timeout for auto-fetch.
func (a *Agent) webFetchQuick(targetURL string) (string, error) {
	// URL-decode percent-encoded characters in the URL
	if decoded, err := url.QueryUnescape(targetURL); err == nil {
		targetURL = decoded
	}
	fmt.Printf("[siki] webFetchQuick: %s\n", targetURL)

	// Try Scrapling first for better HTML parsing
	_, text, _, err := scraplingFetch(targetURL, 5000, false)
	if err == nil && len(text) > 0 {
		return text, nil
	}
	fmt.Printf("[siki] webFetchQuick: scrapling failed (%v), falling back to Go\n", err)

	// Fallback: direct Go HTTP
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	limited := io.LimitReader(resp.Body, 100*1024)
	body, err := io.ReadAll(limited)
	if err != nil {
		return "", err
	}

	goText := extractTextFromHTML(string(body))
	if len(goText) > 5000 {
		goText = goText[:5000]
	}
	return goText, nil
}

func (a *Agent) webFetch(targetURL string) (string, error) {
	fmt.Printf("[siki] webFetch: %s\n", targetURL)

	// Try Scrapling first
	title, text, _, err := scraplingFetch(targetURL, 15000, false)
	if err == nil && len(text) > 0 {
		header := fmt.Sprintf("Content from %s", targetURL)
		if title != "" {
			header = fmt.Sprintf("%s (%s)", title, targetURL)
		}
		return fmt.Sprintf("%s:\n\n%s", header, text), nil
	}
	fmt.Printf("[siki] webFetch: scrapling failed (%v), falling back to Go\n", err)

	// Fallback: direct Go HTTP
	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequest("GET", targetURL, nil)
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

	html := string(body)
	goText := extractTextFromHTML(html)
	if len(goText) > 15000 {
		goText = goText[:15000] + "\n\n... (truncated)"
	}

	return fmt.Sprintf("Content from %s:\n\n%s", targetURL, goText), nil
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

// describeImageForCharacter sends a single image to the vision model with a custom prompt
// for extracting character appearance details. Returns the description text.
func describeImageForCharacter(b64image string, prompt string, visionModel string, config *Config) string {
	if b64image == "" || visionModel == "" {
		return ""
	}

	endpoint := strings.TrimSuffix(config.primaryProvider().Endpoint, "/v1")

	reqBody := map[string]interface{}{
		"model": visionModel,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
				"images":  []string{b64image},
			},
		},
		"stream": false,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		fmt.Printf("[siki] describeImageForCharacter: marshal error: %v\n", err)
		return ""
	}

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Post(endpoint+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Printf("[siki] describeImageForCharacter: request error: %v\n", err)
		return ""
	}
	defer resp.Body.Close()

	var chatResp struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		fmt.Printf("[siki] describeImageForCharacter: decode error: %v\n", err)
		return ""
	}

	return strings.TrimSpace(chatResp.Message.Content)
}

// subModelEndpoint returns the endpoint URL for the sub-model.
// If SubModelEndpoint is set, use it; otherwise fall back based on backend type.
func subModelEndpoint(config *Config) string {
	if config.SubModelEndpoint != "" {
		return strings.TrimSuffix(config.SubModelEndpoint, "/")
	}
	// If sub-model uses ollama backend, default to ollama endpoint
	if config.SubModelBackend == "ollama" || config.SubModelBackend == "" {
		return "http://localhost:11434"
	}
	return strings.TrimSuffix(config.primaryProvider().Endpoint, "/v1")
}

// isSubModelVLLM returns true if the sub-model backend is vllm (OpenAI-compatible API).
func isSubModelVLLM(config *Config) bool {
	return config.SubModelBackend == "vllm"
}

// callOrchestratorGenerate calls the orchestrator model (falls back to sub-model if not configured).
func callOrchestratorGenerate(prompt string, maxTokens int, timeout time.Duration, config *Config) (string, error) {
	model := config.orchestratorModel()
	endpoint := config.orchestratorEndpoint()
	isVLLM := config.orchestratorBackend() == "vllm"

	if isVLLM {
		return callVLLMGenerate(model, prompt, maxTokens, timeout, endpoint)
	}

	// Ollama native API
	reqBody := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("orchestrator request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
		Thinking string `json:"thinking"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}
	content := genResp.Response
	// Strip <think>...</think> tags that some models embed in the response
	if idx := strings.Index(content, "</think>"); idx >= 0 {
		content = strings.TrimSpace(content[idx+len("</think>"):])
	}
	// If response is empty but thinking has content, the model may have
	// embedded the actual answer inside thinking text — extract it
	if content == "" && genResp.Thinking != "" {
		thinking := genResp.Thinking
		// Look for JSON in thinking text (qwen3.5 sometimes puts JSON in thinking)
		if idx := strings.Index(thinking, "{"); idx >= 0 {
			content = thinking[idx:]
		} else {
			content = thinking
		}
	}
	return content, nil
}

// ============================================================================
// Sub-Agent: powerful model for summarization, code generation, analysis
// Falls back to sub-model when sub-agent is not configured.
// ============================================================================

// hasSubAgent returns true if a sub-agent model is configured.
func hasSubAgent(config *Config) bool {
	return config.SubAgent != ""
}

// subAgentEndpoint returns the endpoint URL for the sub-agent.
func subAgentEndpoint(config *Config) string {
	if config.SubAgentEndpoint != "" {
		return strings.TrimSuffix(config.SubAgentEndpoint, "/")
	}
	// Fall back to sub-model endpoint, then primary endpoint
	return subModelEndpoint(config)
}

// isSubAgentVLLM returns true if the sub-agent backend is vllm.
func isSubAgentVLLM(config *Config) bool {
	if config.SubAgentBackend != "" {
		return config.SubAgentBackend == "vllm"
	}
	return true // default to vllm for sub-agent (typically large models)
}

// callSubAgent calls the sub-agent model for complex tasks.
// Falls back to callSubModel if no sub-agent is configured.
func callSubAgent(prompt string, config *Config) (thinking string, response string, err error) {
	if !hasSubAgent(config) {
		return callSubModel(prompt, config)
	}

	endpoint := subAgentEndpoint(config)

	if isSubAgentVLLM(config) {
		content, err := callVLLMGenerate(config.SubAgent, prompt, 32768, 300*time.Second, endpoint)
		if err != nil {
			fmt.Printf("[siki] Sub-agent call failed, falling back to sub-model: %v\n", err)
			return callSubModel(prompt, config)
		}
		return "", content, nil
	}

	// Ollama path for sub-agent
	reqBody := map[string]interface{}{
		"model":      config.SubAgent,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": 32768,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Printf("[siki] Sub-agent call failed, falling back to sub-model: %v\n", err)
		return callSubModel(prompt, config)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
		Thinking string `json:"thinking"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", "", fmt.Errorf("sub-agent decode error: %w", err)
	}

	content := genResp.Response
	thinking = genResp.Thinking
	if content == "" && thinking != "" {
		content = thinking
		thinking = ""
	}
	// Strip inline <think> tags
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			thinking = strings.TrimSpace(content[ti+7 : te])
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}
	return thinking, content, nil
}

// streamSubAgentGenerate streams from the sub-agent (or falls back to sub-model streaming).
func streamSubAgentGenerate(prompt string, config *Config, sendEvent func(StreamEvent)) (string, error) {
	if !hasSubAgent(config) {
		// Fall back to sub-model streaming
		return streamSubModelSummarize("", "none", prompt, config, sendEvent)
	}

	endpoint := subAgentEndpoint(config)

	if isSubAgentVLLM(config) {
		return streamVLLMGenerate(config.SubAgent, prompt, 32768, 300*time.Second, endpoint, sendEvent)
	}

	// Ollama streaming for sub-agent
	reqBody := map[string]interface{}{
		"model":      config.SubAgent,
		"prompt":     prompt,
		"stream":     true,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": 32768,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("sub-agent stream error: %w", err)
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	inThinking := false
	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chunk struct {
			Response string `json:"response"`
			Thinking string `json:"thinking"`
			Done     bool   `json:"done"`
		}
		if err := decoder.Decode(&chunk); err != nil {
			break
		}
		// Capture thinking tokens as thinking events (instead of discarding)
		if chunk.Thinking != "" {
			inThinking = true
			sendEvent(StreamEvent{Type: "thinking", Content: chunk.Thinking})
			continue
		}
		if inThinking && chunk.Response != "" {
			inThinking = false
		}
		if chunk.Response != "" {
			text := chunk.Response
			if strings.Contains(text, "<think>") {
				inThinking = true
				sendEvent(StreamEvent{Type: "thinking", Content: text})
				continue
			}
			if strings.Contains(text, "</think>") {
				inThinking = false
				continue
			}
			if inThinking {
				sendEvent(StreamEvent{Type: "thinking", Content: text})
				continue
			}
			fullResponse.WriteString(text)
			sendEvent(StreamEvent{Type: "content", Content: text})
		}
		if chunk.Done {
			break
		}
	}
	return fullResponse.String(), nil
}

// callVLLMGenerate calls a model via vllm's OpenAI-compatible /v1/chat/completions endpoint.
// Returns the generated text with <think> tags stripped.
func callVLLMGenerate(model, prompt string, maxTokens int, timeout time.Duration, endpoint string) (string, error) {
	ep := strings.TrimSuffix(endpoint, "/")
	if !strings.HasSuffix(ep, "/v1") {
		ep = ep + "/v1"
	}

	reqBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
		"stream":     false,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: timeout}
	resp, err := client.Post(ep+"/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("vllm request error: %w", err)
	}
	defer resp.Body.Close()

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", fmt.Errorf("vllm decode error: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("vllm returned no choices")
	}

	content := chatResp.Choices[0].Message.Content

	// Strip inline <think> tags
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}
	return content, nil
}

// streamVLLMGenerate streams a response from vllm's OpenAI-compatible /v1/chat/completions endpoint.
func streamVLLMGenerate(model, prompt string, maxTokens int, timeout time.Duration, endpoint string, sendEvent func(StreamEvent)) (string, error) {
	ep := strings.TrimSuffix(endpoint, "/")
	if !strings.HasSuffix(ep, "/v1") {
		ep = ep + "/v1"
	}

	reqBody := map[string]interface{}{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"max_tokens": maxTokens,
		"stream":     true,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: timeout}
	resp, err := client.Post(ep+"/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("vllm stream request error: %w", err)
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	inThinking := false
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		text := chunk.Choices[0].Delta.Content
		if text == "" {
			continue
		}
		// Capture <think> blocks as thinking events (instead of discarding)
		if strings.Contains(text, "<think>") {
			inThinking = true
			// Send any content after <think> tag
			if idx := strings.Index(text, "<think>"); idx > 0 {
				fullResponse.WriteString(text[:idx])
				sendEvent(StreamEvent{Type: "content", Content: text[:idx]})
			}
			after := ""
			if idx := strings.Index(text, "<think>"); idx+7 < len(text) {
				after = text[idx+7:]
			}
			if after != "" {
				sendEvent(StreamEvent{Type: "thinking", Content: after})
			}
			continue
		}
		if strings.Contains(text, "</think>") {
			inThinking = false
			// Send any content before </think> as thinking
			if idx := strings.Index(text, "</think>"); idx > 0 {
				sendEvent(StreamEvent{Type: "thinking", Content: text[:idx]})
			}
			// Send any content after </think> as content
			if idx := strings.Index(text, "</think>"); idx+8 < len(text) {
				after := text[idx+8:]
				fullResponse.WriteString(after)
				sendEvent(StreamEvent{Type: "content", Content: after})
			}
			continue
		}
		if inThinking {
			sendEvent(StreamEvent{Type: "thinking", Content: text})
			continue
		}
		fullResponse.WriteString(text)
		sendEvent(StreamEvent{Type: "content", Content: text})
	}

	return fullResponse.String(), nil
}

// streamOllamaGenerate streams a response from Ollama's /api/generate endpoint.
// Handles <think></think> blocks from thinking models (gpt-oss).
func streamOllamaGenerate(model, prompt string, maxTokens int, timeout time.Duration, sendEvent func(StreamEvent)) (string, error) {
	reqBody := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
		"stream": true,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Post("http://localhost:11434/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("ollama stream error: %w", err)
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	inThinking := false
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		var chunk struct {
			Response string `json:"response"`
			Done     bool   `json:"done"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		text := chunk.Response
		if text == "" {
			if chunk.Done {
				break
			}
			continue
		}
		// Handle <think> blocks
		if strings.Contains(text, "<think>") {
			inThinking = true
			if idx := strings.Index(text, "<think>"); idx > 0 {
				fullResponse.WriteString(text[:idx])
				sendEvent(StreamEvent{Type: "content", Content: text[:idx]})
			}
			after := ""
			if idx := strings.Index(text, "<think>"); idx+7 < len(text) {
				after = text[idx+7:]
			}
			if after != "" {
				sendEvent(StreamEvent{Type: "thinking", Content: after})
			}
			continue
		}
		if strings.Contains(text, "</think>") {
			inThinking = false
			if idx := strings.Index(text, "</think>"); idx > 0 {
				sendEvent(StreamEvent{Type: "thinking", Content: text[:idx]})
			}
			if idx := strings.Index(text, "</think>"); idx+8 < len(text) {
				after := text[idx+8:]
				fullResponse.WriteString(after)
				sendEvent(StreamEvent{Type: "content", Content: after})
			}
			continue
		}
		if inThinking {
			sendEvent(StreamEvent{Type: "thinking", Content: text})
			continue
		}
		fullResponse.WriteString(text)
		sendEvent(StreamEvent{Type: "content", Content: text})
	}
	return fullResponse.String(), nil
}

// alternateSubModels returns the two sub-models available for retry alternation.
// Primary is config.SubModel (gpt-oss), secondary is also gpt-oss.
func alternateSubModels(config *Config) (primary, secondary string) {
	primary = config.SubModel
	if primary == "" {
		primary = "gpt-oss:latest"
	}
	secondary = "gpt-oss:latest"
	return
}

// pickRetryModel picks a model for retry.
// attemptNum 0 = primary model (already tried), 1+ = alternate.
func pickRetryModel(config *Config, attemptNum int) string {
	primary, secondary := alternateSubModels(config)
	if attemptNum%2 == 0 {
		return primary
	}
	return secondary
}

// callSubModel calls the configured sub-model via Ollama or vllm.
func callSubModel(prompt string, config *Config) (thinking string, response string, err error) {
	return callSubModelWith(prompt, config, "")
}

// callFastModel calls a fast, lightweight model (LFM2.5) for quick tasks like
// keyword extraction and intent parsing. Falls back to sub-model if unavailable.
const fastModelName = "hf.co/unsloth/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"

func callFastModel(prompt string, config *Config, maxTokens ...int) (string, error) {
	numPredict := 300
	if len(maxTokens) > 0 && maxTokens[0] > 0 {
		numPredict = maxTokens[0]
	}
	reqBody := map[string]interface{}{
		"model":  fastModelName,
		"prompt": prompt,
		"stream": false,
		"options": map[string]interface{}{
			"num_predict": numPredict,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post("http://localhost:11434/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		// Fallback to sub-model
		fmt.Printf("[siki] callFastModel failed (%v), falling back to sub-model\n", err)
		_, content, err2 := callSubModel(prompt, config)
		return content, err2
	}
	defer resp.Body.Close()
	var genResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", err
	}
	return strings.TrimSpace(genResp.Response), nil
}

// callSubModelWith calls a specific model (or default sub-model if modelOverride is empty).
// It handles <think></think> tag extraction, returning (thinking, response).
func callSubModelWith(prompt string, config *Config, modelOverride string, timeout ...time.Duration) (thinking string, response string, err error) {
	model := config.SubModel
	if modelOverride != "" {
		model = modelOverride
	}
	if model == "" {
		return "", "", fmt.Errorf("no sub-model configured")
	}

	timeoutDur := 120 * time.Second
	if len(timeout) > 0 && timeout[0] > 0 {
		timeoutDur = timeout[0]
	}

	endpoint := subModelEndpoint(config)

	// Use vllm (OpenAI-compatible) API if configured
	if modelOverride == "" && isSubModelVLLM(config) {
		content, err := callVLLMGenerate(model, prompt, 32768, timeoutDur, endpoint)
		if err != nil {
			return "", "", err
		}
		return "", content, nil
	}

	// Ollama native API
	reqBody := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": 32768,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: timeoutDur}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", "", fmt.Errorf("sub-model request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
		Thinking string `json:"thinking"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", "", fmt.Errorf("sub-model decode error: %w", err)
	}

	content := genResp.Response
	thinking = genResp.Thinking

	// Thinking models put content in "thinking" field with empty "response"
	if content == "" && thinking != "" {
		content = thinking
		thinking = ""
	}

	// Strip inline <think> tags (some models embed them in response)
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			thinking = strings.TrimSpace(content[ti+7 : te])
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}

	return thinking, content, nil
}

// generateCodeWithSubModel delegates code generation to the sub-agent (or sub-model).
// The orchestrator decides WHAT tool to call, the sub-agent generates the actual code.
func generateCodeWithSubModel(userRequest string, config *Config) (string, error) {
	if config.SubModel == "" && config.SubAgent == "" {
		return "", fmt.Errorf("no sub-model or sub-agent configured")
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

	// Use sub-agent for code generation if available
	if hasSubAgent(config) {
		fmt.Printf("[siki] Using sub-agent (%s) for code generation\n", config.SubAgent)
		_, content, err := callSubAgent(prompt, config)
		if err != nil {
			return "", fmt.Errorf("sub-agent code gen error: %w", err)
		}
		// Extract HTML from response
		if idx := strings.Index(content, "<html"); idx >= 0 {
			content = content[idx:]
			if end := strings.LastIndex(content, "</html>"); end >= 0 {
				content = content[:end+7]
			}
		} else if idx := strings.Index(content, "<!DOCTYPE"); idx >= 0 {
			content = content[idx:]
		}
		return content, nil
	}

	endpoint := subModelEndpoint(config)

	var content string
	if isSubModelVLLM(config) {
		var err2 error
		content, err2 = callVLLMGenerate(config.SubModel, prompt, 32768, 600*time.Second, endpoint)
		if err2 != nil {
			return "", fmt.Errorf("sub-model request error: %w", err2)
		}
	} else {
		reqBody := map[string]interface{}{
			"model":      config.SubModel,
			"prompt":     prompt,
			"stream":     false,
			"keep_alive": -1,
			"options": map[string]interface{}{
				"num_predict": 32768,
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
			Thinking string `json:"thinking"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
			return "", fmt.Errorf("sub-model decode error: %w", err)
		}

		// Thinking models put content in "thinking" field with empty "response"
		content = genResp.Response
		if content == "" && genResp.Thinking != "" {
			content = genResp.Thinking
		}

		// Strip <think>...</think> tags
		if ti := strings.Index(content, "<think>"); ti >= 0 {
			if te := strings.Index(content, "</think>"); te > ti {
				content = strings.TrimSpace(content[:ti] + content[te+8:])
			}
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

// ============================================================================
// Dual-Model Pipeline: lfm (fast ack) + gpt-oss (real orchestration)
// ============================================================================

// callOllamaGenerate calls any model via Ollama's /api/generate or vllm's OpenAI-compatible endpoint.
// Strips <think> tags from response. Returns the generated text.
func callOllamaGenerate(model, prompt string, maxTokens int, timeout time.Duration, config *Config) (string, error) {
	endpoint := subModelEndpoint(config)

	// Use vllm (OpenAI-compatible) API if configured
	if isSubModelVLLM(config) {
		return callVLLMGenerate(model, prompt, maxTokens, timeout, endpoint)
	}

	// Ollama native API
	reqBody := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: timeout}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("generate request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
		Thinking string `json:"thinking"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}

	// Thinking models put content in "thinking" field with empty "response"
	content := genResp.Response
	if content == "" && genResp.Thinking != "" {
		content = genResp.Thinking
	}

	// Strip inline <think> tags (some models embed them in response)
	if ti := strings.Index(content, "<think>"); ti >= 0 {
		if te := strings.Index(content, "</think>"); te > ti {
			content = strings.TrimSpace(content[:ti] + content[te+8:])
		}
	}
	return content, nil
}

// OrchestratorDecision represents gpt-oss's decision on how to handle a request
type OrchestratorDecision struct {
	Tool     string                 `json:"tool"`
	Args     map[string]interface{} `json:"args,omitempty"`
	Response string                 `json:"response,omitempty"`
}

// ============================================================================
// Plan Mode: TODO list management for complex multi-step tasks
// ============================================================================

// PlanTask represents a single task in a plan
type PlanTask struct {
	ID          int    `json:"id"`
	Description string `json:"description"`
	Status      string `json:"status"` // "pending", "in_progress", "completed", "failed"
	Tool        string `json:"tool,omitempty"`
	Result      string `json:"result,omitempty"`
	ImagePrompt string `json:"image_prompt,omitempty"` // Pre-generated prompt for generate_image tasks (e.g., comic panels)
}

// Plan represents a multi-step execution plan
type Plan struct {
	Goal      string     `json:"goal"`
	Tasks     []PlanTask `json:"tasks"`
	CreatedAt string     `json:"created_at"`
	Status    string     `json:"status"` // "planning", "executing", "completed", "failed"
}

// planDir returns the directory for storing plan files
func planDir() string {
	home, _ := os.UserHomeDir()
	dir := filepath.Join(home, ".siki", "plans")
	os.MkdirAll(dir, 0755)
	return dir
}

// savePlan writes a plan to a JSON file
func savePlan(plan *Plan, planID string) error {
	path := filepath.Join(planDir(), planID+".json")
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// loadPlan reads a plan from a JSON file
func loadPlan(planID string) (*Plan, error) {
	path := filepath.Join(planDir(), planID+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var plan Plan
	return &plan, json.Unmarshal(data, &plan)
}

// isComicRequest detects if user wants a multi-panel comic (4コマ漫画 etc.)
func isComicRequest(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	keywords := []string{"4コマ", "４コマ", "漫画", "マンガ", "コミック", "comic"}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// createComicPlan creates a specialized plan for 4-panel comic generation.
// Flow: character design → reference image → vision description → 4 panels
func createComicPlan(userMsg string, messages []Message, config *Config) (*Plan, error) {
	// Extract the topic from user message
	topic := userMsg

	fmt.Printf("[siki] Creating 4-panel comic plan for: %s\n", topic)

	// Step 1: Ask sub-model to write a 4-panel scenario with character designs
	scenarioPrompt := fmt.Sprintf(`ユーザーが「%s」というリクエストをしました。
まずテーマについてあなたの知識を使って情報を整理し、キャラクターをデザインし、4コマ漫画のシナリオ（起承転結）を作成せよ。

以下のJSON形式のみ出力せよ:
{
  "topic_summary": "テーマの簡単な説明（1-2文）",
  "characters": [
    {
      "name": "キャラクター名",
      "appearance": "外見の詳細（髪型、髪色、目の色、服装、体型、特徴的なアクセサリーなど）を英語で具体的に記述",
      "personality": "性格の簡単な説明（日本語）"
    }
  ],
  "character_design_prompt": "メインキャラクター全員を1枚の画像に描いた英語プロンプト。character reference sheet style, full body, front view, white background, detailed features, anime style",
  "panels": [
    {"id": 1, "title": "起", "scene": "シーンの描写（何が描かれているか）", "dialogue": "セリフ（あれば）", "image_prompt": "英語の画像生成プロンプト。comic panel style, manga style を含めること。キャラクターの外見描写も含めること"},
    {"id": 2, "title": "承", "scene": "...", "dialogue": "...", "image_prompt": "..."},
    {"id": 3, "title": "転", "scene": "...", "dialogue": "...", "image_prompt": "..."},
    {"id": 4, "title": "結", "scene": "...", "dialogue": "...", "image_prompt": "..."}
  ]
}

注意:
- キャラクターデザインを最初に決め、各コマで一貫した外見にすること
- characters配列にはメインキャラクター（1-3人程度）を定義すること
- character_design_promptは全キャラクターを1枚に描くための英語プロンプト
- appearanceは英語で、髪型・髪色・目の色・服装・体型を具体的に書くこと
- 4コマ漫画は起承転結の構造を持つこと
- 各コマのimage_promptに、キャラクターのappearanceの詳細を必ず含めること
- comic panel style, simple background, bold outlines を含めること
- テーマに関する正確な情報を反映すること`, topic)

	var scenarioJSON string
	var err error
	if hasSubAgent(config) {
		_, scenarioJSON, err = callSubAgent(scenarioPrompt, config)
	} else {
		scenarioJSON, err = callOrchestratorGenerate(scenarioPrompt, 4096, 600*time.Second, config)
	}
	if err != nil {
		return nil, fmt.Errorf("comic scenario generation failed: %w", err)
	}

	// Parse scenario JSON
	scenarioJSON = strings.TrimSpace(scenarioJSON)
	// Strip any <think>...</think> tags from the response
	if idx := strings.Index(scenarioJSON, "</think>"); idx >= 0 {
		scenarioJSON = strings.TrimSpace(scenarioJSON[idx+len("</think>"):])
	}
	if idx := strings.Index(scenarioJSON, "{"); idx >= 0 {
		scenarioJSON = scenarioJSON[idx:]
		depth := 0
		for i, ch := range scenarioJSON {
			if ch == '{' {
				depth++
			}
			if ch == '}' {
				depth--
				if depth == 0 {
					scenarioJSON = scenarioJSON[:i+1]
					break
				}
			}
		}
	} else {
		fmt.Printf("[siki] Comic scenario: no JSON found in response (len=%d): %.100s\n", len(scenarioJSON), scenarioJSON)
		return nil, fmt.Errorf("comic scenario: LLM did not return JSON")
	}

	var scenario struct {
		TopicSummary         string `json:"topic_summary"`
		Characters           []struct {
			Name       string `json:"name"`
			Appearance string `json:"appearance"`
			Personality string `json:"personality"`
		} `json:"characters"`
		CharacterDesignPrompt string `json:"character_design_prompt"`
		Panels               []struct {
			ID          int             `json:"id"`
			Title       string          `json:"title"`
			Scene       string          `json:"scene"`
			Dialogue    json.RawMessage `json:"dialogue"`
			ImagePrompt string          `json:"image_prompt"`
		} `json:"panels"`
	}

	if err := json.Unmarshal([]byte(scenarioJSON), &scenario); err != nil {
		fmt.Printf("[siki] Comic scenario JSON parse failed: %v, raw: %s\n", err, scenarioJSON[:min(len(scenarioJSON), 200)])
		return nil, fmt.Errorf("comic scenario parse failed: %w", err)
	}

	// Helper: extract dialogue as string regardless of JSON type (string, array, etc.)
	extractDialogue := func(raw json.RawMessage) string {
		if len(raw) == 0 {
			return ""
		}
		// Try as string first
		var s string
		if json.Unmarshal(raw, &s) == nil {
			return s
		}
		// Try as array of strings
		var arr []string
		if json.Unmarshal(raw, &arr) == nil {
			return strings.Join(arr, " / ")
		}
		// Fallback: return raw trimmed
		return strings.Trim(string(raw), "\"")
	}

	if len(scenario.Panels) < 4 {
		return nil, fmt.Errorf("comic scenario has %d panels, need 4", len(scenario.Panels))
	}

	// Build character appearance description for prompt embedding
	var charDesc strings.Builder
	for _, c := range scenario.Characters {
		charDesc.WriteString(fmt.Sprintf("%s: %s. ", c.Name, c.Appearance))
	}
	charAppearance := strings.TrimSpace(charDesc.String())

	// Build plan: scenario + character design image + describe image + 4 panels
	plan := &Plan{
		Goal:      userMsg,
		CreatedAt: time.Now().Format(time.RFC3339),
		Status:    "planning",
	}

	// Task 1: Display the scenario and character designs
	var scenarioText strings.Builder
	scenarioText.WriteString(fmt.Sprintf("## 4コマ漫画シナリオ\n**テーマ:** %s\n\n", scenario.TopicSummary))

	if len(scenario.Characters) > 0 {
		scenarioText.WriteString("### キャラクターデザイン\n")
		for _, c := range scenario.Characters {
			scenarioText.WriteString(fmt.Sprintf("- **%s**: %s（%s）\n", c.Name, c.Appearance, c.Personality))
		}
		scenarioText.WriteString("\n")
	}

	for _, p := range scenario.Panels {
		scenarioText.WriteString(fmt.Sprintf("### %d. %s\n", p.ID, p.Title))
		scenarioText.WriteString(fmt.Sprintf("**シーン:** %s\n", p.Scene))
		dlg := extractDialogue(p.Dialogue)
		if dlg != "" {
			scenarioText.WriteString(fmt.Sprintf("**セリフ:** %s\n", dlg))
		}
		scenarioText.WriteString("\n")
	}

	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          1,
		Description: scenarioText.String(),
		Status:      "pending",
		Tool:        "summarize",
	})

	// Task 2: Generate character reference image
	charDesignPrompt := scenario.CharacterDesignPrompt
	if charDesignPrompt == "" {
		charDesignPrompt = fmt.Sprintf("character reference sheet, full body, front view, white background, anime style, detailed features, %s", charAppearance)
	}
	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          2,
		Description: "キャラクターデザイン参照画像を生成",
		Status:      "pending",
		Tool:        "generate_image",
		ImagePrompt: charDesignPrompt,
	})

	// Task 3: Use vision model to extract detailed character description from reference image
	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          3,
		Description: "参照画像からキャラクター外見を詳細解析",
		Status:      "pending",
		Tool:        "describe_image",
	})

	// Tasks 4-7: Generate each panel image with character description prepended
	for i, p := range scenario.Panels[:4] {
		prompt := p.ImagePrompt
		if prompt == "" {
			prompt = fmt.Sprintf("4-panel comic, panel %d, %s, manga style, simple background, bold outlines, %s", p.ID, p.Scene, charAppearance)
		}
		// Prepend character appearance to ensure consistency (will be enhanced at execution time with vision description)
		plan.Tasks = append(plan.Tasks, PlanTask{
			ID:          i + 4,
			Description: fmt.Sprintf("%dコマ目「%s」を画像生成: %s", p.ID, p.Title, p.Scene),
			Status:      "pending",
			Tool:        "generate_image",
			ImagePrompt: prompt,
		})
	}

	return plan, nil
}

// createPlan asks the sub-model to create a plan for a complex task
func createPlan(userMsg string, messages []Message, config *Config) (*Plan, error) {
	var ctx strings.Builder
	start := 0
	if len(messages) > 5 {
		start = len(messages) - 5
	}
	for _, m := range messages[start:] {
		if m.Role == "user" {
			ctx.WriteString(fmt.Sprintf("ユーザー: %s\n", m.Content))
		} else if m.Role == "assistant" && m.Content != "" {
			c := m.Content
			if len(c) > 200 {
				c = c[:200] + "..."
			}
			ctx.WriteString(fmt.Sprintf("アシスタント: %s\n", c))
		}
	}

	// Determine final output tool based on user request
	finalTool := "summarize"
	finalToolDesc := "最終まとめを生成"
	if needsImageGeneration(userMsg) {
		finalTool = "generate_image"
		finalToolDesc = "調査結果をもとにインフォグラフィック画像をAI生成"
	} else if needsRunCode(userMsg) {
		finalTool = "run_code"
		finalToolDesc = "HTML/JS/Canvasでインタラクティブコンテンツを生成"
	} else {
		lower := strings.ToLower(userMsg)
		for _, kw := range []string{"図", "ダイアグラム", "関係図", "構成図", "フロー図"} {
			if strings.Contains(lower, kw) {
				finalTool = "diagram"
				finalToolDesc = "調査結果をもとにGraphviz図を生成"
				break
			}
		}
	}

	prompt := fmt.Sprintf(`ユーザーのリクエストを実行可能なステップに分解せよ。

ツール: web_search, web_fetch, summarize, diagram, generate_image, run_code, execute_command, read_file, write_file

リクエスト: %s

最終出力ツール: %s

以下の形式のJSONのみ出力せよ。他の文章は書くな:
{"tasks":[
{"id":1,"description":"...を検索","tool":"web_search"},
{"id":2,"description":"検索結果の上位ページを取得","tool":"web_fetch"},
{"id":3,"description":"情報を要約・整理","tool":"summarize"},
{"id":4,"description":"%s","tool":"%s"}
]}

上記は例。タスク数は3〜10個で調整せよ。最後のタスクのtoolは必ず「%s」にしろ。`, userMsg, finalTool, finalToolDesc, finalTool, finalTool)

	// Use sub-agent for plan creation if available (better at complex decomposition)
	var response string
	var err error
	if hasSubAgent(config) {
		fmt.Printf("[siki] Using sub-agent (%s) for plan creation\n", config.SubAgent)
		_, response, err = callSubAgent(prompt, config)
	} else {
		response, err = callOllamaGenerate(config.SubModel, prompt, 4096, 600*time.Second, config)
	}
	if err != nil {
		return nil, fmt.Errorf("plan creation failed: %w", err)
	}

	// Parse JSON from response
	response = strings.TrimSpace(response)
	var planData struct {
		Tasks []struct {
			ID          int    `json:"id"`
			Description string `json:"description"`
			Tool        string `json:"tool"`
		} `json:"tasks"`
	}

	// Try to extract JSON
	jsonStr := response
	if idx := strings.Index(jsonStr, "{"); idx >= 0 {
		jsonStr = jsonStr[idx:]
		depth := 0
		for i, ch := range jsonStr {
			if ch == '{' {
				depth++
			}
			if ch == '}' {
				depth--
				if depth == 0 {
					jsonStr = jsonStr[:i+1]
					break
				}
			}
		}
	}

	if err := json.Unmarshal([]byte(jsonStr), &planData); err != nil {
		return nil, fmt.Errorf("plan JSON parse failed: %w", err)
	}

	if len(planData.Tasks) == 0 {
		return nil, fmt.Errorf("plan has no tasks")
	}

	// Post-process: if user wants image generation, FORCE last task to generate_image
	// The sub-model consistently ignores prompt instructions, so we override unconditionally.
	if needsImageGeneration(userMsg) {
		lastIdx := len(planData.Tasks) - 1
		if lastIdx >= 0 {
			oldTool := planData.Tasks[lastIdx].Tool
			if oldTool != "generate_image" {
				fmt.Printf("[siki] Plan post-process: forcing last task (%s → generate_image)\n", oldTool)
				planData.Tasks[lastIdx].Tool = "generate_image"
				planData.Tasks[lastIdx].Description = "調査結果をもとにAI画像（インフォグラフィック）を生成する"
			}
		}
		// Also remove any intermediate write_file/run_code tasks that are HTML-generation artifacts
		filtered := planData.Tasks[:0]
		for _, t := range planData.Tasks {
			if t.Tool == "write_file" || (t.Tool == "run_code" && t.ID != planData.Tasks[lastIdx].ID) {
				fmt.Printf("[siki] Plan post-process: removing HTML artifact task %d (%s)\n", t.ID, t.Tool)
				continue
			}
			filtered = append(filtered, t)
		}
		planData.Tasks = filtered
		// Re-number tasks sequentially
		for i := range planData.Tasks {
			planData.Tasks[i].ID = i + 1
		}
	}

	plan := &Plan{
		Goal:      userMsg,
		CreatedAt: time.Now().Format(time.RFC3339),
		Status:    "planning",
	}
	for _, t := range planData.Tasks {
		plan.Tasks = append(plan.Tasks, PlanTask{
			ID:          t.ID,
			Description: t.Description,
			Status:      "pending",
			Tool:        t.Tool,
		})
	}

	return plan, nil
}

// executePlanTask asks the orchestrator to execute a specific plan task
func executePlanTask(task *PlanTask, plan *Plan, agent *Agent, config *Config, sendEvent func(StreamEvent)) (string, error) {
	// describe_image: directly use vision model without sub-model decision
	if task.Tool == "describe_image" {
		sendEvent(StreamEvent{Type: "thinking", Content: "参照画像を解析中...", Model: config.VisionModel})
		// Find the most recent generate_image result (should be the character reference image)
		var imagePath string
		for _, t := range plan.Tasks {
			if t.Status == "completed" && t.Tool == "generate_image" && t.Result != "" {
				imagePath = t.Result
			}
		}
		if imagePath == "" {
			return "参照画像が見つかりません", nil
		}

		// Resolve path: result is like "/playground/image_xxx.png"
		actualPath := imagePath
		if strings.HasPrefix(imagePath, "/playground/") {
			actualPath = filepath.Join(playgroundDir, strings.TrimPrefix(imagePath, "/playground/"))
		}
		imgData, err := os.ReadFile(actualPath)
		if err != nil {
			fmt.Printf("[siki] describe_image: failed to read image %s: %v\n", actualPath, err)
			return fmt.Sprintf("画像ファイルの読み込みに失敗: %v", err), nil
		}
		b64 := base64.StdEncoding.EncodeToString(imgData)

		if config.VisionModel == "" {
			fmt.Printf("[siki] describe_image: no vision model configured\n")
			return "Vision modelが未設定のため、シナリオのキャラクター描写を使用します。", nil
		}

		charDescPrompt := "Describe this character reference image in detail. Focus on: hair style, hair color, eye color, clothing, accessories, body type, distinctive features. Be very specific and use English. This description will be used to maintain character consistency across multiple comic panels."
		desc := describeImageForCharacter(b64, charDescPrompt, config.VisionModel, config)
		if desc == "" {
			return "キャラクター描写を取得できませんでした", nil
		}

		fmt.Printf("[siki] describe_image: character description extracted (%d bytes)\n", len(desc))
		sendEvent(StreamEvent{Type: "content", Content: fmt.Sprintf("\n**キャラクター外見解析結果:**\n%s\n", desc)})
		return desc, nil
	}

	// Build context from plan progress so far
	var ctx strings.Builder
	ctx.WriteString(fmt.Sprintf("## 全体目標\n%s\n\n## 進捗\n", plan.Goal))
	for _, t := range plan.Tasks {
		status := "⬜"
		switch t.Status {
		case "completed":
			status = "✅"
		case "in_progress":
			status = "🔄"
		case "failed":
			status = "❌"
		}
		ctx.WriteString(fmt.Sprintf("%s %d. %s", status, t.ID, t.Description))
		if t.Result != "" {
			r := t.Result
			if len(r) > 3000 {
				r = r[:3000] + "..."
			}
			ctx.WriteString(fmt.Sprintf("\n結果:\n%s\n", r))
		}
		ctx.WriteString("\n")
	}

	prompt := fmt.Sprintf(`%s

## 現在実行するタスク
タスク%d: %s
推奨ツール: %s

このタスクを実行するためのツール呼び出しをJSON形式で出力せよ:
{"tool":"ツール名","args":{引数}}

注意:
- web_searchの場合、具体的な検索クエリを指定
- run_codeの場合、完全なHTMLを生成
- execute_commandの場合、具体的なコマンドを指定
- generate_imageの場合、英語で詳細な画像プロンプトを {"tool":"generate_image","args":{"prompt":"..."}} で指定
- summarize の場合は {"tool":"none","response":"まとめテキスト"} を返せ`,
		ctx.String(), task.ID, task.Description, task.Tool)

	response, err := callOllamaGenerate(config.SubModel, prompt, 4096, 600*time.Second, config)
	if err != nil {
		return "", err
	}

	// Parse the tool decision
	response = strings.TrimSpace(response)
	var decision OrchestratorDecision
	jsonStr := response
	if idx := strings.Index(jsonStr, "{"); idx >= 0 {
		jsonStr = jsonStr[idx:]
		depth := 0
		for i, ch := range jsonStr {
			if ch == '{' {
				depth++
			}
			if ch == '}' {
				depth--
				if depth == 0 {
					jsonStr = jsonStr[:i+1]
					break
				}
			}
		}
	}
	if err := json.Unmarshal([]byte(jsonStr), &decision); err != nil {
		// Fallback: use the task's recommended tool
		decision = OrchestratorDecision{Tool: task.Tool, Args: map[string]interface{}{}}
	}

	// Execute the tool — normalize common typos from sub-model
	toolName := decision.Tool
	toolAliases := map[string]string{
		"diagrams": "diagram", "search": "web_search", "fetch": "web_fetch",
		"image": "generate_image", "code": "run_code", "command": "execute_command",
	}
	if alias, ok := toolAliases[toolName]; ok {
		fmt.Printf("[siki] Plan task: normalizing tool '%s' → '%s'\n", toolName, alias)
		toolName = alias
	}
	if toolName == "" || toolName == "none" || toolName == "summarize" {
		// No tool needed - use sub-model for summarization
		if decision.Response != "" {
			return decision.Response, nil
		}
		// Collect all previous task results as context for summarization
		var prevResults strings.Builder
		for _, t := range plan.Tasks {
			if t.Status == "completed" && t.Result != "" {
				prevResults.WriteString(fmt.Sprintf("## タスク%d: %s\n%s\n\n", t.ID, t.Description, t.Result))
			}
		}
		resp, err := streamSubModelSummarize(
			fmt.Sprintf("%s\n\nタスク: %s", plan.Goal, task.Description),
			"previous_results", prevResults.String(), config, sendEvent)
		if err != nil {
			return "", err
		}
		return resp, nil
	}

	args := decision.Args
	if args == nil {
		args = map[string]interface{}{}
	}

	// Set defaults
	if toolName == "web_search" {
		if _, ok := args["query"]; !ok {
			args["query"] = task.Description
		}
	}

	// For web_fetch: auto-extract URLs from previous web_search results
	if toolName == "web_fetch" {
		argURL, _ := args["url"].(string)
		// If no URL or dummy URL (example.com, etc.), extract from previous search results
		if argURL == "" || strings.Contains(argURL, "example.com") || strings.Contains(argURL, "example.org") || !strings.HasPrefix(argURL, "http") {
			// Find previous web_search result
			var searchResult string
			for _, t := range plan.Tasks {
				if t.Status == "completed" && t.Tool == "web_search" && t.Result != "" {
					searchResult = t.Result
					break
				}
			}
			if searchResult != "" {
				// Extract all URLs from search results using markdown link format [text](url)
				var urls []string
				remaining := searchResult
				for {
					idx := strings.Index(remaining, "](http")
					if idx == -1 {
						break
					}
					urlStart := idx + 2
					urlEnd := strings.Index(remaining[urlStart:], ")")
					if urlEnd == -1 {
						break
					}
					u := remaining[urlStart : urlStart+urlEnd]
					// Skip DuckDuckGo internal URLs
					if !strings.Contains(u, "duckduckgo.com") {
						urls = append(urls, u)
					}
					remaining = remaining[urlStart+urlEnd:]
				}

				if len(urls) > 0 {
					// Determine which URL to use based on task description
					urlIndex := 0
					desc := task.Description
					if strings.Contains(desc, "2番目") || strings.Contains(desc, "2つ目") || strings.Contains(desc, "second") {
						urlIndex = 1
					} else if strings.Contains(desc, "3番目") || strings.Contains(desc, "3つ目") || strings.Contains(desc, "third") {
						urlIndex = 2
					} else if strings.Contains(desc, "4番目") || strings.Contains(desc, "4つ目") {
						urlIndex = 3
					} else if strings.Contains(desc, "5番目") || strings.Contains(desc, "5つ目") {
						urlIndex = 4
					}
					// Also detect generic Nth pattern
					for n := 1; n <= 10; n++ {
						if strings.Contains(desc, fmt.Sprintf("%d番目", n)) {
							urlIndex = n - 1
							break
						}
					}
					if urlIndex >= len(urls) {
						urlIndex = len(urls) - 1
					}
					args["url"] = urls[urlIndex]
					fmt.Printf("[siki] Plan web_fetch: auto-extracted URL[%d]: %s\n", urlIndex, urls[urlIndex])
				}
			}
		}
	}

	// For diagram: generate DOT from previous task results if not provided
	if toolName == "diagram" {
		dotCode, _ := args["dot_source"].(string)
		if dotCode == "" {
			dotCode, _ = args["dot_code"].(string)
		}
		if dotCode == "" || !strings.Contains(dotCode, "digraph") {
			// Collect previous task results as context for diagram generation
			var prevData strings.Builder
			for _, t := range plan.Tasks {
				if t.Status == "completed" && t.Result != "" {
					prevData.WriteString(fmt.Sprintf("## タスク%d: %s\n%s\n\n", t.ID, t.Description, t.Result))
				}
			}
			// Ask sub-model to generate DOT code from accumulated data
			dotPrompt := fmt.Sprintf(`以下の情報を元に、Graphviz DOTコードを生成せよ。
目標: %s

## これまでの調査結果:
%s

## ルール:
- digraph G { ... } 形式のDOTコードのみを出力せよ
- 日本語のラベルを使え
- node [shape=box, style=filled] を使え
- 重要度に応じてfillcolorを変えろ
- エッジにlabelで関係性を書け
- コード以外の説明文は一切書くな
- DOTコードだけを出力せよ`, plan.Goal, prevData.String())

			dotResponse, err := callOllamaGenerate(config.SubModel, dotPrompt, 4096, 120*time.Second, config)
			if err == nil {
				// Extract DOT code from response
				dotResponse = strings.TrimSpace(dotResponse)
				if idx := strings.Index(dotResponse, "digraph"); idx >= 0 {
					dotCode = dotResponse[idx:]
					// Find matching closing brace
					depth := 0
					for i, ch := range dotCode {
						if ch == '{' {
							depth++
						} else if ch == '}' {
							depth--
							if depth == 0 {
								dotCode = dotCode[:i+1]
								break
							}
						}
					}
				}
			}
			if dotCode != "" && strings.Contains(dotCode, "digraph") {
				args["dot_source"] = dotCode
				fmt.Printf("[siki] Plan diagram: generated DOT code (%d bytes)\n", len(dotCode))
			} else {
				// Fallback default diagram
				args["dot_source"] = fmt.Sprintf(`digraph G {
  rankdir=LR;
  node [shape=box, style=filled, fillcolor="#e8e8e8"];
  Goal [label="%s", fillcolor="#e94560", fontcolor=white];
  Data [label="調査データ", fillcolor="#16213e", fontcolor=white];
  Result [label="結果"];
  Goal -> Data [label="調査"];
  Data -> Result [label="分析"];
}`, strings.ReplaceAll(plan.Goal, `"`, `'`))
				fmt.Printf("[siki] Plan diagram: using fallback DOT\n")
			}
		}
	}

	// For run_code, ensure quality HTML
	if toolName == "run_code" {
		html, _ := args["html"].(string)
		if len(html) < 100 {
			if newHTML, err := generateCodeWithSubModel(task.Description, config); err == nil {
				args["html"] = newHTML
			}
		}
	}

	// For generate_image: use pre-generated prompt (e.g., comic panels) or build from previous results
	if toolName == "generate_image" {
		prompt, _ := args["prompt"].(string)
		// Use pre-generated ImagePrompt if available (from comic plan etc.)
		if (prompt == "" || len(prompt) < 20) && task.ImagePrompt != "" {
			args["prompt"] = task.ImagePrompt
			prompt = task.ImagePrompt
			sendEvent(modelThinkingEvent(fmt.Sprintf("Image prompt: %s", prompt), config, false))
		}
		// For comic panels: prepend character description from describe_image task for consistency
		// Only for panel tasks (not the character reference image itself)
		if task.ID >= 4 { // Tasks 4-7 are comic panels in the new pipeline
			for _, t := range plan.Tasks {
				if t.Tool == "describe_image" && t.Status == "completed" && t.Result != "" {
					// Prepend vision-extracted character description to the prompt
					charPrefix := fmt.Sprintf("Characters: %s. ", t.Result)
					if len(charPrefix) > 500 {
						charPrefix = charPrefix[:500] + "... "
					}
					prompt = charPrefix + prompt
					args["prompt"] = prompt
					fmt.Printf("[siki] Comic panel %d: prepended character description (%d bytes)\n", task.ID, len(charPrefix))
					break
				}
			}
		}
		if prompt == "" || len(prompt) < 20 {
			// Collect all previous task results as context
			var prevData strings.Builder
			for _, t := range plan.Tasks {
				if t.Status == "completed" && t.Result != "" {
					r := t.Result
					if len(r) > 2000 {
						r = r[:2000] + "..."
					}
					prevData.WriteString(fmt.Sprintf("## %s\n%s\n\n", t.Description, r))
				}
			}
			imgPromptReq := fmt.Sprintf(`以下の調査結果をもとに、インフォグラフィック画像を生成するための英語プロンプトを作成せよ。
プロンプトのみ出力し、他の文章は書くな。
スタイル: modern infographic, clean design, data visualization, professional, dark background

目標: %s

調査結果:
%s`, plan.Goal, prevData.String())
			_, imgPrompt, err := callSubAgent(imgPromptReq, config)
			if err == nil && len(imgPrompt) > 10 {
				imgPrompt = strings.TrimSpace(imgPrompt)
				imgPrompt = strings.TrimPrefix(imgPrompt, "```")
				imgPrompt = strings.TrimSuffix(imgPrompt, "```")
				imgPrompt = strings.TrimSpace(imgPrompt)
				if len(imgPrompt) > 2 && imgPrompt[0] == '"' && imgPrompt[len(imgPrompt)-1] == '"' {
					imgPrompt = imgPrompt[1 : len(imgPrompt)-1]
				}
				args["prompt"] = imgPrompt
				sendEvent(modelThinkingEvent(fmt.Sprintf("Image prompt: %s", imgPrompt), config, hasSubAgent(config)))
			} else {
				// Fallback: use task description directly
				args["prompt"] = "Modern infographic about " + plan.Goal + ", data visualization, professional design, dark theme"
			}
		}
	}

	sendEvent(StreamEvent{Type: "tool_start", Name: toolName})
	result, err := agent.executeTool(toolName, args)
	if err != nil {
		result = fmt.Sprintf("Error: %v", err)
	}

	displayResult := result
	if len(displayResult) > 2000 {
		displayResult = displayResult[:2000] + "\n... (truncated)"
	}
	sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: displayResult})

	return result, nil
}

// verifyGoalFulfillment checks if a plan actually fulfilled the user's request.
// Returns true if the essential tasks completed successfully.
func verifyGoalFulfillment(plan *Plan, userMsg string) bool {
	if plan == nil {
		return false
	}

	totalTasks := len(plan.Tasks)
	if totalTasks == 0 {
		return false
	}

	completedCount := 0
	failedCount := 0
	imageTaskCount := 0
	imageCompletedCount := 0

	for _, t := range plan.Tasks {
		switch t.Status {
		case "completed":
			completedCount++
		case "failed":
			failedCount++
		}
		if t.Tool == "generate_image" {
			imageTaskCount++
			if t.Status == "completed" {
				imageCompletedCount++
			}
		}
	}

	// For comic requests: at least some images must have been generated
	if isComicRequest(userMsg) {
		if imageTaskCount > 0 && imageCompletedCount == 0 {
			fmt.Printf("[siki] Goal check: comic request but 0/%d images generated\n", imageTaskCount)
			return false
		}
	}

	// General: if more than half the tasks failed, goal is not met
	if failedCount > totalTasks/2 {
		fmt.Printf("[siki] Goal check: %d/%d tasks failed\n", failedCount, totalTasks)
		return false
	}

	return true
}

// (ws *WebServer) executePlan runs all tasks in a plan sequentially
func (ws *WebServer) executePlan(ctx context.Context, plan *Plan, planID string, agent *Agent, sendEvent func(StreamEvent), saveMsg func(Message, string)) string {
	plan.Status = "executing"
	savePlan(plan, planID)

	var allResults strings.Builder
	var diagramResults []string // Keep diagram HTML outputs separate

	for i := range plan.Tasks {
		select {
		case <-ctx.Done():
			plan.Status = "failed"
			savePlan(plan, planID)
			return allResults.String()
		default:
		}

		task := &plan.Tasks[i]
		task.Status = "in_progress"
		savePlan(plan, planID)

		// Send plan progress event
		sendEvent(StreamEvent{
			Type:    "plan_progress",
			Content: fmt.Sprintf("タスク %d/%d 実行中: %s", task.ID, len(plan.Tasks), task.Description),
		})

		fmt.Printf("[siki] Plan: executing task %d/%d: %s (tool=%s)\n", task.ID, len(plan.Tasks), task.Description, task.Tool)

		result, err := executePlanTask(task, plan, agent, ws.config, sendEvent)
		if err != nil {
			task.Status = "failed"
			task.Result = fmt.Sprintf("Error: %v", err)
			fmt.Printf("[siki] Plan: task %d failed: %v\n", task.ID, err)
		} else {
			task.Status = "completed"
			r := result
			if len(r) > 5000 {
				r = r[:5000] + "..."
			}
			task.Result = r
			fmt.Printf("[siki] Plan: task %d completed\n", task.ID)
		}

		// If this is a diagram result (contains iframe), save separately
		if task.Tool == "diagram" && strings.Contains(result, "<iframe") {
			diagramResults = append(diagramResults, result)
			allResults.WriteString(fmt.Sprintf("\n## タスク%d: %s\n[図が生成されました]\n", task.ID, task.Description))
		} else {
			allResults.WriteString(fmt.Sprintf("\n## タスク%d: %s\n%s\n", task.ID, task.Description, result))
		}
		savePlan(plan, planID)

		// Send updated progress
		if task.Status == "failed" {
			sendEvent(StreamEvent{
				Type:    "plan_progress",
				Content: fmt.Sprintf("タスク %d/%d 失敗 ❌: %s", task.ID, len(plan.Tasks), task.Result),
			})
		} else {
			sendEvent(StreamEvent{
				Type:    "plan_progress",
				Content: fmt.Sprintf("タスク %d/%d 完了 ✅", task.ID, len(plan.Tasks)),
			})
		}
	}

	plan.Status = "completed"
	savePlan(plan, planID)

	// Generate final summary
	sendEvent(modelThinkingEvent("全タスク完了。最終まとめを生成中...", ws.config, hasSubAgent(ws.config)))
	summary, err := streamSubModelSummarize(plan.Goal, "plan", allResults.String(), ws.config, sendEvent)
	if err != nil || summary == "" {
		summary = allResults.String()
		sendEvent(StreamEvent{Type: "content", Content: summary})
	}

	// Append diagram results (iframe HTML) after the summary
	if len(diagramResults) > 0 {
		for _, diag := range diagramResults {
			summary += "\n\n" + diag
			sendEvent(StreamEvent{Type: "content", Content: "\n\n" + diag})
		}
	}

	// Auto-generate infographic if image server is available
	if imageServerReady && ws.config.ImageEnabled {
		sendEvent(modelThinkingEvent("インフォグラフィックを生成中...", ws.config, hasSubAgent(ws.config)))
		// Generate English image prompt from summary using sub-model
		if ws.config.SubModel != "" {
			imgPromptReq := fmt.Sprintf(`以下のまとめ内容を表現するインフォグラフィック画像の英語プロンプトを生成せよ。
プロンプトのみ出力し、他の文章は書くな。
スタイル: modern infographic, clean design, data visualization, professional

まとめ内容: %s

目標: %s`, summary[:min(len(summary), 2000)], plan.Goal)
			_, imgPrompt, err := callSubAgent(imgPromptReq, ws.config)
			if err == nil && len(imgPrompt) > 10 {
				imgPrompt = strings.TrimSpace(imgPrompt)
				urlPath, err := generateImage(imgPrompt, 768, 768, ws.config)
				if err == nil {
					imgMarkdown := fmt.Sprintf("\n\n![Infographic](%s)", urlPath)
					summary += imgMarkdown
					sendEvent(StreamEvent{Type: "content", Content: imgMarkdown})
				} else {
					fmt.Printf("[siki] Auto infographic generation failed: %v\n", err)
				}
			}
		}
	}

	return summary
}

// quickAck generates a fast 1-sentence acknowledgment using lfm (primary model).
// Gives the user immediate feedback while gpt-oss processes the real request.
// quickAck generates an instant acknowledgment based on keyword matching.
// No model call — deterministic, always Japanese, zero latency.
func quickAck(userMsg string, config *Config) string {
	lower := strings.ToLower(userMsg)

	// Comic requests get a special ack (takes priority over "描いて"/"書いて" match)
	if isComicRequest(userMsg) {
		return "4コマ漫画を作成しますね！シナリオとキャラクターをデザイン中..."
	}

	type ackRule struct {
		keywords []string
		response string
	}
	rules := []ackRule{
		{[]string{"画像生成", "イラスト描", "インフォグラフィック", "画像を生成", "画像を作"}, "画像を生成しますね！Flux Klein 4Bを準備中..."},
		{[]string{"ニュース", "news", "最新", "速報", "トレンド"}, "最新の情報を検索しますね！"},
		{[]string{"描いて", "書いて", "フラクタル", "マンデルブロ", "ゲーム", "アニメーション", "可視化", "シミュレーション"}, "描画しますね！サブモデルでコード生成中..."},
		{[]string{"http://", "https://", "url", "サイト"}, "URLの内容を取得しますね！"},
		{[]string{"調べて", "検索", "教えて", "について"}, "調べますね！"},
		{[]string{"ファイル", "読んで", "開いて"}, "ファイルを確認しますね！"},
		{[]string{"コマンド", "実行", "install", "apt", "pip"}, "コマンドを実行しますね！"},
		{[]string{"図", "ダイアグラム", "アーキテクチャ", "関係図"}, "図を生成しますね！"},
		{[]string{"こんにちは", "おはよう", "こんばんは", "やあ", "hello", "hi"}, "こんにちは！何かお手伝いしましょうか？"},
		{[]string{"ありがとう", "thanks"}, "どういたしまして！"},
	}

	for _, rule := range rules {
		for _, kw := range rule.keywords {
			if strings.Contains(lower, kw) {
				return rule.response
			}
		}
	}

	return "考え中..."
}

// subModelOrchestrate asks gpt-oss to analyze the user's request and decide which tool to call.
func subModelOrchestrate(userMsg string, messages []Message, config *Config) (*OrchestratorDecision, error) {
	// Build conversation context (last 10 messages)
	var ctx strings.Builder
	start := 0
	if len(messages) > 10 {
		start = len(messages) - 10
	}
	for _, m := range messages[start:] {
		switch m.Role {
		case "user":
			ctx.WriteString(fmt.Sprintf("ユーザー: %s\n", m.Content))
		case "assistant":
			if m.Content != "" {
				c := m.Content
				if len(c) > 200 {
					c = c[:200] + "..."
				}
				ctx.WriteString(fmt.Sprintf("アシスタント: %s\n", c))
			}
		case "tool":
			c := m.Content
			if len(c) > 100 {
				c = c[:100] + "..."
			}
			ctx.WriteString(fmt.Sprintf("[ツール結果: %s]\n", c))
		}
	}

	now := time.Now()
	prompt := fmt.Sprintf(`あなたはAIオーケストレーター。ユーザーのリクエストを分析し、適切なアクションを決定せよ。

## 利用可能なツール
- web_search: インターネット検索。引数: {"query": "検索クエリ"}
- web_fetch: URL内容取得。引数: {"url": "URL"}
- run_code: HTML/JS/Canvas実行（描画・可視化・ゲーム・アニメーション）。引数: {"html": "完全なHTML"}
- diagram: Graphviz図生成（構成図・関係図）。引数: {"dot_source": "DOTコード"}
- execute_command: シェルコマンド。引数: {"command": "コマンド"}
- read_file: ファイル読込。引数: {"path": "ファイルパス"}
- write_file: ファイル書込。引数: {"path": "パス", "content": "内容"}
- generate_image: AI画像生成（Flux Klein 4B）。インフォグラフィック・イラスト・コンセプトアート。引数: {"prompt": "英語の詳細プロンプト"}
- generate_video: AI動画生成（Helios）。アニメーション・モーション映像。引数: {"prompt": "英語の詳細プロンプト", "num_frames": 33}
- search_threads: 過去の全スレッド横断で会話ログを検索。引数: {"query": "検索キーワード"}
- search_conversation: 現在のスレッド内の会話を検索。引数: {"query": "検索キーワード"}
- recall_context: 現スレッドの会話ログから文脈を思い出す。引数: {"query": "検索キーワード"}
- plan: 複雑なタスクを複数ステップに分解して順次実行。引数: {"goal": "ユーザーのリクエストそのまま"}
- self_status: 自分の現在の状態・バージョン・ルール・ベンチマークスコアを確認。引数: {}
- self_modify_prompt: 自分のシステムプロンプトを変更。引数: {"action": "replace|append|replace_section", "content": "新しい内容", "reason": "理由"}
- self_modify_params: 自分のパラメータ変更。引数: {"params": "{\"temperature\":0.5}", "reason": "理由"}
- self_add_rule: 自分にルールを追加。引数: {"rule": "ルール内容", "reason": "理由"}
- self_remove_rule: ルールを無効化。引数: {"rule_id": "ルールID"}
- self_rollback: 以前のバージョンに戻す。引数: {"version": バージョン番号}
- self_benchmark: 自己評価ベンチマーク。引数: {"categories": "all"}
- self_evolve: ソースコード自己進化。引数: {"action": "view_source|patch|build_test|deploy|status|abort"}

## 会話履歴
%s
## 現在のリクエスト
%s

## 注意
- 今日は%d年%d月%d日
- run_codeはゲーム・シミュレーション・インタラクティブUI等のプログラム実行専用
- generate_imageはインフォグラフィック・イラスト・ポスター等の画像生成専用（run_codeと混同するな）
- generate_videoは動画・映像・アニメーション生成専用。「動画」「ビデオ」「映像」キーワードにはgenerate_videoを使え
- web_searchの場合、適切な検索クエリを指定
- 「過去の会話」「前に話した」「さっきの」「会話ログ」等の場合はsearch_threadsかrecall_contextを使え（web_searchではない）
- 複数のツールを組み合わせる必要がある複雑なリクエスト（調査→分析→可視化など）には plan を使え
- planのgoalにはユーザーのリクエストをそのまま入れろ。実装方法（HTML等）を勝手に追加するな
- 単純な1ステップの作業にはplanを使うな
- ツール不要（挨拶・計算・知識質問）なら直接回答
- 「システムプロンプト変更」「自分を変えろ」「ルール追加」等の自己改変リクエストにはself_modify_prompt/self_add_rule等を使え
- 「自分の状態」「ステータス」にはself_statusを使え

以下のJSON形式のみ出力せよ（他の文章は絶対に書くな）:
ツール使用: {"tool":"ツール名","args":{引数}}
直接回答: {"tool":"none","response":"回答テキスト"}`,
		ctx.String(), userMsg, now.Year(), int(now.Month()), now.Day())

	fmt.Printf("[siki] Orchestrator model: %s (backend: %s)\n", config.orchestratorModel(), config.orchestratorBackend())
	response, err := callOrchestratorGenerate(prompt, 4096, 600*time.Second, config)
	if err != nil {
		return nil, fmt.Errorf("orchestration failed: %w", err)
	}
	response = strings.TrimSpace(response)

	// Parse JSON from response
	var decision OrchestratorDecision
	if err := json.Unmarshal([]byte(response), &decision); err != nil {
		// Try to extract JSON object from response text
		jsonStr := response
		if idx := strings.Index(jsonStr, "{"); idx >= 0 {
			jsonStr = jsonStr[idx:]
			depth := 0
			for i, ch := range jsonStr {
				if ch == '{' {
					depth++
				}
				if ch == '}' {
					depth--
					if depth == 0 {
						jsonStr = jsonStr[:i+1]
						break
					}
				}
			}
		}
		if err2 := json.Unmarshal([]byte(jsonStr), &decision); err2 != nil {
			// JSON parse failed — do NOT use raw text (may contain hallucinations).
			// Return empty decision; dualModelPipeline will fall back to keyword detection.
			fmt.Printf("[siki] Could not parse orchestrator JSON, falling back to keyword detection\n")
			decision = OrchestratorDecision{
				Tool: "",
			}
		}
	}

	fmt.Printf("[siki] Orchestrator decision: tool=%s\n", decision.Tool)
	return &decision, nil
}

// isFollowUpQuery detects if the user message is a follow-up to a previous result.
func isFollowUpQuery(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	followUpKeywords := []string{
		"深掘り", "詳しく", "もっと", "それぞれ", "各", "具体的",
		"詳細", "掘り下げ", "続き", "さらに", "展開",
	}
	for _, kw := range followUpKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// extractURLsFromConversation extracts URLs from recent tool results in conversation.
func extractURLsFromConversation(messages []Message) []string {
	var urls []string
	// Look at recent messages (last 10) for tool results containing URLs
	start := 0
	if len(messages) > 10 {
		start = len(messages) - 10
	}
	for _, m := range messages[start:] {
		if m.Role == "tool" || (m.Role == "assistant" && len(m.ToolCalls) == 0) {
			for _, line := range strings.Split(m.Content, "\n") {
				line = strings.TrimSpace(line)
				// Markdown link
				if idx := strings.Index(line, "]("); idx >= 0 {
					u := line[idx+2:]
					if end := strings.Index(u, ")"); end >= 0 {
						u = u[:end]
						if strings.HasPrefix(u, "http") {
							urls = append(urls, u)
						}
					}
				} else if strings.HasPrefix(line, "http://") || strings.HasPrefix(line, "https://") {
					urls = append(urls, line)
				}
			}
		}
	}
	return urls
}

// needsToolButDidnt checks if a direct response looks hallucinated when the user
// clearly needed tool-based information (news, search, real-time data).
func needsToolButDidnt(userMsg, response string) bool {
	lower := strings.ToLower(userMsg)
	// Topics that require real-time data
	realTimeKeywords := []string{
		"ニュース", "最新", "速報", "news", "latest", "トレンド",
		"今日", "今週", "今月", "現在",
	}
	needsRealTime := false
	for _, kw := range realTimeKeywords {
		if strings.Contains(lower, kw) {
			needsRealTime = true
			break
		}
	}
	if needsRealTime {
		// If response doesn't contain any real URLs, it's likely hallucinated
		hasURL := strings.Contains(response, "http://") || strings.Contains(response, "https://")
		if !hasURL && len(response) > 100 {
			return true
		}
	}

	// 「できない」「情報がない」系の応答を検出
	refusalPatterns := []string{
		"できません", "わかりません", "情報がありません", "提供することができません",
		"ツール結果が", "利用できるツール", "申し訳ありません",
		"お手数ですが", "再度送付", "情報を提供することが",
	}
	responseLower := strings.ToLower(response)
	for _, pat := range refusalPatterns {
		if strings.Contains(responseLower, pat) {
			return true // 拒否応答 → ツールでリトライすべき
		}
	}

	// ユーザーが「予測」「分析」「調べ」等を求めているのに応答が短すぎる
	analysisKeywords := []string{"予測", "予想", "分析", "調べ", "検索", "過去の", "履歴"}
	needsAnalysis := false
	for _, kw := range analysisKeywords {
		if strings.Contains(lower, kw) {
			needsAnalysis = true
			break
		}
	}
	if needsAnalysis && len(response) < 200 {
		return true
	}

	return false
}

// containsConversationKeywords checks if the message references past conversations or predictions.
func containsConversationKeywords(msg string) bool {
	keywords := []string{"会話", "過去", "ログ", "履歴", "前に", "さっき", "やり取り", "予測", "予想", "次に何"}
	for _, kw := range keywords {
		if strings.Contains(msg, kw) {
			return true
		}
	}
	return false
}

// executeToolAndSummarize is a helper for retry: execute a tool and stream the summary.
func (ws *WebServer) executeToolAndSummarize(agent *Agent, userMsg, toolName string, sendEvent func(StreamEvent), saveMsg func(Message, string), modelOverride ...string) string {
	retryModel := ""
	if len(modelOverride) > 0 {
		retryModel = modelOverride[0]
	}
	args := map[string]interface{}{}
	if toolName == "web_search" {
		args["query"] = userMsg
	} else if toolName == "twitter_search" {
		args["query"] = extractSearchQuery(userMsg, "twitter", nil)
	}

	// Generate DOT code for diagram
	if toolName == "diagram" {
		sendEvent(modelThinkingEvent("図のDOTコードを生成中...", ws.config, false))
		prompt := fmt.Sprintf("以下のリクエストに対して、Graphviz DOTコードのみ出力せよ（説明不要、コードフェンスも不要）。\nリクエスト: %s", userMsg)
		_, genDot, err := callSubModel(prompt, ws.config)
		if err == nil && len(genDot) > 10 {
			dot := genDot
			if idx := strings.Index(dot, "```dot"); idx >= 0 {
				dot = dot[idx+6:]
				if end := strings.Index(dot, "```"); end >= 0 { dot = dot[:end] }
			} else if idx := strings.Index(dot, "```graphviz"); idx >= 0 {
				dot = dot[idx+11:]
				if end := strings.Index(dot, "```"); end >= 0 { dot = dot[:end] }
			} else if idx := strings.Index(dot, "```"); idx >= 0 {
				dot = dot[idx+3:]
				if nl := strings.Index(dot, "\n"); nl >= 0 { dot = dot[nl+1:] }
				if end := strings.Index(dot, "```"); end >= 0 { dot = dot[:end] }
			}
			dot = strings.TrimSpace(dot)
			if strings.Contains(dot, "digraph") || strings.Contains(dot, "graph") {
				args["dot_source"] = dot
			}
		}
	}

	sendEvent(StreamEvent{Type: "tool_start", Name: toolName})
	result, err := agent.executeTool(toolName, args)
	if err != nil {
		fmt.Printf("[siki] Retry tool execution failed: %v\n", err)
		return ""
	}

	displayResult := result
	if len(displayResult) > 2000 {
		displayResult = displayResult[:2000] + "\n... (truncated)"
	}
	sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: displayResult})

	// Save tool call
	toolCallID := fmt.Sprintf("retry-%d", time.Now().UnixMilli())
	argsJSON, _ := json.Marshal(args)
	assistantMsg := Message{
		Role: "assistant",
		ToolCalls: []ToolCall{{
			ID:   toolCallID,
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: toolName, Arguments: string(argsJSON)},
		}},
	}
	agent.messages = append(agent.messages, assistantMsg)
	saveMsg(assistantMsg, "")

	toolMsg := Message{Role: "tool", Content: result, ToolCallID: toolCallID}
	agent.messages = append(agent.messages, toolMsg)
	saveMsg(toolMsg, toolName)

	// For web_search, also fetch top pages
	if toolName == "web_search" {
		urls := extractURLsFromSearchResults(result)
		if len(urls) > 5 {
			urls = urls[:5]
		}
		if len(urls) > 0 {
			sendEvent(StreamEvent{Type: "thinking", Content: "ページを取得中..."})
			type fetchResult struct {
				url     string
				content string
			}
			ch := make(chan fetchResult, len(urls))
			for _, u := range urls {
				go func(fetchURL string) {
					pageContent, fetchErr := agent.webFetchQuick(fetchURL)
					if fetchErr != nil {
						ch <- fetchResult{fetchURL, ""}
						return
					}
					if len(pageContent) > 3000 {
						pageContent = pageContent[:3000]
					}
					ch <- fetchResult{fetchURL, pageContent}
				}(u)
			}
			var fetchedContent strings.Builder
			fetchedContent.WriteString(result)
			fetchedContent.WriteString("\n\n--- ページ本文 ---\n")
			ticker := time.NewTicker(2 * time.Second)
			collected := 0
			for collected < len(urls) {
				select {
				case fr := <-ch:
					collected++
					sendEvent(StreamEvent{Type: "thinking", Content: fmt.Sprintf("取得完了 (%d/%d)", collected, len(urls))})
					if fr.content != "" {
						fetchedContent.WriteString(fmt.Sprintf("\n## %s\n%s\n", fr.url, fr.content))
					}
				case <-ticker.C:
					sendEvent(StreamEvent{Type: "thinking", Content: "取得中..."})
				}
			}
			ticker.Stop()
			result = fetchedContent.String()
		}
	}

	if retryModel != "" {
		sendEvent(modelThinkingEvent(fmt.Sprintf("再回答を生成中（%s）...", retryModel), ws.config, hasSubAgent(ws.config)))
	} else {
		sendEvent(modelThinkingEvent("再回答を生成中...", ws.config, hasSubAgent(ws.config)))
	}
	finalResponse, err := streamSubModelSummarizeWith(userMsg, toolName, result, ws.config, sendEvent, retryModel)
	if err != nil || finalResponse == "" {
		return ""
	}

	finalMsg := Message{Role: "assistant", Content: finalResponse}
	agent.messages = append(agent.messages, finalMsg)
	saveMsg(finalMsg, "")
	return finalResponse
}

// extractURLsFromSearchResults extracts URLs from web_search result text.
// Handles both plain URLs and markdown links [text](url).
func extractURLsFromSearchResults(searchResult string) []string {
	var urls []string
	for _, line := range strings.Split(searchResult, "\n") {
		line = strings.TrimSpace(line)
		// Markdown link: [text](url)
		if idx := strings.Index(line, "]("); idx >= 0 {
			u := line[idx+2:]
			if end := strings.Index(u, ")"); end >= 0 {
				u = u[:end]
				if strings.HasPrefix(u, "http") {
					urls = append(urls, u)
				}
			}
		} else if strings.HasPrefix(line, "http://") || strings.HasPrefix(line, "https://") {
			urls = append(urls, line)
		}
	}
	return urls
}

// detectToolFromKeywords determines which tool to use based on keyword matching.
// Instant, deterministic fallback when gpt-oss orchestration fails.
// isHFGitHubRunRequest returns true if the message contains a HuggingFace/GitHub URL
// along with an action word indicating the user wants to run/use the model.
func isHFGitHubRunRequest(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	if !strings.Contains(lower, "huggingface.co/") && !strings.Contains(lower, "github.com/") {
		return false
	}
	actionWords := []string{"使って", "使い", "動かし", "実行", "生成", "試し", "走らせ", "run", "use", "generate", "やって", "して"}
	for _, aw := range actionWords {
		if strings.Contains(lower, aw) {
			return true
		}
	}
	return false
}

// detectTwitterTool returns "twitter_timeline" for timeline requests,
// "twitter_search" for search requests, or "" if no Twitter keyword found.
func detectTwitterTool(userMsg string) string {
	lower := strings.ToLower(userMsg)
	// Check for timeline-specific keywords first
	timelineKws := []string{"タイムライン", "timeline", "フィード", "feed", "自分のtwitter", "自分のツイッター", "フォロー"}
	for _, kw := range timelineKws {
		if strings.Contains(lower, kw) {
			return "twitter_timeline"
		}
	}
	// General Twitter keywords → search
	searchKws := []string{"twitter", "ツイッター", "ツイート", "tweet", "x.com"}
	for _, kw := range searchKws {
		if strings.Contains(lower, kw) {
			return "twitter_search"
		}
	}
	return ""
}

func containsTwitterKeywords(userMsg string) bool {
	return detectTwitterTool(userMsg) != ""
}

func detectToolFromKeywords(userMsg string) string {
	lower := strings.ToLower(userMsg)

	// HuggingFace/GitHub URL + action word → docker_run_model (before generic URL rule)
	if strings.Contains(lower, "huggingface.co/") || strings.Contains(lower, "github.com/") {
		actionWords := []string{"使って", "使い", "動かし", "実行", "生成", "試し", "走らせ", "run", "use", "generate", "やって", "して"}
		for _, aw := range actionWords {
			if strings.Contains(lower, aw) {
				fmt.Printf("[siki] Keyword fallback: HF/GitHub URL + '%s' → docker_run_model\n", aw)
				return "docker_run_model"
			}
		}
	}

	type toolRule struct {
		keywords []string
		tool     string
	}
	rules := []toolRule{
		{[]string{"4コマ", "４コマ", "漫画", "マンガ", "コミック", "comic"}, "plan"},
		{[]string{"動画生成", "動画を生成", "動画を作", "ビデオ生成", "ビデオを作", "映像生成", "映像を作", "video generat", "generate video", "アニメーション生成"}, "generate_video"},
		{[]string{"画像生成", "イラスト描", "インフォグラフィック", "image generat", "generate image", "画像を生成", "画像を作"}, "generate_image"},
		{[]string{"過去の会話", "会話ログ", "過去ログ", "前の会話", "さっきの会話", "会話履歴", "スレッド検索", "やり取り"}, "search_threads"},
		{[]string{"bluesky", "bsky", "ブルースカイ"}, "bluesky_feed"},
		{[]string{"bluesky検索", "bsky検索", "ブルースカイ検索", "bluesky search"}, "bluesky_search"},
		{[]string{"タイムライン", "timeline", "フィード", "feed"}, "twitter_timeline"},
		{[]string{"twitter", "ツイッター", "ツイート", "tweet", "x.com"}, "twitter_search"},
		{[]string{"ニュース", "news", "最新", "速報", "トレンド"}, "web_search"},
		{[]string{"調べて", "検索", "教えて", "について", "とは"}, "web_search"},
		{[]string{"http://", "https://"}, "web_fetch"},
		{[]string{"描いて", "書いて", "可視化", "グラフ描", "ゲーム作", "フラクタル", "マンデルブロ", "シミュレーション", "アニメーション"}, "run_code"},
		{[]string{"図", "ダイアグラム", "アーキテクチャ", "関係図", "構成図"}, "diagram"},
		{[]string{"ファイル", "読んで", "開いて"}, "read_file"},
		{[]string{"コマンド", "実行して", "install", "apt ", "pip "}, "execute_command"},
		{[]string{"システムプロンプト変更", "プロンプト変更", "プロンプト修正", "自分を変え", "自己改変", "自分の設定", "ルール追加", "ルール変更"}, "self_modify_prompt"},
		{[]string{"自分の状態", "自己診断", "ステータス確認", "self status"}, "self_status"},
		{[]string{"ベンチマーク", "自己評価", "self benchmark"}, "self_benchmark"},
		{[]string{"ロールバック", "元に戻して", "前のバージョン"}, "self_rollback"},
	}

	for _, rule := range rules {
		for _, kw := range rule.keywords {
			if strings.Contains(lower, kw) {
				fmt.Printf("[siki] Keyword fallback: '%s' → %s\n", kw, rule.tool)
				return rule.tool
			}
		}
	}
	return ""
}

// subModelSummarize asks gpt-oss to generate a final response based on tool results.
func subModelSummarize(userMsg, toolName, toolResult string, config *Config) (string, error) {
	result := toolResult
	if len(result) > 4000 {
		result = result[:4000] + "\n... (以下省略)"
	}

	prompt := fmt.Sprintf(`## 絶対ルール
- 以下のツール結果だけを使って回答しろ。自分の知識で補完するな。
- ツール結果に無い情報は絶対に書くな。URLを捏造するな。
- ツール結果のURLをそのまま引用しろ。

## 回答品質の要件
- 各トピックについて、背景・詳細・影響を具体的に書け。見出しだけの羅列は禁止。
- ページ本文から得た具体的な数字・人名・技術名・引用を含めろ。
- 「〇〇が発表」だけでなく「何を、なぜ、どういう影響があるか」まで書け。
- 情報量が多い場合は、各ニュースをセクション（##）に分けて詳述せよ。

## ユーザーのリクエスト
%s

## 実行したツール: %s
## ツール結果（この情報だけで回答せよ）:
%s

上記のツール結果のみに基づいて、日本語で詳しく回答せよ。`, userMsg, toolName, result)

	response, err := callOllamaGenerate(config.SubModel, prompt, 32768, 300*time.Second, config)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}
	return response, nil
}

// validateURLsInResponse extracts URLs from the response and verifies them in parallel.
// URLs are decoded before checking (e.g. %2D → -).
// Total timeout: 5 seconds for all URL checks combined.
// Returns a list of bad URLs (original form) and a cleaned response.
func validateURLsInResponse(response string) (badURLs []string, cleanedResponse string) {
	urlRegex := regexp.MustCompile(`https?://[^\s\)\]>"]+`)
	allURLs := urlRegex.FindAllString(response, -1)
	if len(allURLs) == 0 {
		return nil, response
	}

	// Deduplicate and decode
	seen := map[string]bool{}
	type urlPair struct {
		original string
		decoded  string
	}
	var uniqueURLs []urlPair
	for _, u := range allURLs {
		u = strings.TrimRight(u, ".,;:!?\"')")
		if seen[u] {
			continue
		}
		seen[u] = true
		decoded := u
		if d, err := url.QueryUnescape(u); err == nil {
			decoded = d
		}
		uniqueURLs = append(uniqueURLs, urlPair{u, decoded})
	}

	// Parallel HEAD checks with 5s global deadline
	type checkResult struct {
		original string
		bad      bool
	}
	ch := make(chan checkResult, len(uniqueURLs))
	client := &http.Client{
		Timeout: 3 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}
	for _, pair := range uniqueURLs {
		go func(orig, decoded string) {
			req, err := http.NewRequest("HEAD", decoded, nil)
			if err != nil {
				ch <- checkResult{orig, true}
				return
			}
			req.Header.Set("User-Agent", "Mozilla/5.0")
			resp, err := client.Do(req)
			if err != nil {
				// Network error — don't mark as bad (might be timeout/DNS)
				ch <- checkResult{orig, false}
				return
			}
			resp.Body.Close()
			isBad := resp.StatusCode == 404 || resp.StatusCode == 410
			ch <- checkResult{orig, isBad}
		}(pair.original, pair.decoded)
	}

	badSet := map[string]bool{}
	deadline := time.After(5 * time.Second)
	collected := 0
	for collected < len(uniqueURLs) {
		select {
		case r := <-ch:
			collected++
			if r.bad {
				badSet[r.original] = true
			}
		case <-deadline:
			goto done
		}
	}
done:

	for u := range badSet {
		badURLs = append(badURLs, u)
	}

	// Remove lines containing bad URLs
	lines := strings.Split(response, "\n")
	var cleaned []string
	for _, line := range lines {
		hasBad := false
		for _, u := range urlRegex.FindAllString(line, -1) {
			u = strings.TrimRight(u, ".,;:!?\"')")
			if badSet[u] {
				hasBad = true
				break
			}
		}
		if !hasBad {
			cleaned = append(cleaned, line)
		}
	}

	if len(badURLs) > 0 {
		fmt.Printf("[siki] URL validation: %d bad URLs found: %v\n", len(badURLs), badURLs)
	}
	return badURLs, strings.Join(cleaned, "\n")
}

// validateResponse performs multi-layer validation (all programmatic, no LLM):
// 1. Length check (instant)
// 2. URL accessibility check (parallel HEAD, max 5s)
// 3. Content relevance check (keyword overlap between question/tool results/response)
// Returns (isValid, feedback, cleanedResponse).
// NOTE: Does NOT send SSE events itself — caller handles UI updates.
func validateResponse(userMsg, response, toolResult string, config *Config) (bool, string, string) {
	startTime := time.Now()

	// Layer 1: Length check
	if len(strings.TrimSpace(response)) < 30 {
		fmt.Printf("[siki] Validation: response too short (%d chars) (%.1fs)\n", len(response), time.Since(startTime).Seconds())
		return false, "回答が短すぎる", response
	}

	// Layer 2: URL validation (parallel, 5s max)
	badURLs, cleanedResponse := validateURLsInResponse(response)
	hasBadURLs := len(badURLs) > 0
	if hasBadURLs {
		allURLs := regexp.MustCompile(`https?://[^\s\)\]>"]+`).FindAllString(response, -1)
		fmt.Printf("[siki] Validation: %d/%d bad URLs (%.1fs)\n", len(badURLs), len(allURLs), time.Since(startTime).Seconds())
		if len(badURLs) > len(allURLs)/2 && len(allURLs) > 0 {
			return false, fmt.Sprintf("回答中のURLが%d/%d件無効（捏造の疑い）", len(badURLs), len(allURLs)), cleanedResponse
		}
		response = cleanedResponse
	}

	// Layer 3: Programmatic content relevance check (Japanese-aware)
	// Check if the response references data from tool results using substring matching
	if toolResult != "" && len(toolResult) > 100 {
		responseLower := strings.ToLower(response)
		// Extract distinctive terms from tool results (URLs, proper nouns, numbers)
		termRegex := regexp.MustCompile(`(?i)(?:https?://[^\s]+|[A-Z][a-zA-Z]{3,}|[0-9]{4}年|[0-9]+月[0-9]+日)`)
		terms := termRegex.FindAllString(toolResult, 50)
		overlapCount := 0
		for _, term := range terms {
			if strings.Contains(responseLower, strings.ToLower(term)) {
				overlapCount++
			}
		}
		// Also check for Japanese keyword overlap
		jaKeywords := []string{}
		for _, word := range strings.Fields(toolResult) {
			if len(word) >= 6 && len(word) <= 30 { // Multi-byte Japanese words
				jaKeywords = append(jaKeywords, word)
			}
		}
		for _, kw := range jaKeywords {
			if strings.Contains(response, kw) {
				overlapCount++
			}
		}
		if overlapCount == 0 {
			fmt.Printf("[siki] Validation: zero overlap with tool results — rejecting (%.1fs)\n", time.Since(startTime).Seconds())
			return false, "回答がツール結果に基づいていない（一般論の疑い）", response
		}
		fmt.Printf("[siki] Validation: tool result overlap=%d terms\n", overlapCount)
	}

	fmt.Printf("[siki] Validation: passed (badURLs=%d, %.1fs)\n", len(badURLs), time.Since(startTime).Seconds())
	return true, "", response
}

// generateSuggestions generates 3 follow-up question suggestions based on the topic.
// Uses keyword extraction (no LLM call) for instant results.
func generateSuggestions(userMsg, response, toolName string) []string {
	lower := strings.ToLower(userMsg)

	// Topic-specific suggestions
	if strings.Contains(lower, "ニュース") || strings.Contains(lower, "news") || strings.Contains(lower, "最新") {
		// Extract key topics from the response for targeted suggestions
		topics := []string{}
		topicKeywords := []string{
			"openai", "anthropic", "google", "microsoft", "meta",
			"claude", "gpt", "gemini", "llama", "grok",
			"セキュリティ", "ロボット", "自動運転", "医療", "教育",
			"規制", "オープンソース", "API", "価格",
		}
		responseLower := strings.ToLower(response)
		for _, kw := range topicKeywords {
			if strings.Contains(responseLower, strings.ToLower(kw)) {
				topics = append(topics, kw)
			}
		}

		suggestions := []string{}
		if len(topics) >= 2 {
			suggestions = append(suggestions, fmt.Sprintf("%sについて詳しく教えて", topics[0]))
			suggestions = append(suggestions, fmt.Sprintf("%sと%sの関連性は？", topics[0], topics[1]))
		}
		if len(topics) >= 3 {
			suggestions = append(suggestions, fmt.Sprintf("%sの最新動向を深掘りして", topics[2]))
		}

		// Fill remaining slots with generic follow-ups
		defaults := []string{
			"それぞれのニュースを深掘りして",
			"日本への影響は？",
			"今後の展望を教えて",
		}
		for _, d := range defaults {
			if len(suggestions) >= 3 {
				break
			}
			suggestions = append(suggestions, d)
		}
		return suggestions[:3]
	}

	if strings.Contains(lower, "http") || toolName == "web_fetch" {
		return []string{
			"要約して",
			"重要なポイントを箇条書きにして",
			"関連する情報を検索して",
		}
	}

	if toolName == "run_code" || toolName == "diagram" {
		return []string{
			"もっとかっこよくして",
			"色やデザインを変えて",
			"機能を追加して",
		}
	}

	// Generic follow-ups
	return []string{
		"もっと詳しく教えて",
		"具体例を挙げて",
		"関連する情報を調べて",
	}
}

// streamSubModelSummarize streams gpt-oss's summary response token-by-token via SSE.
func streamSubModelSummarize(userMsg, toolName, toolResult string, config *Config, sendEvent func(StreamEvent)) (string, error) {
	return streamSubModelSummarizeWith(userMsg, toolName, toolResult, config, sendEvent, "")
}

// streamSubModelSummarizeWith is like streamSubModelSummarize but allows overriding the model.
func streamSubModelSummarizeWith(userMsg, toolName, toolResult string, config *Config, sendEvent func(StreamEvent), modelOverride string) (string, error) {
	model := config.SubModel
	if modelOverride != "" {
		model = modelOverride
	}

	result := toolResult
	maxResult := 20000
	if hasSubAgent(config) {
		maxResult = 60000
	}
	if len(result) > maxResult {
		result = result[:maxResult] + "\n... (以下省略)"
	}

	prompt := fmt.Sprintf(`## 絶対ルール
- 以下のツール結果だけを使って回答しろ。自分の知識で補完するな。
- ツール結果に無い情報は絶対に書くな。URLを捏造するな。
- ツール結果のURLをそのまま引用しろ。

## 回答品質の要件
- 各トピックについて、背景・詳細・影響を具体的に書け。見出しだけの羅列は禁止。
- ページ本文から得た具体的な数字・人名・技術名・引用を含めろ。
- 「〇〇が発表」だけでなく「何を、なぜ、どういう影響があるか」まで書け。
- 情報量が多い場合は、各ニュースをセクション（##）に分けて詳述せよ。

## ユーザーのリクエスト
%s

## 実行したツール: %s
## ツール結果（この情報だけで回答せよ）:
%s

上記のツール結果のみに基づいて、日本語で詳しく回答せよ。`, userMsg, toolName, result)

	// Use sub-agent for summarization if available (more powerful model)
	if modelOverride == "" && hasSubAgent(config) {
		fmt.Printf("[siki] Using sub-agent (%s) for summarization\n", config.SubAgent)
		return streamSubAgentGenerate(prompt, config, sendEvent)
	}

	endpoint := subModelEndpoint(config)

	// Use vllm (OpenAI-compatible) streaming API if configured
	if modelOverride == "" && isSubModelVLLM(config) {
		return streamVLLMGenerate(model, prompt, 32768, 300*time.Second, endpoint, sendEvent)
	}

	// Ollama native streaming API
	reqBody := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     true,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": 32768,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}

	client := &http.Client{Timeout: 300 * time.Second}
	resp, err := client.Post(endpoint+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("stream request error: %w", err)
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	inThinking := false
	decoder := json.NewDecoder(resp.Body)

	for decoder.More() {
		var chunk struct {
			Response string `json:"response"`
			Thinking string `json:"thinking"`
			Done     bool   `json:"done"`
		}
		if err := decoder.Decode(&chunk); err != nil {
			break
		}

		// Capture thinking tokens as thinking events (instead of discarding)
		if chunk.Thinking != "" {
			inThinking = true
			sendEvent(StreamEvent{Type: "thinking", Content: chunk.Thinking})
			continue
		}
		if inThinking && chunk.Response != "" {
			inThinking = false
		}

		if chunk.Response != "" {
			text := chunk.Response
			// Handle inline <think> tags
			if strings.Contains(text, "<think>") {
				inThinking = true
				sendEvent(StreamEvent{Type: "thinking", Content: text})
				continue
			}
			if strings.Contains(text, "</think>") {
				inThinking = false
				continue
			}
			if inThinking {
				sendEvent(StreamEvent{Type: "thinking", Content: text})
				continue
			}
			fullResponse.WriteString(text)
			sendEvent(StreamEvent{Type: "content", Content: text})
		}

		if chunk.Done {
			break
		}
	}

	return fullResponse.String(), nil
}

// isDissatisfied checks if the user message expresses dissatisfaction with the previous response.
func isDissatisfied(msg string) bool {
	lower := strings.ToLower(msg)
	keywords := []string{
		"だめ", "ダメ", "駄目", "やり直し", "違う", "もっと良く", "いまいち", "再生成",
		"改善して", "修正して", "直して", "やりなおし", "もう一度", "もっといい",
		"よくない", "不満", "気に入らない", "ちがう", "ちゃんとして", "もっとちゃんと",
		"redo", "try again", "not good", "wrong", "fix it", "improve",
		"do it again", "retry", "bad", "terrible", "ひどい", "使えない",
	}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// retryWithEscalation retries the last tool execution with a more powerful model or enhanced prompt.
// Returns the new response, or empty string if escalation is not possible.
func (ws *WebServer) retryWithEscalation(ctx context.Context, convID string, agent *Agent, userMsg string, sendEvent func(StreamEvent), saveMsg func(Message, string)) string {
	ws.mu.RLock()
	last := ws.lastExec[convID]
	ws.mu.RUnlock()
	if last == nil {
		return ""
	}

	fmt.Printf("[siki] Dissatisfaction detected, escalating. Previous tool: %s, usedAgent: %v\n", last.ToolName, last.UsedAgent)

	// For visual tools (run_code, diagram, generate_image), re-execute the tool with feedback
	if last.ToolName == "run_code" || last.ToolName == "diagram" || last.ToolName == "generate_image" {
		sendEvent(modelThinkingEvent("前回の結果を改善中...", ws.config, hasSubAgent(ws.config)))

		// Generate improved args via sub-agent/sub-model
		feedbackPrompt := fmt.Sprintf(`前回のユーザーリクエスト: %s
前回のツール: %s
ユーザーのフィードバック: %s

ユーザーの不満を踏まえて、改善した結果を生成してください。`, last.UserMsg, last.ToolName, userMsg)

		if last.ToolName == "generate_image" {
			// Re-generate with enhanced prompt
			sendEvent(StreamEvent{Type: "tool_start", Name: "generate_image"})
			enhancePrompt := fmt.Sprintf(`前回の画像生成プロンプトの結果にユーザーが不満です。
前回のリクエスト: %s
ユーザーのフィードバック: %s

より良い英語の画像生成プロンプトを1つだけ出力せよ。`, last.UserMsg, userMsg)
			_, enhanced, err := callSubAgent(enhancePrompt, ws.config)
			if err == nil && len(enhanced) > 10 {
				enhanced = strings.TrimSpace(enhanced)
				enhanced = strings.TrimPrefix(enhanced, "```")
				enhanced = strings.TrimSuffix(enhanced, "```")
				enhanced = strings.TrimSpace(enhanced)
				if len(enhanced) > 2 && enhanced[0] == '"' && enhanced[len(enhanced)-1] == '"' {
					enhanced = enhanced[1 : len(enhanced)-1]
				}
				args := map[string]interface{}{"prompt": enhanced}
				result, err := agent.executeTool("generate_image", args)
				if err == nil {
					sendEvent(StreamEvent{Type: "tool_call", Name: "generate_image", Result: result})
					finalMsg := Message{Role: "assistant", Content: result}
					agent.messages = append(agent.messages, finalMsg)
					saveMsg(finalMsg, "")
					return result
				}
			}
		} else if last.ToolName == "run_code" {
			// Regenerate code with feedback
			sendEvent(StreamEvent{Type: "tool_start", Name: "run_code"})
			newHTML, err := generateCodeWithSubModel(feedbackPrompt, ws.config)
			if err == nil && len(newHTML) > 50 {
				args := map[string]interface{}{"html": newHTML}
				result, err := agent.executeTool("run_code", args)
				if err == nil {
					sendEvent(StreamEvent{Type: "tool_call", Name: "run_code", Result: result})
					finalMsg := Message{Role: "assistant", Content: result}
					agent.messages = append(agent.messages, finalMsg)
					saveMsg(finalMsg, "")
					return result
				}
			}
		}
		// For diagram, fall through to re-summarize
	}

	// For text-based results: escalate to sub-agent if not already used
	if hasSubAgent(ws.config) && !last.UsedAgent {
		fmt.Printf("[siki] Escalating to sub-agent: %s\n", ws.config.SubAgent)
		sendEvent(modelThinkingEvent("より強力なモデルで再処理中...", ws.config, true))

		escalatePrompt := fmt.Sprintf(`## 前回の回答に対するユーザーの不満
ユーザーのフィードバック: %s

## 元のリクエスト
%s

## 前回使用したツール: %s
## ツール結果:
%s

上記のツール結果に基づき、ユーザーの不満を踏まえてより良い回答を日本語で生成せよ。`, userMsg, last.UserMsg, last.ToolName, last.ToolResult)

		resp, err := streamSubAgentGenerate(escalatePrompt, ws.config, sendEvent)
		if err == nil && len(strings.TrimSpace(resp)) > 20 {
			finalMsg := Message{Role: "assistant", Content: resp}
			agent.messages = append(agent.messages, finalMsg)
			saveMsg(finalMsg, "")
			// Update lastExec to reflect sub-agent usage
			ws.mu.Lock()
			ws.lastExec[convID] = &LastToolExecution{
				UserMsg:    last.UserMsg,
				ToolName:   last.ToolName,
				Args:       last.Args,
				ToolResult: last.ToolResult,
				Response:   resp,
				UsedAgent:  true,
			}
			ws.mu.Unlock()
			return resp
		}
	}

	// Same model retry with feedback-enhanced prompt
	if last.ToolResult != "" {
		sendEvent(modelThinkingEvent("フィードバックを反映して再生成中...", ws.config, hasSubAgent(ws.config)))
		feedbackResult := last.ToolResult + fmt.Sprintf("\n\n## ユーザーフィードバック:\nユーザーは前回の回答に不満です: %s\nこのフィードバックを踏まえて、より良い回答を生成せよ。", userMsg)
		resp, err := streamSubModelSummarize(last.UserMsg, last.ToolName, feedbackResult, ws.config, sendEvent)
		if err == nil && len(strings.TrimSpace(resp)) > 20 {
			finalMsg := Message{Role: "assistant", Content: resp}
			agent.messages = append(agent.messages, finalMsg)
			saveMsg(finalMsg, "")
			ws.mu.Lock()
			ws.lastExec[convID] = &LastToolExecution{
				UserMsg:    last.UserMsg,
				ToolName:   last.ToolName,
				Args:       last.Args,
				ToolResult: last.ToolResult,
				Response:   resp,
				UsedAgent:  last.UsedAgent,
			}
			ws.mu.Unlock()
			return resp
		}
	}

	return ""
}

// dualModelPipeline coordinates lfm (fast ack) and gpt-oss (real orchestration).
// Returns the final assistant reply text.
func (ws *WebServer) dualModelPipeline(ctx context.Context, agent *Agent, userMsg string, sendEvent func(StreamEvent), saveMsg func(Message, string), convID ...string) string {
	// Resolve conversation ID (optional parameter for backwards compatibility)
	cID := ""
	if len(convID) > 0 {
		cID = convID[0]
	}

	// Check for dissatisfaction and attempt escalation recovery
	if cID != "" && isDissatisfied(userMsg) {
		escalated := ws.retryWithEscalation(ctx, cID, agent, userMsg, sendEvent, saveMsg)
		if escalated != "" {
			return escalated
		}
		// Fall through to normal pipeline if escalation didn't work
	}

	// Fast-path: skip orchestrator for Bluesky requests
	if containsBlueskyKeywords(userMsg) && ws.config.BlueskyEnabled {
		// Detect if this is a search request
		bskyTool := "bluesky_feed"
		if isBlueskySearchRequest(userMsg) {
			bskyTool = "bluesky_search"
		}
		fmt.Printf("[siki] Fast-path: %s (skipping orchestrator)\n", bskyTool)
		agent.sendEvent = sendEvent
		args := map[string]interface{}{}

		sendEvent(StreamEvent{Type: "tool_start", Name: bskyTool})
		result, err := agent.executeTool(bskyTool, args)
		if err != nil {
			sendEvent(StreamEvent{Type: "error", Error: fmt.Sprintf("Bluesky error: %v", err)})
			return ""
		}
		sendEvent(StreamEvent{Type: "tool_call", Name: bskyTool, Result: result})

		toolCallID := fmt.Sprintf("fast-%d", time.Now().UnixMilli())
		argsJSON, _ := json.Marshal(args)
		assistantMsg := Message{Role: "assistant", ToolCalls: []ToolCall{{ID: toolCallID, Type: "function", Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: bskyTool, Arguments: string(argsJSON)}}}}
		agent.messages = append(agent.messages, assistantMsg)
		saveMsg(assistantMsg, "")
		toolMsg := Message{Role: "tool", Content: result, ToolCallID: toolCallID}
		agent.messages = append(agent.messages, toolMsg)
		saveMsg(toolMsg, bskyTool)

		// Generate summary with gpt-oss
		summaryInput := result
		if len([]rune(summaryInput)) > 6000 {
			summaryInput = string([]rune(summaryInput)[:6000]) + "\n...(省略)"
		}
		summaryPrompt := fmt.Sprintf(`あなたはニュースキュレーターです。以下はBlueskyから取得した投稿です。ユーザーの要求「%s」に基づき、重要なニュースや話題をわかりやすくまとめてください。
- 重要度の高い順に整理
- 各トピックの要点を簡潔に
- 注目すべきリンクがあれば言及
- 日本語で回答

選別結果:
%s`, userMsg, summaryInput)

		sendEvent(StreamEvent{Type: "progress", Content: "gpt-ossでまとめを生成中..."})
		summary, err := streamVLLMGenerate(
			ws.config.ModelName,
			summaryPrompt,
			2048,
			120*time.Second,
			ws.config.APIEndpoint,
			sendEvent,
		)
		if err != nil {
			fmt.Printf("[siki] vllm summary failed: %v, trying ollama gpt-oss...\n", err)
			summary, err = streamOllamaGenerate("gpt-oss:latest", summaryPrompt, 2048, 120*time.Second, sendEvent)
			if err != nil {
				fmt.Printf("[siki] ollama summary also failed: %v, using raw result\n", err)
				summary = result
			}
		}

		finalMsg := Message{Role: "assistant", Content: summary}
		agent.messages = append(agent.messages, finalMsg)
		saveMsg(finalMsg, "")
		sendEvent(StreamEvent{Type: "done"})
		if cID != "" {
			ws.mu.Lock()
			ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: bskyTool, Args: args, ToolResult: result, Response: summary, UsedAgent: false}
			ws.mu.Unlock()
		}
		return summary
	}

	// Fast-path: skip orchestrator entirely for Twitter requests (keyword match is definitive)
	if containsTwitterKeywords(userMsg) {
		twitterTool := detectTwitterTool(userMsg)
		fmt.Printf("[siki] Fast-path: %s (skipping orchestrator)\n", twitterTool)

		agent.sendEvent = sendEvent // allow tools to emit progress to UI
		args := map[string]interface{}{}
		if twitterTool == "twitter_search" {
			args["query"] = extractSearchQuery(userMsg, "twitter", nil)
		}

		sendEvent(StreamEvent{Type: "tool_start", Name: twitterTool})
		result, err := agent.executeTool(twitterTool, args)
		if err != nil {
			sendEvent(StreamEvent{Type: "error", Error: fmt.Sprintf("Twitter error: %v", err)})
			return ""
		}
		sendEvent(StreamEvent{Type: "tool_call", Name: twitterTool, Result: result})

		// Save tool call to conversation
		toolCallID := fmt.Sprintf("fast-%d", time.Now().UnixMilli())
		argsJSON, _ := json.Marshal(args)
		assistantMsg := Message{Role: "assistant", ToolCalls: []ToolCall{{ID: toolCallID, Type: "function", Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: twitterTool, Arguments: string(argsJSON)}}}}
		agent.messages = append(agent.messages, assistantMsg)
		saveMsg(assistantMsg, "")
		toolMsg := Message{Role: "tool", Content: result, ToolCallID: toolCallID}
		agent.messages = append(agent.messages, toolMsg)
		saveMsg(toolMsg, twitterTool)

		// Generate final summary using gpt-oss (streaming)
		// Truncate result to fit context
		summaryInput := result
		if len([]rune(summaryInput)) > 6000 {
			summaryInput = string([]rune(summaryInput)[:6000]) + "\n...(省略)"
		}
		summaryPrompt := fmt.Sprintf(`あなたはニュースキュレーターです。以下はTwitterタイムラインからAIが選別したツイートです。ユーザーの要求「%s」に基づき、重要なニュースや話題をわかりやすくまとめてください。
- 重要度の高い順に整理
- 各トピックの要点を簡潔に
- 注目すべきリンクがあれば言及
- 日本語で回答

選別結果:
%s`, userMsg, summaryInput)

		sendEvent(StreamEvent{Type: "progress", Content: "gpt-ossでまとめを生成中..."})
		// Try vllm first, fallback to ollama
		summary, err := streamVLLMGenerate(
			ws.config.ModelName,
			summaryPrompt,
			2048,
			120*time.Second,
			ws.config.APIEndpoint,
			sendEvent,
		)
		if err != nil {
			fmt.Printf("[siki] vllm summary failed: %v, trying ollama gpt-oss...\n", err)
			summary, err = streamOllamaGenerate("gpt-oss:latest", summaryPrompt, 2048, 120*time.Second, sendEvent)
			if err != nil {
				fmt.Printf("[siki] ollama summary also failed: %v, using raw result\n", err)
				summary = result
			}
		}

		finalMsg := Message{Role: "assistant", Content: summary}
		agent.messages = append(agent.messages, finalMsg)
		saveMsg(finalMsg, "")
		sendEvent(StreamEvent{Type: "done"})
		if cID != "" {
			ws.mu.Lock()
			ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: twitterTool, Args: args, ToolResult: result, Response: summary, UsedAgent: false}
			ws.mu.Unlock()
		}
		return summary
	}

	// Phase 1: Quick ack from lfm (parallel with Phase 2)
	ackCh := make(chan string, 1)
	go func() {
		ackCh <- quickAck(userMsg, ws.config)
	}()

	// Phase 2: gpt-oss orchestration (parallel with Phase 1)
	type orchResult struct {
		decision *OrchestratorDecision
		err      error
	}
	orchCh := make(chan orchResult, 1)
	go func() {
		d, err := subModelOrchestrate(userMsg, agent.messages, ws.config)
		orchCh <- orchResult{d, err}
	}()

	// Show ack immediately (keyword-based, instant)
	select {
	case ack := <-ackCh:
		if ack != "" {
			sendEvent(StreamEvent{Type: "content", Content: ack + "\n\n"})
		}
	case <-time.After(1 * time.Second):
	}

	// Show thinking indicator while waiting for orchestrator (with keepalive)
	sendEvent(orchestratorThinkingEvent("オーケストレーターで処理中...", ws.config))

	// Wait for orchestrator decision with periodic keepalive
	var decision *OrchestratorDecision
	orchTicker := time.NewTicker(2 * time.Second)
	orchDone := false
	for !orchDone {
		select {
		case r := <-orchCh:
			orchDone = true
			if r.err != nil {
				orchTicker.Stop()
				sendEvent(StreamEvent{Type: "error", Error: fmt.Sprintf("Orchestration error: %v", r.err)})
				return ""
			}
			decision = r.decision
		case <-orchTicker.C:
			sendEvent(orchestratorThinkingEvent("処理中...", ws.config))
		case <-ctx.Done():
			orchTicker.Stop()
			return ""
		}
	}
	orchTicker.Stop()

	// FIRST: Check for follow-up questions BEFORE any tool/none branching.
	// Follow-ups like "深掘りして" "詳しく" should fetch previous URLs, not hallucinate.
	if isFollowUpQuery(userMsg) {
		prevURLs := extractURLsFromConversation(agent.messages)
		if len(prevURLs) > 0 {
			fmt.Printf("[siki] Follow-up detected, fetching %d previous URLs\n", len(prevURLs))
			if len(prevURLs) > 5 {
				prevURLs = prevURLs[:5]
			}
			sendEvent(StreamEvent{Type: "tool_start", Name: "web_fetch"})
			type fetchResult struct {
				url     string
				content string
			}
			ch := make(chan fetchResult, len(prevURLs))
			for _, u := range prevURLs {
				go func(fetchURL string) {
					pageContent, err := agent.webFetchQuick(fetchURL)
					if err != nil {
						fmt.Printf("[siki] Failed to fetch %s: %v\n", fetchURL, err)
						ch <- fetchResult{fetchURL, ""}
						return
					}
					if len(pageContent) > 3000 {
						pageContent = pageContent[:3000] + "\n...(省略)"
					}
					ch <- fetchResult{fetchURL, pageContent}
				}(u)
			}
			var fetchedContent strings.Builder
			ticker := time.NewTicker(2 * time.Second)
			collected := 0
			for collected < len(prevURLs) {
				select {
				case fr := <-ch:
					collected++
					sendEvent(StreamEvent{Type: "thinking", Content: fmt.Sprintf("ページ取得完了 (%d/%d)", collected, len(prevURLs))})
					if fr.content != "" {
						fetchedContent.WriteString(fmt.Sprintf("\n## %s\n%s\n", fr.url, fr.content))
					}
				case <-ticker.C:
					sendEvent(StreamEvent{Type: "thinking", Content: "ページ取得中..."})
				}
			}
			ticker.Stop()
			sendEvent(StreamEvent{Type: "tool_call", Name: "web_fetch", Result: fmt.Sprintf("%d件のページを取得しました", len(prevURLs))})

			sendEvent(modelThinkingEvent("回答を生成中...", ws.config, hasSubAgent(ws.config)))
			fetchedStr := fetchedContent.String()
			finalResponse, err := streamSubModelSummarize(userMsg, "web_fetch", fetchedStr, ws.config, sendEvent)
			if err != nil || finalResponse == "" {
				finalResponse = "ページの取得に失敗しました。"
				sendEvent(StreamEvent{Type: "content", Content: finalResponse})
			} else {
				// Validate the follow-up response
				isValid, reason, cleaned := validateResponse(userMsg, finalResponse, fetchedStr, ws.config)
				if !isValid {
					fmt.Printf("[siki] Follow-up validation failed: %s — retrying\n", reason)
					sendEvent(modelThinkingEvent(fmt.Sprintf("回答品質チェック不合格: %s — 再生成します", reason), ws.config, hasSubAgent(ws.config)))
					retryResponse, retryErr := streamSubModelSummarize(userMsg, "web_fetch", fetchedStr+"\n\n## 前回の回答が不合格だった理由:\n"+reason, ws.config, sendEvent)
					if retryErr == nil && retryResponse != "" {
						finalResponse = retryResponse
					}
				} else {
					finalResponse = cleaned
				}
			}
			finalMsg := Message{Role: "assistant", Content: finalResponse}
			agent.messages = append(agent.messages, finalMsg)
			saveMsg(finalMsg, "")
			suggestions := generateSuggestions(userMsg, finalResponse, "web_fetch")
			sendEvent(StreamEvent{Type: "suggestions", Suggestions: suggestions})
			if cID != "" {
				ws.mu.Lock()
				ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: "web_fetch", ToolResult: fetchedStr, Response: finalResponse, UsedAgent: hasSubAgent(ws.config)}
				ws.mu.Unlock()
			}
			return finalResponse
		}
	}

	// If orchestration failed to produce a decision, fall back to keyword detection
	if decision.Tool == "" {
		fmt.Printf("[siki] Orchestration returned no tool, using keyword fallback\n")
		detected := detectToolFromKeywords(userMsg)
		if detected != "" {
			decision.Tool = detected
			decision.Args = nil
		}
	}

	// Force twitter tool when message contains Twitter keywords
	// The orchestrator often misroutes these to web_search
	if decision.Tool != "twitter_search" && decision.Tool != "twitter_timeline" && containsTwitterKeywords(userMsg) {
		twitterTool := detectTwitterTool(userMsg)
		fmt.Printf("[siki] Forcing %s: Twitter keyword detected (was: %s)\n", twitterTool, decision.Tool)
		decision.Tool = twitterTool
		decision.Args = nil
	}

	// Force bluesky tool when Bluesky keywords detected
	if decision.Tool != "bluesky_feed" && decision.Tool != "bluesky_search" && containsBlueskyKeywords(userMsg) {
		if isBlueskySearchRequest(userMsg) {
			fmt.Printf("[siki] Forcing bluesky_search: Bluesky search keyword detected (was: %s)\n", decision.Tool)
			decision.Tool = "bluesky_search"
		} else {
			fmt.Printf("[siki] Forcing bluesky_feed: Bluesky keyword detected (was: %s)\n", decision.Tool)
			decision.Tool = "bluesky_feed"
		}
		decision.Args = nil
	}

	// Force docker_run_model when HuggingFace/GitHub URL + action word is present
	// The orchestrator often misroutes these to generate_video or web_fetch
	if decision.Tool != "docker_run_model" && isHFGitHubRunRequest(userMsg) {
		fmt.Printf("[siki] Forcing docker_run_model: HF/GitHub URL + action detected (was: %s)\n", decision.Tool)
		decision.Tool = "docker_run_model"
		decision.Args = nil
	}

	// Force plan mode for comic requests (4コマ漫画 requires multi-step: scenario + 4 images)
	if decision.Tool != "plan" && isComicRequest(userMsg) {
		fmt.Printf("[siki] Forcing plan mode: comic request detected (was: %s)\n", decision.Tool)
		decision.Tool = "plan"
	}

	// Force plan mode for complex requests needing research + specific output (image/code)
	// The orchestrator often fails to detect multi-step requirements.
	if decision.Tool != "plan" && needsImageGeneration(userMsg) && containsResearchKeywords(userMsg) {
		fmt.Printf("[siki] Forcing plan mode: research + image generation detected\n")
		decision.Tool = "plan"
	}

	// No tool needed — but verify: should a tool have been called?
	if decision.Tool == "none" || decision.Tool == "" {
		// Self-check: re-evaluate with keyword detection before committing to no-tool answer
		requiredTool := detectToolFromKeywords(userMsg)
		if requiredTool != "" {
			fmt.Printf("[siki] Self-check: orchestrator said no tool, but keyword detected '%s' — overriding\n", requiredTool)
			decision.Tool = requiredTool
			decision.Args = nil
			// Fall through to tool execution below
		} else {
			fmt.Printf("[siki] No tool, streaming direct answer from gpt-oss\n")
			resp, _ := streamSubModelSummarize(userMsg, "none", "", ws.config, sendEvent)
			if resp == "" {
				resp = "すみません、うまく処理できませんでした。もう一度お試しください。"
				sendEvent(StreamEvent{Type: "content", Content: resp})
			}
			// Post-response validation: check if the answer looks hallucinated
			if needsToolButDidnt(userMsg, resp) {
				fmt.Printf("[siki] Post-response validation failed: response looks hallucinated, retrying with tool\n")
				retryTool := detectToolFromKeywords(userMsg)
				if retryTool == "" || retryTool == "plan" {
					// 会話検索系キーワードがあれば search_threads を優先
					if containsConversationKeywords(userMsg) {
						retryTool = "search_threads"
					} else {
						retryTool = "web_search"
					}
				}
				sendEvent(modelThinkingEvent("回答を検証中...ツールで再確認します", ws.config, hasSubAgent(ws.config)))
				altModel := pickRetryModel(ws.config, 1)
				fmt.Printf("[siki] Retry #1 using alternate model: %s\n", altModel)
				retryResult := ws.executeToolAndSummarize(agent, userMsg, retryTool, sendEvent, saveMsg, altModel)
				if retryResult != "" {
					return retryResult
				}
			}
			// Validate content quality (URL check + sub-model judgment)
			isValid, reason, cleaned := validateResponse(userMsg, resp, "", ws.config)
			if !isValid {
				fmt.Printf("[siki] Direct answer validation failed: %s\n", reason)
				// Try with a tool as fallback — use alternate model for retry
				retryTool := "web_search"
				altModel := pickRetryModel(ws.config, 2)
				fmt.Printf("[siki] Retry #2 using alternate model: %s\n", altModel)
				sendEvent(modelThinkingEvent(fmt.Sprintf("回答品質チェック不合格: %s — ツールで再検索します", reason), ws.config, hasSubAgent(ws.config)))
				retryResult := ws.executeToolAndSummarize(agent, userMsg, retryTool, sendEvent, saveMsg, altModel)
				if retryResult != "" {
					return retryResult
				}
			} else {
				resp = cleaned
			}
			assistantMsg := Message{Role: "assistant", Content: resp}
			agent.messages = append(agent.messages, assistantMsg)
			saveMsg(assistantMsg, "")
			suggestions := generateSuggestions(userMsg, resp, "none")
			sendEvent(StreamEvent{Type: "suggestions", Suggestions: suggestions})
			if cID != "" {
				ws.mu.Lock()
				ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: "none", Response: resp, UsedAgent: false}
				ws.mu.Unlock()
			}
			return resp
		}
	}

	// Plan mode: create and execute a multi-step plan
	if decision.Tool == "plan" {
		fmt.Printf("[siki] Entering plan mode for: %s\n", userMsg)

		// Always use the user's original message as the goal.
		// Do NOT use decision.Args["goal"] — the sub-model may inject biased
		// implementation details (e.g., "via HTML") that constrain createPlan.
		goal := userMsg

		// Use specialized comic plan for 4コマ漫画 requests
		var plan *Plan
		var err error
		if isComicRequest(userMsg) {
			sendEvent(modelThinkingEvent("4コマ漫画のシナリオを作成中...", ws.config, hasSubAgent(ws.config)))
			plan, err = createComicPlan(goal, agent.messages, ws.config)
			// Retry once on failure (LLM may produce invalid JSON on first try)
			if err != nil {
				fmt.Printf("[siki] Comic plan first attempt failed: %v, retrying...\n", err)
				sendEvent(modelThinkingEvent("シナリオ生成を再試行中...", ws.config, hasSubAgent(ws.config)))
				plan, err = createComicPlan(goal, agent.messages, ws.config)
			}
		} else {
			sendEvent(modelThinkingEvent("複雑なタスクを検出。プランを作成中...", ws.config, hasSubAgent(ws.config)))
			plan, err = createPlan(goal, agent.messages, ws.config)
		}
		if err != nil {
			fmt.Printf("[siki] Plan creation failed: %v, falling back to direct execution\n", err)
			// For comic requests: don't fall back to web_search, return error message
			// so user knows the comic couldn't be generated
			if isComicRequest(userMsg) {
				errMsg := fmt.Sprintf("4コマ漫画のシナリオ生成に失敗しました（%v）。もう一度お試しください。", err)
				sendEvent(StreamEvent{Type: "content", Content: errMsg})
				finalMsg := Message{Role: "assistant", Content: errMsg}
				agent.messages = append(agent.messages, finalMsg)
				saveMsg(finalMsg, "")
				return errMsg
			}
			sendEvent(modelThinkingEvent("プラン作成失敗。直接実行します...", ws.config, false))
			// Fall through to normal tool execution with web_search as fallback
			fallbackTool := detectToolFromKeywords(userMsg)
			if fallbackTool == "" || fallbackTool == "plan" {
				fallbackTool = "web_search"
			}
			decision.Tool = fallbackTool
			decision.Args = map[string]interface{}{"query": userMsg}
		} else {
			planID := fmt.Sprintf("plan_%d", time.Now().UnixMilli())

			// Display the plan as a TODO list
			var planDisplay strings.Builder
			planDisplay.WriteString(fmt.Sprintf("## 📋 実行プラン\n**目標:** %s\n\n", goal))
			for _, t := range plan.Tasks {
				planDisplay.WriteString(fmt.Sprintf("- [ ] **タスク%d:** %s (`%s`)\n", t.ID, t.Description, t.Tool))
			}
			planDisplay.WriteString(fmt.Sprintf("\n---\n*%d個のタスクを順次実行します*\n\n", len(plan.Tasks)))
			sendEvent(StreamEvent{Type: "content", Content: planDisplay.String()})

			// Save plan and execute
			savePlan(plan, planID)
			result := ws.executePlan(ctx, plan, planID, agent, sendEvent, saveMsg)

			// Goal verification: check if the plan actually fulfilled the user's request
			goalMet := verifyGoalFulfillment(plan, userMsg)
			if !goalMet {
				fmt.Printf("[siki] Goal verification FAILED for: %s\n", userMsg)
				// Collect failure details
				var failures []string
				for _, t := range plan.Tasks {
					if t.Status == "failed" {
						failures = append(failures, fmt.Sprintf("タスク%d(%s): %s", t.ID, t.Tool, t.Result))
					}
				}
				failMsg := "⚠️ **一部のタスクが完了できませんでした:**\n"
				for _, f := range failures {
					failMsg += "- " + f + "\n"
				}
				failMsg += "\nもう一度お試しいただくか、別の表現でリクエストしてください。"
				sendEvent(StreamEvent{Type: "content", Content: "\n\n" + failMsg})
				result += "\n\n" + failMsg
			}

			// Save final result
			finalMsg := Message{Role: "assistant", Content: result}
			agent.messages = append(agent.messages, finalMsg)
			saveMsg(finalMsg, "")

			suggestions := generateSuggestions(userMsg, result, "plan")
			sendEvent(StreamEvent{Type: "suggestions", Suggestions: suggestions})
			if cID != "" {
				ws.mu.Lock()
				ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: "plan", Response: result, UsedAgent: hasSubAgent(ws.config)}
				ws.mu.Unlock()
			}
			return result
		}
	}

	// Tool execution
	toolName := decision.Tool

	// Redirect to generate_image if user clearly wants AI image generation
	// (orchestrator may misroute to diagram/run_code)
	if toolName != "generate_image" && needsImageGeneration(userMsg) {
		fmt.Printf("[siki] Redirecting %s → generate_image (image generation keywords detected)\n", toolName)
		toolName = "generate_image"
	}

	// Redirect diagram → run_code for complex visualizations
	if toolName == "diagram" && needsRunCode(userMsg) {
		fmt.Printf("[siki] Redirecting diagram → run_code\n")
		toolName = "run_code"
	}

	sendEvent(StreamEvent{Type: "tool_start", Name: toolName})

	// Use gpt-oss's args, but for run_code ensure we have good HTML
	args := decision.Args
	if args == nil {
		args = map[string]interface{}{}
	}

	// Set default args for keyword-detected tools
	if toolName == "web_search" {
		if _, ok := args["query"]; !ok {
			args["query"] = userMsg
		}
	} else if toolName == "twitter_search" {
		if _, ok := args["query"]; !ok {
			// Use LLM to extract a good search query from the user message
			args["query"] = extractSearchQuery(userMsg, "twitter", ws.config)
		}
	} else if toolName == "search_threads" || toolName == "search_conversation" || toolName == "recall_context" {
		if _, ok := args["query"]; !ok {
			args["query"] = userMsg
		}
	} else if strings.HasPrefix(toolName, "self_") {
		// Self-modification tools: ensure args are populated from orchestrator or default
		if toolName == "self_modify_prompt" {
			if _, ok := args["action"]; !ok {
				args["action"] = "append"
			}
			if _, ok := args["reason"]; !ok {
				args["reason"] = userMsg
			}
			if _, ok := args["content"]; !ok {
				args["content"] = userMsg
			}
		} else if toolName == "self_add_rule" {
			if _, ok := args["rule"]; !ok {
				args["rule"] = userMsg
			}
			if _, ok := args["reason"]; !ok {
				args["reason"] = "ユーザーリクエスト"
			}
		}
	} else if toolName == "web_fetch" {
		if _, ok := args["url"]; !ok {
			// Extract URL from user message
			for _, word := range strings.Fields(userMsg) {
				if strings.HasPrefix(word, "http://") || strings.HasPrefix(word, "https://") {
					args["url"] = word
					break
				}
			}
		}
	} else if toolName == "docker_run_model" {
		// Extract HuggingFace/GitHub URL from user message
		if _, ok := args["url"]; !ok {
			for _, word := range strings.Fields(userMsg) {
				if (strings.Contains(word, "huggingface.co/") || strings.Contains(word, "github.com/")) &&
					strings.HasPrefix(word, "http") {
					args["url"] = word
					break
				}
			}
		}
		// Use remaining text (minus URL) as prompt
		if _, ok := args["prompt"]; !ok {
			var parts []string
			for _, word := range strings.Fields(userMsg) {
				if !strings.HasPrefix(word, "http") {
					parts = append(parts, word)
				}
			}
			args["prompt"] = strings.Join(parts, " ")
		}
	}

	// Validate run_code HTML — regenerate if too short
	if toolName == "run_code" {
		html, _ := args["html"].(string)
		if len(html) < 100 {
			fmt.Printf("[siki] run_code HTML too short (%d), regenerating\n", len(html))
			if newHTML, err := generateCodeWithSubModel(userMsg, ws.config); err == nil {
				args["html"] = newHTML
			}
		}
	}

	// Generate image: ensure prompt exists, enhance if needed
	if toolName == "generate_image" {
		prompt, _ := args["prompt"].(string)
		if prompt == "" {
			// Auto-enhance user's message to English image prompt via sub-agent
			if ws.config.SubModel != "" || hasSubAgent(ws.config) {
				sendEvent(modelThinkingEvent("画像プロンプトを生成中...", ws.config, hasSubAgent(ws.config)))
				enhanceReq := fmt.Sprintf(`以下のユーザーリクエストから、画像生成AI用の英語プロンプトを生成せよ。
詳細で描写的な英語プロンプトのみを出力し、他の文章は書くな。
スタイル指定（digital art, infographic, illustration等）を含めること。

ユーザーリクエスト: %s`, userMsg)
				_, enhanced, err := callSubAgent(enhanceReq, ws.config)
				if err == nil && len(enhanced) > 10 {
					enhanced = strings.TrimSpace(enhanced)
					enhanced = strings.TrimPrefix(enhanced, "```")
					enhanced = strings.TrimSuffix(enhanced, "```")
					enhanced = strings.TrimSpace(enhanced)
					// Remove surrounding quotes if present
					if len(enhanced) > 2 && enhanced[0] == '"' && enhanced[len(enhanced)-1] == '"' {
						enhanced = enhanced[1 : len(enhanced)-1]
					}
					args["prompt"] = enhanced
					sendEvent(modelThinkingEvent(fmt.Sprintf("Prompt: %s", enhanced), ws.config, hasSubAgent(ws.config)))
				} else {
					args["prompt"] = userMsg
				}
			} else {
				args["prompt"] = userMsg
			}
		}
	}

	// Generate video: ensure prompt exists, enhance if needed
	if toolName == "generate_video" {
		prompt, _ := args["prompt"].(string)
		if prompt == "" {
			if ws.config.SubModel != "" || hasSubAgent(ws.config) {
				sendEvent(modelThinkingEvent("動画プロンプトを生成中...", ws.config, hasSubAgent(ws.config)))
				enhanceReq := fmt.Sprintf(`以下のユーザーリクエストから、動画生成AI用の英語プロンプトを生成せよ。
動きや場面の変化を含む詳細で描写的な英語プロンプトのみを出力し、他の文章は書くな。

ユーザーリクエスト: %s`, userMsg)
				_, enhanced, err := callSubAgent(enhanceReq, ws.config)
				if err == nil && len(enhanced) > 10 {
					enhanced = strings.TrimSpace(enhanced)
					enhanced = strings.TrimPrefix(enhanced, "```")
					enhanced = strings.TrimSuffix(enhanced, "```")
					enhanced = strings.TrimSpace(enhanced)
					if len(enhanced) > 2 && enhanced[0] == '"' && enhanced[len(enhanced)-1] == '"' {
						enhanced = enhanced[1 : len(enhanced)-1]
					}
					args["prompt"] = enhanced
					sendEvent(modelThinkingEvent(fmt.Sprintf("Video Prompt: %s", enhanced), ws.config, hasSubAgent(ws.config)))
				} else {
					args["prompt"] = userMsg
				}
			} else {
				args["prompt"] = userMsg
			}
		}
	}

	// Generate DOT code for diagram if not provided
	if toolName == "diagram" {
		dotCode, _ := args["dot_source"].(string)
		if dotCode == "" {
			dotCode, _ = args["dot_code"].(string)
		}
		if dotCode == "" || !strings.Contains(dotCode, "graph") {
			fmt.Printf("[siki] diagram: DOT code missing or invalid, generating via sub-model\n")
			sendEvent(modelThinkingEvent("図のDOTコードを生成中...", ws.config, false))
			prompt := fmt.Sprintf(`以下のリクエストに対して、Graphviz DOTコードのみ出力せよ。
説明不要。コードフェンス不要。digraphまたはgraphで始まるDOTコードだけを出力しろ。

例:
digraph G {
  A -> B -> C;
}

リクエスト: %s`, userMsg)
			_, genDot, err := callSubModel(prompt, ws.config)
			if err != nil {
				fmt.Printf("[siki] diagram: sub-model call failed: %v\n", err)
			} else {
				fmt.Printf("[siki] diagram: sub-model returned %d bytes\n", len(genDot))
				// Extract DOT from code fences if present
				dot := genDot
				if idx := strings.Index(dot, "```dot"); idx >= 0 {
					dot = dot[idx+6:]
					if end := strings.Index(dot, "```"); end >= 0 {
						dot = dot[:end]
					}
				} else if idx := strings.Index(dot, "```graphviz"); idx >= 0 {
					dot = dot[idx+11:]
					if end := strings.Index(dot, "```"); end >= 0 {
						dot = dot[:end]
					}
				} else if idx := strings.Index(dot, "```"); idx >= 0 {
					dot = dot[idx+3:]
					if nl := strings.Index(dot, "\n"); nl >= 0 {
						dot = dot[nl+1:]
					}
					if end := strings.Index(dot, "```"); end >= 0 {
						dot = dot[:end]
					}
				}
				dot = strings.TrimSpace(dot)
				if strings.Contains(dot, "digraph") || strings.Contains(dot, "graph") {
					args["dot_source"] = dot
					fmt.Printf("[siki] diagram: generated %d bytes of DOT code\n", len(dot))
				} else {
					// Last resort: try to find digraph/graph anywhere in the response
					if idx := strings.Index(genDot, "digraph"); idx >= 0 {
						dot = genDot[idx:]
						args["dot_source"] = dot
						fmt.Printf("[siki] diagram: extracted digraph from response (%d bytes)\n", len(dot))
					} else if idx := strings.Index(genDot, "graph"); idx >= 0 {
						dot = genDot[idx:]
						args["dot_source"] = dot
						fmt.Printf("[siki] diagram: extracted graph from response (%d bytes)\n", len(dot))
					} else {
						fmt.Printf("[siki] diagram: no valid DOT found in response, using default diagram\n")
						// Fallback: generate a default sample diagram
						defaultDot := `digraph G {
  rankdir=LR;
  node [shape=box, style=filled, fillcolor="#e8e8e8"];
  User [label="ユーザー", fillcolor="#e94560", fontcolor=white];
  AI [label="AI", fillcolor="#16213e", fontcolor=white];
  Tool [label="ツール"];
  Result [label="結果"];
  User -> AI [label="リクエスト"];
  AI -> Tool [label="実行"];
  Tool -> Result [label="出力"];
  Result -> AI [label="要約"];
  AI -> User [label="回答"];
}`
						args["dot_source"] = defaultDot
					}
				}
			}
		}
	}

	agent.sendEvent = sendEvent // allow tools to emit progress to UI
	result, err := agent.executeTool(toolName, args)
	if err != nil {
		errStr := fmt.Sprintf("Error: %v", err)
		fmt.Printf("[siki] Tool %s failed: %v — attempting recovery with alternate model\n", toolName, err)
		sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: errStr})
		sendEvent(modelThinkingEvent("ツール実行エラー。別モデルで再検討中...", ws.config, hasSubAgent(ws.config)))

		// Ask alternate LLM to re-analyze and suggest a fix
		altModel := pickRetryModel(ws.config, 1)
		argsJSONForPrompt, _ := json.Marshal(args)
		recoveryPrompt := fmt.Sprintf(`ツール「%s」の実行でエラーが発生した。

## ユーザーリクエスト:
%s

## 実行したツール: %s
## 引数: %s
## エラー:
%s

以下のいずれかを提案せよ：
1. 引数を修正して同じツールを再実行（JSON形式: {"action":"retry","tool":"%s","args":{...}}）
2. 別のツールで代替（JSON形式: {"action":"switch","tool":"別ツール名","args":{...}}）
3. ツールなしで直接回答（JSON形式: {"action":"direct","response":"回答テキスト"}）

JSONのみ出力。説明不要。`, toolName, userMsg, toolName, string(argsJSONForPrompt), errStr, toolName)

		_, recovery, recErr := callSubModelWith(recoveryPrompt, ws.config, altModel)
		recovered := false
		if recErr == nil && len(recovery) > 5 {
			recovery = strings.TrimSpace(recovery)
			if idx := strings.Index(recovery, "```json"); idx >= 0 {
				recovery = recovery[idx+7:]
				if end := strings.Index(recovery, "```"); end >= 0 { recovery = recovery[:end] }
			} else if idx := strings.Index(recovery, "```"); idx >= 0 {
				recovery = recovery[idx+3:]
				if end := strings.Index(recovery, "```"); end >= 0 { recovery = recovery[:end] }
			}
			recovery = strings.TrimSpace(recovery)

			var plan struct {
				Action   string                 `json:"action"`
				Tool     string                 `json:"tool"`
				Args     map[string]interface{} `json:"args"`
				Response string                 `json:"response"`
			}
			if json.Unmarshal([]byte(recovery), &plan) == nil {
				switch plan.Action {
				case "retry":
					fmt.Printf("[siki] Recovery: retrying %s with fixed args (via %s)\n", toolName, altModel)
					if plan.Args != nil {
						for k, v := range plan.Args { args[k] = v }
					}
					sendEvent(modelThinkingEvent("修正した引数で再実行中...", ws.config, false))
					result, err = agent.executeTool(toolName, args)
					if err == nil { recovered = true } else {
						result = fmt.Sprintf("Error (retry): %v", err)
					}
				case "switch":
					if plan.Tool != "" && plan.Tool != toolName {
						fmt.Printf("[siki] Recovery: switching %s → %s (via %s)\n", toolName, plan.Tool, altModel)
						sendEvent(modelThinkingEvent(fmt.Sprintf("別ツール %s で再試行中...", plan.Tool), ws.config, false))
						newArgs := plan.Args
						if newArgs == nil { newArgs = args }
						toolName = plan.Tool
						result, err = agent.executeTool(toolName, newArgs)
						if err == nil { recovered = true; args = newArgs } else {
							result = fmt.Sprintf("Error (switch): %v", err)
						}
					}
				case "direct":
					if plan.Response != "" {
						fmt.Printf("[siki] Recovery: direct response from %s\n", altModel)
						sendEvent(StreamEvent{Type: "content", Content: plan.Response})
						finalMsg := Message{Role: "assistant", Content: plan.Response}
						agent.messages = append(agent.messages, finalMsg)
						saveMsg(finalMsg, "")
						return plan.Response
					}
				}
			}
		}
		if !recovered {
			result = errStr
		}
	}

	displayResult := result
	// Don't truncate twitter results — they are already LLM-filtered
	if len(displayResult) > 2000 && toolName != "twitter_timeline" && toolName != "twitter_search" {
		displayResult = displayResult[:2000] + "\n... (truncated)"
	}
	sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: displayResult})

	// Save tool call to conversation history
	toolCallID := fmt.Sprintf("orch-%d", time.Now().UnixMilli())
	argsJSON, _ := json.Marshal(args)
	assistantMsg := Message{
		Role: "assistant",
		ToolCalls: []ToolCall{{
			ID:   toolCallID,
			Type: "function",
			Function: struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}{Name: toolName, Arguments: string(argsJSON)},
		}},
	}
	agent.messages = append(agent.messages, assistantMsg)
	saveMsg(assistantMsg, "")

	toolMsg := Message{
		Role:       "tool",
		Content:    result,
		ToolCallID: toolCallID,
	}
	agent.messages = append(agent.messages, toolMsg)
	saveMsg(toolMsg, toolName)

	// For these tools, the tool result IS the response (no LLM re-summarization needed)
	if toolName == "run_code" || toolName == "diagram" || toolName == "generate_image" || toolName == "twitter_timeline" || toolName == "twitter_search" {
		finalMsg := Message{Role: "assistant", Content: result}
		agent.messages = append(agent.messages, finalMsg)
		saveMsg(finalMsg, "")
		if cID != "" {
			ws.mu.Lock()
			ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: toolName, Args: args, ToolResult: result, Response: result, UsedAgent: false}
			ws.mu.Unlock()
		}
		return result
	}

	// For web_search, fetch top pages in parallel to get actual content
	if toolName == "web_search" {
		urls := extractURLsFromSearchResults(result)
		if len(urls) > 5 {
			urls = urls[:5]
		}
		if len(urls) > 0 {
			sendEvent(StreamEvent{Type: "thinking", Content: fmt.Sprintf("上位%d件のページを取得中...", len(urls))})
			type fetchResult struct {
				url     string
				content string
			}
			ch := make(chan fetchResult, len(urls))
			for _, u := range urls {
				go func(fetchURL string) {
					pageContent, err := agent.webFetchQuick(fetchURL)
					if err != nil {
						fmt.Printf("[siki] Failed to fetch %s: %v\n", fetchURL, err)
						ch <- fetchResult{fetchURL, ""}
						return
					}
					if len(pageContent) > 3000 {
						pageContent = pageContent[:3000] + "\n...(省略)"
					}
					ch <- fetchResult{fetchURL, pageContent}
				}(u)
			}
			var fetchedContent strings.Builder
			fetchedContent.WriteString(result)
			fetchedContent.WriteString("\n\n--- 以下はページ本文 ---\n")
			// Collect results with keepalive heartbeat
			ticker := time.NewTicker(2 * time.Second)
			collected := 0
			for collected < len(urls) {
				select {
				case fr := <-ch:
					collected++
					sendEvent(StreamEvent{Type: "thinking", Content: fmt.Sprintf("ページ取得完了 (%d/%d)", collected, len(urls))})
					if fr.content != "" {
						fetchedContent.WriteString(fmt.Sprintf("\n## %s\n%s\n", fr.url, fr.content))
					}
				case <-ticker.C:
					sendEvent(StreamEvent{Type: "thinking", Content: "ページ取得中..."})
				}
			}
			ticker.Stop()
			result = fetchedContent.String()
		}
	}

	// Phase 3: gpt-oss summarizes tool results (streaming)
	sendEvent(modelThinkingEvent("回答を生成中...", ws.config, hasSubAgent(ws.config)))

	finalResponse, err := streamSubModelSummarize(userMsg, toolName, result, ws.config, sendEvent)
	if err != nil {
		fmt.Printf("[siki] Summarization failed: %v\n", err)
		finalResponse = displayResult
		sendEvent(StreamEvent{Type: "content", Content: finalResponse})
	}

	// Phase 4: Validate the response (programmatic, fast)
	isValid, reason, cleaned := validateResponse(userMsg, finalResponse, result, ws.config)

	if !isValid {
		// Validation failed — notify user and regenerate
		sendEvent(StreamEvent{Type: "content", Content: fmt.Sprintf("\n\n---\n**検証不合格: %s**\n回答を再生成します...\n\n", reason)})
		fmt.Printf("[siki] Validation failed: %s — regenerating\n", reason)

		retryResponse, retryErr := streamSubModelSummarize(
			userMsg, toolName,
			result+"\n\n## 重要な注意（前回の回答に問題があった）:\n"+reason+"\nこの問題を修正して回答せよ。URLはツール結果のものだけを使え。",
			ws.config, sendEvent,
		)
		if retryErr == nil && len(strings.TrimSpace(retryResponse)) > 20 {
			finalResponse = retryResponse
		} else {
			finalResponse = cleaned
		}
	} else if cleaned != finalResponse {
		finalResponse = cleaned
	}

	// Send follow-up suggestions
	suggestions := generateSuggestions(userMsg, finalResponse, toolName)
	sendEvent(StreamEvent{Type: "suggestions", Suggestions: suggestions})

	// Final fallback: if response is still too short
	if len(strings.TrimSpace(finalResponse)) < 20 && len(result) > 100 {
		fmt.Printf("[siki] Response too short (%d chars), falling back to raw result\n", len(finalResponse))
		finalResponse = displayResult
		sendEvent(StreamEvent{Type: "content", Content: "\n\n" + finalResponse})
	}

	finalMsg := Message{Role: "assistant", Content: finalResponse}
	agent.messages = append(agent.messages, finalMsg)
	saveMsg(finalMsg, "")

	if cID != "" {
		ws.mu.Lock()
		ws.lastExec[cID] = &LastToolExecution{UserMsg: userMsg, ToolName: toolName, Args: args, ToolResult: result, Response: finalResponse, UsedAgent: hasSubAgent(ws.config)}
		ws.mu.Unlock()
	}

	return finalResponse
}

// preWarmSubModel sends a tiny prompt to gpt-oss at startup to load it into memory.
func preWarmSubModel(config *Config) {
	if config.SubModel == "" {
		return
	}
	go func() {
		fmt.Printf("[siki] Pre-warming sub-model: %s ...\n", config.SubModel)
		start := time.Now()
		_, err := callOllamaGenerate(config.SubModel, "Say OK.", 5, 600*time.Second, config)
		if err != nil {
			fmt.Printf("[siki] Sub-model warm-up failed: %v\n", err)
		} else {
			fmt.Printf("[siki] Sub-model ready (%.1fs)\n", time.Since(start).Seconds())
		}
		// Also pre-warm orchestrator if different from sub-model
		orchModel := config.orchestratorModel()
		if orchModel != "" && orchModel != config.SubModel {
			fmt.Printf("[siki] Pre-warming orchestrator: %s ...\n", orchModel)
			start2 := time.Now()
			_, err2 := callOllamaGenerate(orchModel, "Say OK.", 5, 600*time.Second, config)
			if err2 != nil {
				fmt.Printf("[siki] Orchestrator warm-up failed: %v\n", err2)
			} else {
				fmt.Printf("[siki] Orchestrator ready (%.1fs)\n", time.Since(start2).Seconds())
			}
		}
	}()
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
	// Client-side rendering using d3-graphviz (no server-side Graphviz needed)
	// Escape the DOT code for embedding in HTML
	escapedDot := strings.ReplaceAll(dotCode, "`", "\\`")
	escapedDot = strings.ReplaceAll(escapedDot, "${", "\\${")
	escapedTitle := strings.ReplaceAll(title, `"`, "&quot;")

	// Return an HTML snippet with d3-graphviz that renders in an iframe
	html := fmt.Sprintf(`<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>%s</title>
<script src="/js/d3.v7.min.js"></script>
<script src="/js/graphviz.umd.js"></script>
<script src="/js/d3-graphviz.min.js"></script>
<style>
body { margin: 0; padding: 16px; background: #1a1a2e; color: #eee; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }
h3 { margin: 0 0 12px 0; color: #e94560; }
#graph { background: #16213e; border-radius: 8px; padding: 16px; width: 95vw; min-height: 200px; overflow: auto; }
#graph svg { max-width: 100%%; height: auto; }
#graph svg text { fill: #eee !important; }
#graph svg polygon[fill="white"] { fill: #16213e !important; }
#graph svg polygon[stroke="black"] { stroke: #444 !important; }
#graph svg path[stroke="black"] { stroke: #888 !important; }
#graph svg ellipse[stroke="black"] { stroke: #e94560 !important; }
#graph svg ellipse[fill="none"] { fill: #1a1a3e !important; }
#graph svg polygon[fill="none"] { fill: #1a1a3e !important; }
.error { color: #ff6b6b; padding: 12px; background: #2d1b1b; border-radius: 8px; }
</style>
</head><body>
<h3>%s</h3>
<div id="graph"></div>
<script>
try {
  d3.select("#graph").graphviz()
    .fit(true)
    .zoom(false)
    .renderDot(`+"`%s`"+`);
} catch(e) {
  document.getElementById("graph").innerHTML = '<div class="error">Rendering error: ' + e.message + '</div><pre>' + `+"`%s`"+` + '</pre>';
}
</script>
</body></html>`, escapedTitle, escapedTitle, escapedDot, escapedDot)

	// Save as playground file for iframe rendering
	if err := initPlaygroundDir(); err != nil {
		return "", fmt.Errorf("failed to create playground dir: %w", err)
	}
	filename := fmt.Sprintf("diagram_%d.html", time.Now().UnixNano())
	outputPath := filepath.Join(playgroundDir, filename)
	if err := os.WriteFile(outputPath, []byte(html), 0644); err != nil {
		return "", fmt.Errorf("failed to write diagram file: %w", err)
	}

	playgroundURL := fmt.Sprintf("/playground/%s", filename)
	iframeHTML := fmt.Sprintf(`<iframe src="%s" style="width:100%%;height:500px;border:none;border-radius:8px;" sandbox="allow-scripts allow-same-origin"></iframe>`, playgroundURL)
	fmt.Printf("[siki] Diagram rendered client-side via d3-graphviz (%d bytes DOT)\n", len(dotCode))
	return iframeHTML, nil
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

// ---- User Profile System ----

// UserProfile stores user interests and preferences learned from conversations.
type UserProfile struct {
	// AI推定フィールド
	Interests     []string  `json:"interests"`
	Occupation    string    `json:"occupation"`
	TechLevel     string    `json:"tech_level"`
	Preferences   []string  `json:"preferences"`
	FrequentTools []string  `json:"frequent_tools"`
	LastUpdated   time.Time `json:"last_updated"`
	// ユーザー申告フィールド
	Name       string   `json:"name,omitempty"`
	Age        int      `json:"age,omitempty"`
	Company    string   `json:"company,omitempty"`
	Department string   `json:"department,omitempty"`
	Role       string   `json:"role,omitempty"`
	Clients    []string `json:"clients,omitempty"`
	Skills     []string `json:"skills,omitempty"`
	Location   string   `json:"location,omitempty"`
	Bio        string   `json:"bio,omitempty"`
	// 質問管理
	AskedFields []string  `json:"asked_fields,omitempty"`
	LastAsked   time.Time `json:"last_asked,omitempty"`
}

func userProfilePath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".siki", "user_profile.json")
}

func loadUserProfile() *UserProfile {
	data, err := os.ReadFile(userProfilePath())
	if err != nil {
		return nil
	}
	var p UserProfile
	if err := json.Unmarshal(data, &p); err != nil {
		return nil
	}
	// 読み込み時にも肥大化を防止
	trimmed := false
	if len(p.Interests) > 30 {
		p.Interests = p.Interests[len(p.Interests)-30:]
		trimmed = true
	}
	if len(p.Preferences) > 20 {
		p.Preferences = p.Preferences[len(p.Preferences)-20:]
		trimmed = true
	}
	if len(p.FrequentTools) > 15 {
		p.FrequentTools = p.FrequentTools[len(p.FrequentTools)-15:]
		trimmed = true
	}
	if trimmed {
		saveUserProfile(&p)
	}
	return &p
}

func saveUserProfile(p *UserProfile) error {
	p.LastUpdated = time.Now()
	// トリミング: 肥大化防止
	if len(p.Interests) > 30 {
		p.Interests = p.Interests[len(p.Interests)-30:]
	}
	if len(p.Preferences) > 20 {
		p.Preferences = p.Preferences[len(p.Preferences)-20:]
	}
	if len(p.FrequentTools) > 15 {
		p.FrequentTools = p.FrequentTools[len(p.FrequentTools)-15:]
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(userProfilePath())
	os.MkdirAll(dir, 0755)
	return os.WriteFile(userProfilePath(), data, 0644)
}

// mergeProfile merges new profile data into existing, adding new items without duplicates.
func mergeProfile(existing, incoming *UserProfile) *UserProfile {
	if existing == nil {
		return incoming
	}
	if incoming == nil {
		return existing
	}
	addUnique := func(base, additions []string) []string {
		seen := make(map[string]bool)
		for _, s := range base {
			seen[s] = true
		}
		result := append([]string{}, base...)
		for _, s := range additions {
			if !seen[s] && s != "" {
				result = append(result, s)
				seen[s] = true
			}
		}
		return result
	}
	existing.Interests = addUnique(existing.Interests, incoming.Interests)
	existing.Preferences = addUnique(existing.Preferences, incoming.Preferences)
	existing.FrequentTools = addUnique(existing.FrequentTools, incoming.FrequentTools)
	if incoming.Occupation != "" {
		existing.Occupation = incoming.Occupation
	}
	if incoming.TechLevel != "" {
		existing.TechLevel = incoming.TechLevel
	}
	// 新フィールド: AI推定で判明した場合のみ上書き（ユーザー手動入力を優先）
	if incoming.Name != "" && existing.Name == "" {
		existing.Name = incoming.Name
	}
	if incoming.Age != 0 && existing.Age == 0 {
		existing.Age = incoming.Age
	}
	if incoming.Company != "" && existing.Company == "" {
		existing.Company = incoming.Company
	}
	if incoming.Role != "" && existing.Role == "" {
		existing.Role = incoming.Role
	}
	if incoming.Location != "" && existing.Location == "" {
		existing.Location = incoming.Location
	}
	existing.Clients = addUnique(existing.Clients, incoming.Clients)
	return existing
}

func (ws *WebServer) updateUserProfile() {
	threads, err := listThreads()
	if err != nil || len(threads) == 0 {
		return
	}

	// Collect user messages from last 24h
	cutoff := time.Now().Add(-24 * time.Hour)
	var userMsgs strings.Builder
	msgCount := 0
	for _, t := range threads {
		if !t.UpdatedAt.After(cutoff) {
			continue
		}
		thread, err := loadThread(t.ID)
		if err != nil {
			continue
		}
		for _, m := range thread.Messages {
			if m.Role == "user" && m.Content != "" {
				userMsgs.WriteString("- " + truncateString(m.Content, 200) + "\n")
				msgCount++
			}
		}
	}
	if msgCount < 3 {
		return // Not enough data
	}

	prompt := fmt.Sprintf(`以下のユーザー発言から、ユーザーのプロファイルを分析せよ。
JSON形式で出力:
{"interests":["興味1","興味2"],"occupation":"推定職業","tech_level":"beginner|intermediate|advanced|expert","preferences":["傾向1"],"frequent_tools":["よく使うツール"],"name":"判明した名前","age":0,"company":"判明した勤務先","role":"判明した役職","clients":["判明した取引先"],"location":"判明した所在地"}

不明なフィールドは空文字列や空配列、age不明なら0にせよ。

ユーザー発言:
%s

JSONのみ出力せよ。`, userMsgs.String())

	_, response, err := callSubModel(prompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] User profile analysis failed: %v\n", err)
		return
	}

	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start < 0 || end <= start {
		return
	}

	var incoming UserProfile
	if err := json.Unmarshal([]byte(response[start:end+1]), &incoming); err != nil {
		fmt.Printf("[siki] User profile parse failed: %v\n", err)
		return
	}

	existing := loadUserProfile()
	merged := mergeProfile(existing, &incoming)

	// 定期コンパクト化: interests/preferencesが閾値を超えたらLLMで要約統合
	if len(merged.Interests) > 25 || len(merged.Preferences) > 15 {
		compacted := compactProfile(merged, ws.config)
		if compacted != nil {
			merged = compacted
		}
	}

	if err := saveUserProfile(merged); err != nil {
		fmt.Printf("[siki] User profile save failed: %v\n", err)
		return
	}
	fmt.Printf("[siki] User profile updated: interests=%d, preferences=%d, occupation=%s\n", len(merged.Interests), len(merged.Preferences), merged.Occupation)
}

// compactProfile uses LLM to consolidate and deduplicate profile lists.
func compactProfile(p *UserProfile, config *Config) *UserProfile {
	prompt := fmt.Sprintf(`以下のユーザープロフィールのリストを整理・統合してください。
重複や類似項目を統合し、重要度の高い順に並べてください。

興味リスト（現在%d件 → 最大15件に要約）:
%s

傾向リスト（現在%d件 → 最大10件に要約）:
%s

ツールリスト（現在%d件 → 最大10件に要約）:
%s

JSON形式で出力:
{"interests":["要約済み興味1","要約済み興味2"],"preferences":["要約済み傾向1"],"frequent_tools":["ツール1"]}

JSONのみ出力せよ。`,
		len(p.Interests), strings.Join(p.Interests, ", "),
		len(p.Preferences), strings.Join(p.Preferences, ", "),
		len(p.FrequentTools), strings.Join(p.FrequentTools, ", "))

	_, response, err := callSubModel(prompt, config)
	if err != nil {
		fmt.Printf("[siki] Profile compaction failed: %v\n", err)
		return nil
	}

	start := strings.Index(response, "{")
	end := strings.LastIndex(response, "}")
	if start < 0 || end <= start {
		return nil
	}

	var compacted struct {
		Interests     []string `json:"interests"`
		Preferences   []string `json:"preferences"`
		FrequentTools []string `json:"frequent_tools"`
	}
	if err := json.Unmarshal([]byte(response[start:end+1]), &compacted); err != nil {
		fmt.Printf("[siki] Profile compaction parse failed: %v\n", err)
		return nil
	}

	if len(compacted.Interests) > 0 {
		p.Interests = compacted.Interests
	}
	if len(compacted.Preferences) > 0 {
		p.Preferences = compacted.Preferences
	}
	if len(compacted.FrequentTools) > 0 {
		p.FrequentTools = compacted.FrequentTools
	}
	fmt.Printf("[siki] Profile compacted: interests=%d, preferences=%d, tools=%d\n",
		len(p.Interests), len(p.Preferences), len(p.FrequentTools))
	return p
}

// profileMissingFields returns field names that are important but not yet filled.
func profileMissingFields(p *UserProfile) []string {
	var missing []string
	if p.Name == "" {
		missing = append(missing, "name")
	}
	if p.Occupation == "" {
		missing = append(missing, "occupation")
	}
	if p.Age == 0 {
		missing = append(missing, "age")
	}
	if p.Company == "" {
		missing = append(missing, "company")
	}
	if p.Role == "" {
		missing = append(missing, "role")
	}
	if p.Location == "" {
		missing = append(missing, "location")
	}
	return missing
}

// handleProfile handles GET/POST /api/profile
func (ws *WebServer) handleProfile(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		profile := loadUserProfile()
		if profile == nil {
			profile = &UserProfile{}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(profile)
	case "POST":
		var input struct {
			Name       string   `json:"name"`
			Age        int      `json:"age"`
			Company    string   `json:"company"`
			Department string   `json:"department"`
			Role       string   `json:"role"`
			Clients    []string `json:"clients"`
			Skills     []string `json:"skills"`
			Location   string   `json:"location"`
			Bio        string   `json:"bio"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		profile := loadUserProfile()
		if profile == nil {
			profile = &UserProfile{}
		}
		// 手動入力フィールドのみ上書き（AI推定フィールドは触らない）
		profile.Name = input.Name
		profile.Age = input.Age
		profile.Company = input.Company
		profile.Department = input.Department
		profile.Role = input.Role
		profile.Clients = input.Clients
		profile.Skills = input.Skills
		profile.Location = input.Location
		profile.Bio = input.Bio
		if err := saveUserProfile(profile); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(profile)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// ---- Email Digest System ----

// DigestConfig is the persistable subset of digest/email settings.
type DigestConfig struct {
	EmailTo            string `json:"email_to"`
	EmailFrom          string `json:"email_from"`
	SMTPHost           string `json:"smtp_host"`
	SMTPPort           int    `json:"smtp_port"`
	SMTPUser           string `json:"smtp_user"`
	SMTPPass           string `json:"smtp_pass"`
	DigestEnabled      bool   `json:"digest_enabled"`
	DigestHours        []int  `json:"digest_hours"`
	TwitterBearerToken    string `json:"twitter_bearer_token"`
	TwitterEnabled        bool   `json:"twitter_enabled"`
	TwitterConsumerKey    string `json:"twitter_consumer_key"`
	TwitterConsumerSecret string `json:"twitter_consumer_secret"`
	TwitterAccessToken    string `json:"twitter_access_token"`
	TwitterAccessSecret   string `json:"twitter_access_secret"`
	// Bluesky
	BlueskyEnabled      bool     `json:"bluesky_enabled"`
	BlueskyStarterPacks []string `json:"bluesky_starter_packs"`
	BlueskyIdentifier   string   `json:"bluesky_identifier"`
	BlueskyAppPassword  string   `json:"bluesky_app_password"`
	JetstreamKeywords   []string `json:"jetstream_keywords"`
	// External skill API keys
	BraveAPIKey string `json:"brave_api_key"`
	GroqAPIKey  string `json:"groq_api_key"`
}

// digestConfigDir can be overridden in tests to avoid writing to the real ~/.siki/
var digestConfigDir string

func digestConfigPath() string {
	if digestConfigDir != "" {
		return filepath.Join(digestConfigDir, "digest_config.json")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".siki", "digest_config.json")
}

func loadDigestConfig() *DigestConfig {
	data, err := os.ReadFile(digestConfigPath())
	if err != nil {
		return nil
	}
	var dc DigestConfig
	if err := json.Unmarshal(data, &dc); err != nil {
		return nil
	}
	return &dc
}

func saveDigestConfig(dc *DigestConfig) error {
	data, err := json.MarshalIndent(dc, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(digestConfigPath())
	os.MkdirAll(dir, 0755)
	return os.WriteFile(digestConfigPath(), data, 0644)
}

// applyDigestConfig loads saved digest settings into the running config.
func applyDigestConfig(config *Config) {
	dc := loadDigestConfig()
	if dc == nil {
		return
	}
	config.EmailTo = dc.EmailTo
	config.EmailFrom = dc.EmailFrom
	config.SMTPHost = dc.SMTPHost
	config.SMTPPort = dc.SMTPPort
	config.SMTPUser = dc.SMTPUser
	config.SMTPPass = dc.SMTPPass
	config.DigestEnabled = dc.DigestEnabled
	config.DigestHours = dc.DigestHours
	config.TwitterBearerToken = dc.TwitterBearerToken
	config.TwitterEnabled = dc.TwitterEnabled
	config.TwitterConsumerKey = dc.TwitterConsumerKey
	config.TwitterConsumerSecret = dc.TwitterConsumerSecret
	config.TwitterAccessToken = dc.TwitterAccessToken
	config.TwitterAccessSecret = dc.TwitterAccessSecret
	config.BlueskyEnabled = dc.BlueskyEnabled
	config.BlueskyStarterPacks = dc.BlueskyStarterPacks
	if dc.BlueskyIdentifier != "" {
		config.BlueskyIdentifier = dc.BlueskyIdentifier
	}
	if dc.BlueskyAppPassword != "" {
		config.BlueskyAppPassword = dc.BlueskyAppPassword
	}
	if len(dc.JetstreamKeywords) > 0 {
		config.JetstreamKeywords = dc.JetstreamKeywords
	}
	if dc.BraveAPIKey != "" {
		config.BraveAPIKey = dc.BraveAPIKey
		os.Setenv("BRAVE_API_KEY", dc.BraveAPIKey)
	}
	if dc.GroqAPIKey != "" {
		config.GroqAPIKey = dc.GroqAPIKey
		os.Setenv("GROQ_API_KEY", dc.GroqAPIKey)
	}
}

func (ws *WebServer) digestLoop() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()
	var lastSentHour int = -1
	for range ticker.C {
		if !ws.config.DigestEnabled || ws.config.EmailTo == "" {
			continue
		}
		now := time.Now()
		hour := now.Hour()
		digestHours := ws.config.DigestHours
		if len(digestHours) == 0 {
			digestHours = []int{9, 18}
		}
		shouldSend := false
		for _, h := range digestHours {
			if hour == h && lastSentHour != hour {
				shouldSend = true
				break
			}
		}
		if !shouldSend {
			continue
		}
		lastSentHour = hour
		ws.sendDigestEmail()
	}
}

func (ws *WebServer) sendDigestEmail() {
	fmt.Println("[siki] Digest: starting...")

	// 1. Load user profile for interests
	profile := loadUserProfile()
	var interests []string
	if profile != nil && len(profile.Interests) > 0 {
		interests = profile.Interests
	} else {
		interests = []string{"AI", "テクノロジー", "プログラミング"}
	}
	fmt.Printf("[siki] Digest: interests=%v\n", interests)

	// 2. Generate search queries from interests (with current date for freshness)
	now := time.Now()
	dateStr := fmt.Sprintf("%d年%d月", now.Year(), int(now.Month()))
	fmt.Println("[siki] Digest: generating search queries...")
	queryPrompt := fmt.Sprintf(`以下のユーザーの興味分野に基づいて、%sの最新ニュース検索クエリを3個生成せよ。
各クエリには必ず「%s」を含めること。具体的で検索に適した形式にすること。

興味分野: %s

JSON配列のみ出力: ["クエリ1","クエリ2","クエリ3"]`, dateStr, dateStr, strings.Join(interests, ", "))

	_, queryResp, err := callSubModel(queryPrompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] Digest: query generation failed: %v\n", err)
		return
	}
	fmt.Printf("[siki] Digest: LLM response: %s\n", truncateString(queryResp, 200))

	start := strings.Index(queryResp, "[")
	end := strings.LastIndex(queryResp, "]")
	if start < 0 || end <= start {
		fmt.Printf("[siki] Digest: failed to parse queries from: %s\n", truncateString(queryResp, 200))
		return
	}
	var queries []string
	if err := json.Unmarshal([]byte(queryResp[start:end+1]), &queries); err != nil {
		fmt.Printf("[siki] Digest: query parse failed: %v\n", err)
		return
	}
	if len(queries) > 3 {
		queries = queries[:3]
	}
	fmt.Printf("[siki] Digest: queries=%v\n", queries)

	// 3. Execute web searches (with per-query timeout)
	tempAgent := &Agent{config: ws.config}
	var allResults strings.Builder
	for i, q := range queries {
		fmt.Printf("[siki] Digest: searching %d/%d: %s\n", i+1, len(queries), q)
		done := make(chan struct{})
		var result string
		var searchErr error
		go func() {
			result, searchErr = tempAgent.webSearch(q)
			close(done)
		}()
		select {
		case <-done:
			if searchErr != nil {
				fmt.Printf("[siki] Digest: search failed for '%s': %v\n", q, searchErr)
				continue
			}
			allResults.WriteString(fmt.Sprintf("## 検索: %s\n%s\n\n", q, truncateString(result, 2000)))
		case <-time.After(30 * time.Second):
			fmt.Printf("[siki] Digest: search timeout for '%s'\n", q)
		}
	}

	if allResults.Len() == 0 {
		fmt.Println("[siki] Digest: no search results, aborting")
		return
	}
	fmt.Printf("[siki] Digest: collected %d bytes of search results\n", allResults.Len())

	// 4. Freshness verification — filter out stale information
	fmt.Println("[siki] Digest: verifying freshness of results...")
	freshnessPrompt := fmt.Sprintf(`今日は%s%d日です。
以下の検索結果から、%s以降の新しい情報のみを抽出せよ。
2024年以前の古い情報、日付が不明で古そうな情報は全て除外せよ。

検索結果:
%s

新しい情報のみ残して、同じフォーマットで出力せよ。古い情報しかない場合は「なし」と出力せよ。`,
		dateStr, now.Day(), dateStr, truncateString(allResults.String(), 6000))

	_, freshResults, err := callSubModel(freshnessPrompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] Digest: freshness check failed: %v, using raw results\n", err)
		freshResults = allResults.String()
	}
	if strings.TrimSpace(freshResults) == "なし" || len(strings.TrimSpace(freshResults)) < 50 {
		fmt.Println("[siki] Digest: no fresh information found, aborting")
		return
	}
	fmt.Printf("[siki] Digest: fresh results: %d bytes\n", len(freshResults))

	// 4.5 Twitter timeline (if enabled)
	if ws.config.TwitterEnabled && ws.config.TwitterBearerToken != "" {
		fmt.Println("[siki] Digest: fetching Twitter timeline...")
		tweets, tErr := fetchTwitterTimeline(ws.config)
		if tErr != nil {
			fmt.Printf("[siki] Digest: Twitter fetch failed: %v\n", tErr)
		} else if len(tweets) > 0 {
			twitterHTML, sErr := filterAndSummarizeTwitter(tweets, ws.config)
			if sErr != nil {
				fmt.Printf("[siki] Digest: Twitter summary failed: %v\n", sErr)
			} else if twitterHTML != "" {
				freshResults += "\n\n## Twitterタイムライン（AI関連）:\n" + twitterHTML
				fmt.Printf("[siki] Digest: Twitter section added (%d bytes)\n", len(twitterHTML))
			} else {
				fmt.Println("[siki] Digest: no AI-related tweets found in timeline")
			}
		} else {
			fmt.Println("[siki] Digest: no tweets in timeline (last 12h)")
		}
	}

	// 4.6 Bluesky AI/MLフィード
	if ws.config.BlueskyEnabled {
		feed := loadBlueskyFeed()
		recentPosts := filterRecentBlueskyPosts(feed.Posts, 12*time.Hour)
		if len(recentPosts) > 0 {
			fmt.Printf("[siki] Digest: processing %d Bluesky posts...\n", len(recentPosts))
			bskyHTML := filterAndSummarizeBluesky(recentPosts, ws.config)
			if bskyHTML != "" {
				freshResults += "\n\n## Bluesky AI/MLフィード:\n" + bskyHTML
				fmt.Printf("[siki] Digest: Bluesky section added (%d bytes)\n", len(bskyHTML))
			}

			// Deep-dive high-engagement posts
			deepDive := blueskyDeepDivePosts(recentPosts, ws.config)
			if deepDive != "" {
				freshResults += "\n\n## Bluesky注目記事の要約:\n" + deepDive
				fmt.Printf("[siki] Digest: Bluesky deep-dive added (%d bytes)\n", len(deepDive))
			}
		} else {
			fmt.Println("[siki] Digest: no recent Bluesky posts (last 12h)")
		}
	}

	// 5. Summarize with LLM
	fmt.Println("[siki] Digest: summarizing results...")
	summaryPrompt := fmt.Sprintf(`今日は%s%d日です。以下の検索結果をもとに、ユーザーにとって有益な最新情報を選んでダイジェストにまとめよ。
古い情報（%d年以前）は絶対に含めるな。
HTMLメール本文として出力せよ（<html>タグ不要、本文のみ）。
見出し（<h2>）、段落（<p>）、リンク（<a href>）を使って読みやすくすること。
各項目に日付を明記すること。

検索結果:
%s

HTML本文のみ出力せよ。`, dateStr, now.Day(), now.Year(), truncateString(freshResults, 6000))

	_, htmlBody, err := callSubModel(summaryPrompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] Digest: summary failed: %v\n", err)
		return
	}
	fmt.Printf("[siki] Digest: summary generated (%d bytes)\n", len(htmlBody))

	// 6. Generate digest illustration image
	var digestImages []EmailImage
	if canRunImageServer() {
		fmt.Println("[siki] Digest: generating illustration image...")
		imgPrompt := fmt.Sprintf(`Generate a single English prompt for an illustration that visually represents today's tech news digest. The image should be a clean, modern infographic-style illustration. Topics: %s. Output only the English prompt, nothing else.`, truncateString(htmlBody, 500))
		_, imgPromptEn, imgErr := callSubModel(imgPrompt, ws.config)
		if imgErr == nil && len(imgPromptEn) > 10 {
			imgPromptEn = strings.TrimSpace(imgPromptEn)
			if len(imgPromptEn) > 300 {
				imgPromptEn = imgPromptEn[:300]
			}
			fmt.Printf("[siki] Digest: image prompt: %s\n", truncateString(imgPromptEn, 100))
			imgPath, imgErr2 := generateImage(imgPromptEn, 768, 512, ws.config)
			if imgErr2 == nil {
				fullPath := filepath.Join(".", imgPath) // imgPath is /playground/xxx.png
				if strings.HasPrefix(imgPath, "/playground/") {
					fullPath = filepath.Join(playgroundDir, filepath.Base(imgPath))
				}
				imgData, readErr := os.ReadFile(fullPath)
				if readErr == nil {
					digestImages = append(digestImages, EmailImage{
						CID:      "digest-illustration",
						Data:     imgData,
						MimeType: "image/png",
						Filename: filepath.Base(imgPath),
					})
					// Add image tag at the top of HTML body
					htmlBody = fmt.Sprintf(`<div style="text-align:center;margin-bottom:20px"><img src="cid:digest-illustration" style="max-width:100%%;border-radius:8px" alt="Digest illustration"></div>`) + htmlBody
					fmt.Println("[siki] Digest: illustration image attached")
				} else {
					fmt.Printf("[siki] Digest: failed to read image file: %v\n", readErr)
				}
			} else {
				fmt.Printf("[siki] Digest: image generation failed: %v\n", imgErr2)
			}
		} else {
			fmt.Printf("[siki] Digest: image prompt generation failed: %v\n", imgErr)
		}
	}

	// 7. Send email
	subject := fmt.Sprintf("siki ダイジェスト — %s", now.Format("2006/01/02 15:00"))
	fmt.Printf("[siki] Digest: sending email to %s...\n", ws.config.EmailTo)
	if err := sendEmailWithImages(ws.config, subject, htmlBody, digestImages); err != nil {
		fmt.Printf("[siki] Digest: email send failed: %v\n", err)
		return
	}
	fmt.Printf("[siki] Digest: email sent to %s\n", ws.config.EmailTo)

	// 8. Save digest as a thread
	digestThreadID := fmt.Sprintf("digest-%d", now.Unix())
	digestMsg := Message{Role: "assistant", Content: fmt.Sprintf("# %s\n\n%s", subject, htmlBody)}
	appendMessageToThread(digestThreadID, digestMsg, "")
	saveThreadMeta(&Thread{ID: digestThreadID, Title: subject, CreatedAt: now, UpdatedAt: now, MessageCount: 1})
}

// ============================================================================
// Twitter Timeline Integration (API v2)
// ============================================================================

// TwitterMedia represents a media attachment (photo, video, animated_gif).
type TwitterMedia struct {
	Type       string // "photo", "video", "animated_gif"
	URL        string // direct URL for photo, preview for video
	PreviewURL string // thumbnail
}

// TwitterTweet represents a single tweet from the timeline.
type TwitterTweet struct {
	ID        string `json:"id"`
	Text      string `json:"text"`
	CreatedAt string `json:"created_at"`
	AuthorID  string `json:"author_id"`
	Author          string // expanded from includes.users
	ProfileImageURL string // user's profile icon
	Media           []TwitterMedia
	URLs            []string // expanded URLs from entities
	ReplyCount      int    // from public_metrics
	RetweetCount    int    // from public_metrics
	LikeCount       int    // from public_metrics
}

// --- Bluesky types ---

type BlueskyPost struct {
	URI           string    `json:"uri"`
	CID           string    `json:"cid"`
	Text          string    `json:"text"`
	CreatedAt     string    `json:"created_at"`
	AuthorHandle  string    `json:"author_handle"`
	AuthorName    string    `json:"author_name"`
	AvatarURL     string    `json:"avatar_url"`
	ReplyCount    int       `json:"reply_count"`
	RepostCount   int       `json:"repost_count"`
	LikeCount     int       `json:"like_count"`
	QuoteCount    int       `json:"quote_count"`
	ExternalURL   string    `json:"external_url,omitempty"`
	ExternalTitle string    `json:"external_title,omitempty"`
	ImageURLs     []string  `json:"image_urls,omitempty"`
	FetchedAt     time.Time `json:"fetched_at"`
}

func (p BlueskyPost) EngagementScore() int {
	return p.LikeCount + p.RepostCount*3 + p.ReplyCount*2 + p.QuoteCount*2
}

type BlueskyFeed struct {
	Handles        []string      `json:"handles"`
	HandlesFetched time.Time     `json:"handles_fetched"`
	Posts          []BlueskyPost `json:"posts"`
	LastFetched    time.Time     `json:"last_fetched"`
}

// Default AI/ML starter packs from blueskystarterpack.com/aiml
var defaultBlueskyAIStarterPacks = []string{
	"at://did:plc:bmkptaqvfcwmgom75fmo5oo6/app.bsky.graph.starterpack/3laeesmwbi62l", // Engineers, Developers & Tech People
	"at://did:plc:oqqpxzlqy7m7z2zqps3rjrts/app.bsky.graph.starterpack/3m2mrpgmdql2d", // Plant modelling scientists II
	"at://did:plc:oqqpxzlqy7m7z2zqps3rjrts/app.bsky.graph.starterpack/3lbdjhpwhwa23", // Plant modelling scientists I
	"at://did:plc:z3lil7hj3jloch4r3owljui5/app.bsky.graph.starterpack/3lcpvfls27723", // MLSupple
	"at://did:plc:q7tjqlj55q5j54aoipdc6r7i/app.bsky.graph.starterpack/3lawbuqz4yv27", // Paul van der Meer
	"at://did:plc:5ldqhnk4quyil4mie2yjg2po/app.bsky.graph.starterpack/3llhdncyla32y", // HABSFRANK STARTER PACK 5
	"at://did:plc:mptsx33lhqsobeooj5k23cqh/app.bsky.graph.starterpack/3lnsym7j76w2f", // Tom Wittig's Tech Stack
	"at://did:plc:vtpyqvwce4x6gpa5dcizqecy/app.bsky.graph.starterpack/3lbuyxxgv432y", // TechCrunch
	"at://did:plc:al62dnktcv4nwprgml2ryfnz/app.bsky.graph.starterpack/3lbhmp4cwl72w", // Red Hat
	"at://did:plc:7xmdqtvxy43625hisyy4wksb/app.bsky.graph.starterpack/3lgxxttp4u722", // Canadian AI Researchers
}

// --- Bluesky API functions (public, no auth required) ---

const blueskyPublicAPI = "https://public.api.bsky.app/xrpc/"

// blueskyAPIGet performs a GET request to the Bluesky public API.
func blueskyAPIGet(endpoint string) ([]byte, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", blueskyPublicAPI+endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("bluesky API %s: status %d: %s", endpoint, resp.StatusCode, string(body[:min(len(body), 200)]))
	}
	return body, nil
}

// fetchBlueskyStarterPackMembers fetches all member handles from a starter pack.
func fetchBlueskyStarterPackMembers(starterPackURI string) ([]string, error) {
	// Step 1: Get starter pack details to find the list URI
	data, err := blueskyAPIGet("app.bsky.graph.getStarterPack?starterPack=" + url.QueryEscape(starterPackURI))
	if err != nil {
		return nil, fmt.Errorf("getStarterPack failed: %w", err)
	}

	var spResp struct {
		StarterPack struct {
			List struct {
				URI string `json:"uri"`
			} `json:"list"`
		} `json:"starterPack"`
	}
	if err := json.Unmarshal(data, &spResp); err != nil {
		return nil, fmt.Errorf("parse starterPack response: %w", err)
	}
	listURI := spResp.StarterPack.List.URI
	if listURI == "" {
		return nil, fmt.Errorf("no list URI in starter pack %s", starterPackURI)
	}

	// Step 2: Get all members from the list (with pagination)
	var handles []string
	cursor := ""
	for {
		ep := "app.bsky.graph.getList?list=" + url.QueryEscape(listURI) + "&limit=100"
		if cursor != "" {
			ep += "&cursor=" + url.QueryEscape(cursor)
		}
		data, err := blueskyAPIGet(ep)
		if err != nil {
			return handles, fmt.Errorf("getList failed: %w", err)
		}
		var listResp struct {
			Items []struct {
				Subject struct {
					Handle string `json:"handle"`
				} `json:"subject"`
			} `json:"items"`
			Cursor string `json:"cursor"`
		}
		if err := json.Unmarshal(data, &listResp); err != nil {
			return handles, fmt.Errorf("parse getList: %w", err)
		}
		for _, item := range listResp.Items {
			if item.Subject.Handle != "" {
				handles = append(handles, item.Subject.Handle)
			}
		}
		if listResp.Cursor == "" || len(listResp.Items) == 0 {
			break
		}
		cursor = listResp.Cursor
	}
	return handles, nil
}

// blueskyFeedPath returns the path to the Bluesky feed cache file.
func blueskyFeedPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".siki", "bluesky_feed.json")
}

// loadBlueskyFeed loads the cached Bluesky feed from disk.
func loadBlueskyFeed() *BlueskyFeed {
	data, err := os.ReadFile(blueskyFeedPath())
	if err != nil {
		return &BlueskyFeed{}
	}
	var feed BlueskyFeed
	if err := json.Unmarshal(data, &feed); err != nil {
		return &BlueskyFeed{}
	}
	return &feed
}

// saveBlueskyFeed saves the Bluesky feed to disk.
func saveBlueskyFeed(feed *BlueskyFeed) error {
	data, err := json.MarshalIndent(feed, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(blueskyFeedPath())
	os.MkdirAll(dir, 0755)
	return os.WriteFile(blueskyFeedPath(), data, 0644)
}

// mergeBlueskyPosts merges new posts into existing, deduplicating by URI, updating metrics, pruning old.
func mergeBlueskyPosts(existing, newPosts []BlueskyPost) []BlueskyPost {
	byURI := make(map[string]BlueskyPost)
	for _, p := range existing {
		byURI[p.URI] = p
	}
	for _, p := range newPosts {
		byURI[p.URI] = p // newer metrics overwrite
	}
	cutoff := time.Now().Add(-7 * 24 * time.Hour)
	var result []BlueskyPost
	for _, p := range byURI {
		t, _ := time.Parse(time.RFC3339, p.CreatedAt)
		if !t.IsZero() && t.Before(cutoff) {
			continue
		}
		result = append(result, p)
	}
	// Sort by created_at descending
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt > result[j].CreatedAt
	})
	if len(result) > 2000 {
		result = result[:2000]
	}
	return result
}

// resolveBlueskyHandles gets all unique handles from starter packs, with 24h cache.
func resolveBlueskyHandles(config *Config) ([]string, error) {
	feed := loadBlueskyFeed()
	if len(feed.Handles) > 0 && time.Since(feed.HandlesFetched) < 24*time.Hour {
		return feed.Handles, nil
	}

	packs := config.BlueskyStarterPacks
	if len(packs) == 0 {
		packs = defaultBlueskyAIStarterPacks
	}

	seen := make(map[string]bool)
	var allHandles []string
	for _, packURI := range packs {
		handles, err := fetchBlueskyStarterPackMembers(packURI)
		if err != nil {
			fmt.Printf("[siki] Bluesky: starter pack %s failed: %v\n", packURI, err)
			continue
		}
		for _, h := range handles {
			if !seen[h] {
				seen[h] = true
				allHandles = append(allHandles, h)
			}
		}
		time.Sleep(200 * time.Millisecond) // rate limit courtesy
	}

	if len(allHandles) > 0 {
		feed.Handles = allHandles
		feed.HandlesFetched = time.Now()
		saveBlueskyFeed(feed)
		fmt.Printf("[siki] Bluesky: resolved %d unique handles from %d starter packs\n", len(allHandles), len(packs))
	}
	return allHandles, nil
}

// fetchBlueskyAuthorFeed fetches recent posts from a single Bluesky user.
func fetchBlueskyAuthorFeed(handle string, limit int) ([]BlueskyPost, error) {
	if limit <= 0 {
		limit = 50
	}
	ep := fmt.Sprintf("app.bsky.feed.getAuthorFeed?actor=%s&limit=%d&filter=posts_no_replies", url.QueryEscape(handle), limit)
	data, err := blueskyAPIGet(ep)
	if err != nil {
		return nil, err
	}

	var feedResp struct {
		Feed []struct {
			Post struct {
				URI    string `json:"uri"`
				CID    string `json:"cid"`
				Author struct {
					Handle    string `json:"handle"`
					DisplayName string `json:"displayName"`
					Avatar    string `json:"avatar"`
				} `json:"author"`
				Record json.RawMessage `json:"record"`
				ReplyCount  int `json:"replyCount"`
				RepostCount int `json:"repostCount"`
				LikeCount   int `json:"likeCount"`
				QuoteCount  int `json:"quoteCount"`
				Embed json.RawMessage `json:"embed"`
			} `json:"post"`
		} `json:"feed"`
	}
	if err := json.Unmarshal(data, &feedResp); err != nil {
		return nil, fmt.Errorf("parse author feed: %w", err)
	}

	var posts []BlueskyPost
	for _, item := range feedResp.Feed {
		p := BlueskyPost{
			URI:          item.Post.URI,
			CID:          item.Post.CID,
			AuthorHandle: item.Post.Author.Handle,
			AuthorName:   item.Post.Author.DisplayName,
			AvatarURL:    item.Post.Author.Avatar,
			ReplyCount:   item.Post.ReplyCount,
			RepostCount:  item.Post.RepostCount,
			LikeCount:    item.Post.LikeCount,
			QuoteCount:   item.Post.QuoteCount,
			FetchedAt:    time.Now(),
		}

		// Parse record for text and createdAt
		var record struct {
			Text      string `json:"text"`
			CreatedAt string `json:"createdAt"`
			Embed     *struct {
				Type     string `json:"$type"`
				External *struct {
					URI   string `json:"uri"`
					Title string `json:"title"`
				} `json:"external"`
			} `json:"embed"`
		}
		if err := json.Unmarshal(item.Post.Record, &record); err == nil {
			p.Text = record.Text
			p.CreatedAt = record.CreatedAt
			if record.Embed != nil && record.Embed.External != nil {
				p.ExternalURL = record.Embed.External.URI
				p.ExternalTitle = record.Embed.External.Title
			}
		}

		// Parse embed for images and external links at post level
		if len(item.Post.Embed) > 0 {
			var embed struct {
				Type     string `json:"$type"`
				Images   []struct {
					Fullsize string `json:"fullsize"`
					Thumb    string `json:"thumb"`
				} `json:"images"`
				External *struct {
					URI   string `json:"uri"`
					Title string `json:"title"`
				} `json:"external"`
			}
			if err := json.Unmarshal(item.Post.Embed, &embed); err == nil {
				for _, img := range embed.Images {
					if img.Fullsize != "" {
						p.ImageURLs = append(p.ImageURLs, img.Fullsize)
					} else if img.Thumb != "" {
						p.ImageURLs = append(p.ImageURLs, img.Thumb)
					}
				}
				if p.ExternalURL == "" && embed.External != nil {
					p.ExternalURL = embed.External.URI
					p.ExternalTitle = embed.External.Title
				}
			}
		}

		posts = append(posts, p)
	}
	return posts, nil
}

// blueskySession caches the auth session token.
var blueskySession struct {
	AccessJwt string
	ExpiresAt time.Time
}

// blueskyCreateSession authenticates with Bluesky and returns an access JWT.
func blueskyCreateSession(config *Config) (string, error) {
	if blueskySession.AccessJwt != "" && time.Now().Before(blueskySession.ExpiresAt) {
		return blueskySession.AccessJwt, nil
	}
	if config.BlueskyIdentifier == "" || config.BlueskyAppPassword == "" {
		return "", fmt.Errorf("Bluesky認証情報が未設定です（設定画面でIdentifierとApp Passwordを入力してください）")
	}
	body, _ := json.Marshal(map[string]string{
		"identifier": config.BlueskyIdentifier,
		"password":   config.BlueskyAppPassword,
	})
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Post("https://bsky.social/xrpc/com.atproto.server.createSession", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("bluesky auth failed: %d %s", resp.StatusCode, string(respBody[:min(len(respBody), 200)]))
	}
	var session struct {
		AccessJwt string `json:"accessJwt"`
	}
	if err := json.Unmarshal(respBody, &session); err != nil {
		return "", err
	}
	blueskySession.AccessJwt = session.AccessJwt
	blueskySession.ExpiresAt = time.Now().Add(90 * time.Minute) // JWT typically lasts ~2h
	return session.AccessJwt, nil
}

// blueskyAuthAPIGet makes an authenticated GET request to the Bluesky API.
func blueskyAuthAPIGet(endpoint string, config *Config) ([]byte, error) {
	jwt, err := blueskyCreateSession(config)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", "https://bsky.social/xrpc/"+endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+jwt)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == 401 {
		// Token expired, retry once
		blueskySession.AccessJwt = ""
		jwt, err = blueskyCreateSession(config)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+jwt)
		resp2, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp2.Body.Close()
		respBody, _ = io.ReadAll(resp2.Body)
		if resp2.StatusCode != 200 {
			return nil, fmt.Errorf("bluesky API %s: status %d", endpoint, resp2.StatusCode)
		}
		return respBody, nil
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("bluesky API %s: status %d: %s", endpoint, resp.StatusCode, string(respBody[:min(len(respBody), 200)]))
	}
	return respBody, nil
}

// blueskySearchAPI is the endpoint that supports unauthenticated searchPosts.
const blueskySearchAPI = "https://api.bsky.app/xrpc/"

// blueskySearchAPIGet makes an unauthenticated GET request to api.bsky.app.
func blueskySearchAPIGet(endpoint string) ([]byte, error) {
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequest("GET", blueskySearchAPI+endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("bluesky search API %s: status %d: %s", endpoint, resp.StatusCode, string(body[:min(len(body), 200)]))
	}
	return body, nil
}

// searchBlueskyPosts searches Bluesky posts using the public search API (api.bsky.app).
func searchBlueskyPosts(query string, config *Config) ([]BlueskyPost, error) {
	posts, err := fetchBlueskySearchPublic(query, 300)
	if err != nil {
		return nil, err
	}
	if len(posts) == 0 {
		return nil, fmt.Errorf("Blueskyで「%s」に該当する投稿が見つかりませんでした", query)
	}
	return posts, nil
}

// fetchBlueskySearchPublic searches Bluesky posts via api.bsky.app (no auth required) with pagination.
func fetchBlueskySearchPublic(query string, maxTotal int) ([]BlueskyPost, error) {
	if maxTotal <= 0 {
		maxTotal = 300
	}
	perPage := 100 // API max per request
	maxPages := (maxTotal + perPage - 1) / perPage
	if maxPages > 5 {
		maxPages = 5 // cap at 5 pages to avoid excessive API calls
	}

	var allPosts []BlueskyPost
	seen := make(map[string]bool)
	cursor := ""

	for page := 0; page < maxPages; page++ {
		ep := fmt.Sprintf("app.bsky.feed.searchPosts?q=%s&limit=%d&sort=latest", url.QueryEscape(query), perPage)
		if cursor != "" {
			ep += "&cursor=" + url.QueryEscape(cursor)
		}
		data, err := blueskySearchAPIGet(ep)
		if err != nil {
			if page == 0 {
				return nil, err
			}
			break // return what we have so far
		}

		var searchResp struct {
			Cursor string `json:"cursor"`
			Posts  []struct {
				URI    string `json:"uri"`
				CID    string `json:"cid"`
				Author struct {
					Handle      string `json:"handle"`
					DisplayName string `json:"displayName"`
					Avatar      string `json:"avatar"`
				} `json:"author"`
				Record      json.RawMessage `json:"record"`
				ReplyCount  int             `json:"replyCount"`
				RepostCount int             `json:"repostCount"`
				LikeCount   int             `json:"likeCount"`
				QuoteCount  int             `json:"quoteCount"`
				Embed       json.RawMessage `json:"embed"`
			} `json:"posts"`
		}
		if err := json.Unmarshal(data, &searchResp); err != nil {
			break
		}
		if len(searchResp.Posts) == 0 {
			break
		}

		for _, item := range searchResp.Posts {
			if seen[item.URI] {
				continue
			}
			seen[item.URI] = true

			p := BlueskyPost{
				URI:          item.URI,
				CID:          item.CID,
				AuthorHandle: item.Author.Handle,
				AuthorName:   item.Author.DisplayName,
				AvatarURL:    item.Author.Avatar,
				ReplyCount:   item.ReplyCount,
				RepostCount:  item.RepostCount,
				LikeCount:    item.LikeCount,
				QuoteCount:   item.QuoteCount,
				FetchedAt:    time.Now(),
			}
			var record struct {
				Text      string `json:"text"`
				CreatedAt string `json:"createdAt"`
				Embed     *struct {
					Type     string `json:"$type"`
					External *struct {
						URI   string `json:"uri"`
						Title string `json:"title"`
					} `json:"external"`
				} `json:"embed"`
			}
			if err := json.Unmarshal(item.Record, &record); err == nil {
				p.Text = record.Text
				p.CreatedAt = record.CreatedAt
				if record.Embed != nil && record.Embed.External != nil {
					p.ExternalURL = record.Embed.External.URI
					p.ExternalTitle = record.Embed.External.Title
				}
			}
			if len(item.Embed) > 0 {
				var embed struct {
					Type   string `json:"$type"`
					Images []struct {
						Fullsize string `json:"fullsize"`
						Thumb    string `json:"thumb"`
					} `json:"images"`
					External *struct {
						URI   string `json:"uri"`
						Title string `json:"title"`
					} `json:"external"`
				}
				if err := json.Unmarshal(item.Embed, &embed); err == nil {
					for _, img := range embed.Images {
						if img.Fullsize != "" {
							p.ImageURLs = append(p.ImageURLs, img.Fullsize)
						} else if img.Thumb != "" {
							p.ImageURLs = append(p.ImageURLs, img.Thumb)
						}
					}
					if embed.External != nil && p.ExternalURL == "" {
						p.ExternalURL = embed.External.URI
						p.ExternalTitle = embed.External.Title
					}
				}
			}
			allPosts = append(allPosts, p)
		}

		cursor = searchResp.Cursor
		if cursor == "" {
			break // no more pages
		}
		if len(allPosts) >= maxTotal {
			break
		}
	}

	fmt.Printf("[siki] Bluesky search %q: fetched %d posts across pages\n", query, len(allPosts))
	return allPosts, nil
}

// fetchAllBlueskyPosts fetches posts from all handles concurrently (max 5 goroutines).
func fetchAllBlueskyPosts(handles []string) []BlueskyPost {
	type result struct {
		posts []BlueskyPost
		err   error
	}

	cutoff := time.Now().Add(-24 * time.Hour)
	ch := make(chan result, len(handles))
	sem := make(chan struct{}, 5) // concurrency limit

	for _, handle := range handles {
		sem <- struct{}{}
		go func(h string) {
			defer func() { <-sem }()
			time.Sleep(200 * time.Millisecond) // rate limit courtesy
			posts, err := fetchBlueskyAuthorFeed(h, 50)
			ch <- result{posts, err}
		}(handle)
	}

	var allPosts []BlueskyPost
	seen := make(map[string]bool)
	for range handles {
		r := <-ch
		if r.err != nil {
			continue
		}
		for _, p := range r.posts {
			t, _ := time.Parse(time.RFC3339, p.CreatedAt)
			if !t.IsZero() && t.Before(cutoff) {
				continue
			}
			if !seen[p.URI] {
				seen[p.URI] = true
				allPosts = append(allPosts, p)
			}
		}
	}

	// Sort by created_at descending
	sort.Slice(allPosts, func(i, j int) bool {
		return allPosts[i].CreatedAt > allPosts[j].CreatedAt
	})
	return allPosts
}

// filterRecentBlueskyPosts filters posts within the given duration.
func filterRecentBlueskyPosts(posts []BlueskyPost, dur time.Duration) []BlueskyPost {
	cutoff := time.Now().Add(-dur)
	var result []BlueskyPost
	for _, p := range posts {
		t, _ := time.Parse(time.RFC3339, p.CreatedAt)
		if !t.IsZero() && t.After(cutoff) {
			result = append(result, p)
		}
	}
	return result
}

// formatBlueskyPosts renders Bluesky posts as markdown (similar to formatOneTweet).
func formatBlueskyPosts(posts []BlueskyPost, title string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# %s\n\n", title))
	for i, p := range posts {
		sb.WriteString(fmt.Sprintf("### %d. ", i+1))
		if p.AvatarURL != "" {
			sb.WriteString(fmt.Sprintf("<img src=\"%s\" width=\"32\" height=\"32\" style=\"border-radius:50%%;vertical-align:middle;margin-right:6px\"> ", p.AvatarURL))
		}
		name := p.AuthorName
		if name == "" {
			name = p.AuthorHandle
		}
		sb.WriteString(fmt.Sprintf("**%s** (@%s)\n%s\n", name, p.AuthorHandle, p.Text))
		for _, imgURL := range p.ImageURLs {
			sb.WriteString(fmt.Sprintf("\n![image](%s)\n", imgURL))
		}
		if p.ExternalURL != "" {
			title := p.ExternalTitle
			if title == "" {
				title = p.ExternalURL
			}
			sb.WriteString(fmt.Sprintf("\n🔗 [%s](%s)\n", title, p.ExternalURL))
		}
		if p.ReplyCount > 0 || p.LikeCount > 0 || p.RepostCount > 0 {
			sb.WriteString(fmt.Sprintf("\n💬%d 🔁%d ❤%d", p.ReplyCount, p.RepostCount, p.LikeCount))
			if p.QuoteCount > 0 {
				sb.WriteString(fmt.Sprintf(" 💭%d", p.QuoteCount))
			}
		}
		if p.CreatedAt != "" {
			sb.WriteString(fmt.Sprintf("\n*%s*", p.CreatedAt))
		}
		sb.WriteString(fmt.Sprintf(" [↗](https://bsky.app/profile/%s)\n\n---\n\n", p.AuthorHandle))
	}
	return sb.String()
}

// filterAndSummarizeBluesky summarizes Bluesky posts using LLM (like filterAndSummarizeTwitter).
func filterAndSummarizeBluesky(posts []BlueskyPost, config *Config) string {
	if len(posts) == 0 {
		return ""
	}
	// Sort by engagement and take top 30
	sorted := make([]BlueskyPost, len(posts))
	copy(sorted, posts)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].EngagementScore() > sorted[j].EngagementScore()
	})
	if len(sorted) > 30 {
		sorted = sorted[:30]
	}

	var sb strings.Builder
	for _, p := range sorted {
		name := p.AuthorName
		if name == "" {
			name = p.AuthorHandle
		}
		sb.WriteString(fmt.Sprintf("[%s] %s (@%s): %s", p.CreatedAt, name, p.AuthorHandle, p.Text))
		if p.ExternalURL != "" {
			sb.WriteString(fmt.Sprintf(" URL: %s", p.ExternalURL))
		}
		sb.WriteString(fmt.Sprintf(" (❤%d 🔁%d 💬%d)\n\n", p.LikeCount, p.RepostCount, p.ReplyCount))
	}

	prompt := fmt.Sprintf(`以下はBluesky AI/MLコミュニティの%d件の投稿です。
AI・機械学習・LLM・大規模言語モデル・テクノロジー・プログラミングに関連する投稿のみ抽出し、重要度順にまとめてください。

## 投稿:
%s

## ルール:
- AI/ML/LLM/テクノロジーに関連するもののみ抽出
- 各項目に元の投稿著者名を含めること
- 投稿に含まれるURLはそのまま<a href>で残すこと
- HTML形式（<h3>, <p>, <a>タグ使用）で出力
- 関連投稿が1件もない場合は「該当なし」とだけ出力
- 日本語で出力すること`, len(sorted), truncateStr(sb.String(), 8000))

	_, result, err := callSubModel(prompt, config)
	if err != nil {
		fmt.Printf("[siki] Bluesky summary LLM failed: %v\n", err)
		return ""
	}
	result = strings.TrimSpace(result)
	if result == "該当なし" || len(result) < 20 {
		return ""
	}
	return result
}

// blueskyDeepDivePosts fetches and summarizes URLs from high-engagement posts.
func blueskyDeepDivePosts(posts []BlueskyPost, config *Config) string {
	// Filter posts with engagement >= 50 and external URL
	var candidates []BlueskyPost
	for _, p := range posts {
		if p.EngagementScore() >= 50 && p.ExternalURL != "" {
			candidates = append(candidates, p)
		}
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].EngagementScore() > candidates[j].EngagementScore()
	})
	if len(candidates) > 3 {
		candidates = candidates[:3]
	}
	if len(candidates) == 0 {
		return ""
	}

	var sb strings.Builder
	for _, p := range candidates {
		text, _, _, err := scraplingFetch(p.ExternalURL, 3000, false)
		if err != nil {
			fmt.Printf("[siki] Bluesky deep-dive fetch failed for %s: %v\n", p.ExternalURL, err)
			continue
		}
		if len(text) < 50 {
			continue
		}
		name := p.AuthorName
		if name == "" {
			name = p.AuthorHandle
		}
		summaryPrompt := fmt.Sprintf(`以下のWebページ内容を3-5行で簡潔に要約してください。日本語で出力。

URL: %s
タイトル: %s
共有者: %s (@%s)
エンゲージメント: ❤%d 🔁%d 💬%d

内容:
%s`, p.ExternalURL, p.ExternalTitle, name, p.AuthorHandle, p.LikeCount, p.RepostCount, p.ReplyCount, truncateStr(text, 3000))

		_, summary, err := callSubModel(summaryPrompt, config)
		if err != nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("<h3><a href=\"%s\">%s</a></h3>\n", p.ExternalURL, p.ExternalTitle))
		sb.WriteString(fmt.Sprintf("<p><em>共有: %s (@%s) | ❤%d 🔁%d</em></p>\n", name, p.AuthorHandle, p.LikeCount, p.RepostCount))
		sb.WriteString(fmt.Sprintf("<p>%s</p>\n\n", strings.TrimSpace(summary)))
	}
	return sb.String()
}

// blueskyFeedLoop runs the background Bluesky feed fetcher (WebServer method).
func (ws *WebServer) blueskyFeedLoop() {
	// Initial delay
	time.Sleep(30 * time.Second)

	// First fetch
	ws.runBlueskyFetch()

	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		ws.runBlueskyFetch()
	}
}

func (ws *WebServer) runBlueskyFetch() {
	ws.mu.RLock()
	enabled := ws.config.BlueskyEnabled
	ws.mu.RUnlock()

	if !enabled {
		return
	}

	handles, err := resolveBlueskyHandles(ws.config)
	if err != nil || len(handles) == 0 {
		fmt.Printf("[siki] Bluesky: no handles to fetch (%v)\n", err)
		return
	}

	fmt.Printf("[siki] Bluesky: fetching posts from %d handles...\n", len(handles))
	newPosts := fetchAllBlueskyPosts(handles)
	fmt.Printf("[siki] Bluesky: fetched %d new posts\n", len(newPosts))

	feed := loadBlueskyFeed()
	feed.Posts = mergeBlueskyPosts(feed.Posts, newPosts)
	feed.LastFetched = time.Now()
	if err := saveBlueskyFeed(feed); err != nil {
		fmt.Printf("[siki] Bluesky: save failed: %v\n", err)
	} else {
		fmt.Printf("[siki] Bluesky: saved %d total posts\n", len(feed.Posts))
	}
}

// --- Bluesky Jetstream real-time monitoring ---

// jetstreamConn is a minimal WebSocket client for Jetstream (no external dependencies).
type jetstreamConn struct {
	conn net.Conn
	br   *bufio.Reader
}

// jetstreamDial connects to Bluesky Jetstream via WebSocket.
func jetstreamDial() (*jetstreamConn, error) {
	host := "jetstream2.us-east.bsky.network"
	path := "/subscribe?wantedCollections=app.bsky.feed.post"

	tlsConn, err := tls.DialWithDialer(
		&net.Dialer{Timeout: 15 * time.Second},
		"tcp", host+":443",
		&tls.Config{ServerName: host},
	)
	if err != nil {
		return nil, fmt.Errorf("TLS dial: %w", err)
	}

	// Generate WebSocket key
	keyBytes := make([]byte, 16)
	rand.Read(keyBytes)
	wsKey := base64.StdEncoding.EncodeToString(keyBytes)

	// HTTP upgrade
	req := fmt.Sprintf("GET %s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n",
		path, host, wsKey)
	if _, err := tlsConn.Write([]byte(req)); err != nil {
		tlsConn.Close()
		return nil, fmt.Errorf("write upgrade: %w", err)
	}

	br := bufio.NewReaderSize(tlsConn, 64*1024)

	statusLine, err := br.ReadString('\n')
	if err != nil {
		tlsConn.Close()
		return nil, fmt.Errorf("read status: %w", err)
	}
	if !strings.Contains(statusLine, "101") {
		tlsConn.Close()
		return nil, fmt.Errorf("upgrade failed: %s", strings.TrimSpace(statusLine))
	}
	// Read remaining headers
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			tlsConn.Close()
			return nil, fmt.Errorf("read header: %w", err)
		}
		if strings.TrimSpace(line) == "" {
			break
		}
	}
	return &jetstreamConn{conn: tlsConn, br: br}, nil
}

// readMessage reads the next WebSocket text/binary message, handling control frames.
func (jc *jetstreamConn) readMessage() ([]byte, error) {
	for {
		opcode, payload, err := jc.readFrame()
		if err != nil {
			return nil, err
		}
		switch opcode {
		case 0x1, 0x2: // text, binary
			return payload, nil
		case 0x8: // close
			return nil, fmt.Errorf("connection closed by server")
		case 0x9: // ping -> pong
			jc.sendPong(payload)
		case 0xA: // pong - ignore
		}
	}
}

func (jc *jetstreamConn) readFrame() (opcode byte, payload []byte, err error) {
	var header [2]byte
	if _, err = io.ReadFull(jc.br, header[:]); err != nil {
		return 0, nil, err
	}
	opcode = header[0] & 0x0F
	masked := (header[1] & 0x80) != 0
	length := uint64(header[1] & 0x7F)

	if length == 126 {
		var ext [2]byte
		if _, err = io.ReadFull(jc.br, ext[:]); err != nil {
			return 0, nil, err
		}
		length = uint64(ext[0])<<8 | uint64(ext[1])
	} else if length == 127 {
		var ext [8]byte
		if _, err = io.ReadFull(jc.br, ext[:]); err != nil {
			return 0, nil, err
		}
		length = uint64(ext[0])<<56 | uint64(ext[1])<<48 | uint64(ext[2])<<40 | uint64(ext[3])<<32 |
			uint64(ext[4])<<24 | uint64(ext[5])<<16 | uint64(ext[6])<<8 | uint64(ext[7])
	}
	if length > 16*1024*1024 {
		return 0, nil, fmt.Errorf("frame too large: %d", length)
	}

	var maskKey [4]byte
	if masked {
		if _, err = io.ReadFull(jc.br, maskKey[:]); err != nil {
			return 0, nil, err
		}
	}
	payload = make([]byte, length)
	if _, err = io.ReadFull(jc.br, payload); err != nil {
		return 0, nil, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= maskKey[i%4]
		}
	}
	return opcode, payload, nil
}

func (jc *jetstreamConn) sendPong(data []byte) {
	frame := []byte{0x8A} // FIN + pong
	maskKey := make([]byte, 4)
	rand.Read(maskKey)
	l := len(data)
	if l < 126 {
		frame = append(frame, byte(l)|0x80)
	} else {
		frame = append(frame, 126|0x80, byte(l>>8), byte(l))
	}
	frame = append(frame, maskKey...)
	masked := make([]byte, l)
	for i := range data {
		masked[i] = data[i] ^ maskKey[i%4]
	}
	frame = append(frame, masked...)
	jc.conn.Write(frame)
}

func (jc *jetstreamConn) close() {
	jc.conn.Close()
}

// matchJetstreamKeywords checks if text contains any keyword.
// Short keywords (<=3 chars) use word-boundary matching to avoid false positives (e.g. "AI" in "brain").
func matchJetstreamKeywords(text string, keywords []string) bool {
	textLower := strings.ToLower(text)
	for _, kw := range keywords {
		kwLower := strings.ToLower(kw)
		if len([]rune(kw)) <= 3 {
			// Word-boundary match for short keywords
			re, err := regexp.Compile(`(?i)\b` + regexp.QuoteMeta(kw) + `\b`)
			if err == nil && re.MatchString(text) {
				return true
			}
		} else {
			if strings.Contains(textLower, kwLower) {
				return true
			}
		}
	}
	return false
}

// JetstreamPostMeta holds enriched metadata for a saved Jetstream post.
type JetstreamPostMeta struct {
	OGPTitle         string `json:"ogp_title,omitempty"`
	OGPDesc          string `json:"ogp_desc,omitempty"`
	OGPImage         string `json:"ogp_image,omitempty"`
	URL              string `json:"url,omitempty"`
	Evaluation       string `json:"evaluation,omitempty"`
	Score            int    `json:"score,omitempty"`
	AnalyzedAt       string `json:"analyzed_at,omitempty"`
	Reported         bool   `json:"reported,omitempty"`
	PostText         string `json:"post_text,omitempty"`
	DeepEvaluated    bool   `json:"deep_evaluated,omitempty"`
	DeliveryPriority int    `json:"delivery_priority,omitempty"` // 0-100, higher = more important
	DeepSummary      string `json:"deep_summary,omitempty"`      // detailed evaluation from deep analysis
}

func jetstreamMetaPath(t time.Time) string {
	return filepath.Join(jetstreamPostsDir(), t.Format("2006"), t.Format("0102")+"_meta.json")
}

func loadJetstreamMeta(t time.Time) map[string]JetstreamPostMeta {
	data, err := os.ReadFile(jetstreamMetaPath(t))
	if err != nil {
		return make(map[string]JetstreamPostMeta)
	}
	var m map[string]JetstreamPostMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return make(map[string]JetstreamPostMeta)
	}
	return m
}

func saveJetstreamMeta(t time.Time, m map[string]JetstreamPostMeta) error {
	dir := filepath.Dir(jetstreamMetaPath(t))
	os.MkdirAll(dir, 0755)
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(jetstreamMetaPath(t), data, 0644)
}

// fetchOGP extracts Open Graph metadata from a URL via simple HTTP GET + regex.
func fetchOGP(targetURL string) (title, desc, image string) {
	client := &http.Client{
		Timeout: 10 * time.Second,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 3 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}
	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; siki/1.0)")
	resp, err := client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	// Read only first 32KB to find OGP tags
	head := make([]byte, 32*1024)
	n, _ := io.ReadFull(resp.Body, head)
	html := string(head[:n])

	// Extract OGP meta tags
	ogRe := regexp.MustCompile(`<meta\s+[^>]*property=["']og:([^"']+)["'][^>]*content=["']([^"']*)["']`)
	ogRe2 := regexp.MustCompile(`<meta\s+[^>]*content=["']([^"']*)["'][^>]*property=["']og:([^"']+)["']`)
	for _, m := range ogRe.FindAllStringSubmatch(html, -1) {
		switch m[1] {
		case "title":
			title = m[2]
		case "description":
			desc = m[2]
		case "image":
			image = m[2]
		}
	}
	for _, m := range ogRe2.FindAllStringSubmatch(html, -1) {
		switch m[2] {
		case "title":
			if title == "" {
				title = m[1]
			}
		case "description":
			if desc == "" {
				desc = m[1]
			}
		case "image":
			if image == "" {
				image = m[1]
			}
		}
	}
	// Fallback to <title> tag
	if title == "" {
		titleRe := regexp.MustCompile(`<title[^>]*>([^<]+)</title>`)
		if m := titleRe.FindStringSubmatch(html); len(m) > 1 {
			title = strings.TrimSpace(m[1])
		}
	}
	return
}

// extractURLsFromText extracts HTTP(S) URLs from text.
func extractURLsFromText(text string) []string {
	re := regexp.MustCompile(`https?://[^\s<>")\]]+`)
	return re.FindAllString(text, -1)
}

// analyzeJetstreamPosts processes unanalyzed Jetstream posts during idle time.
func (ws *WebServer) analyzeJetstreamPosts() {
	ws.mu.RLock()
	enabled := ws.config.BlueskyEnabled
	keywords := ws.config.JetstreamKeywords
	config := ws.config
	ws.mu.RUnlock()
	if !enabled || len(keywords) == 0 {
		return
	}

	now := time.Now()
	// Process today and yesterday
	for d := 0; d < 2; d++ {
		t := now.AddDate(0, 0, -d)
		ws.analyzeJetstreamDay(t, config)
	}

	// Deep-evaluate high-score posts for delivery prioritization
	ws.evaluateDeliveryPriority(config)

	// Send report if interval has passed
	ws.jetstreamDeepDiveReport()
}

func (ws *WebServer) analyzeJetstreamDay(t time.Time, config *Config) {
	filename := filepath.Join(jetstreamPostsDir(), t.Format("2006"), t.Format("0102")+".txt")
	data, err := os.ReadFile(filename)
	if err != nil {
		return
	}

	meta := loadJetstreamMeta(t)
	lines := strings.Split(string(data), "\n")

	// Find unanalyzed posts with URLs (limit per cycle)
	type candidate struct {
		key  string // did:rkey
		text string
		urls []string
	}
	var candidates []candidate
	// Iterate newest first (end of file)
	for i := len(lines) - 1; i >= 0; i-- {
		line := lines[i]
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "\t", 5)
		if len(parts) < 5 {
			continue
		}
		did, rkey, text := parts[1], parts[2], parts[4]
		key := did + ":" + rkey
		if _, exists := meta[key]; exists {
			continue // already analyzed
		}
		urls := extractURLsFromText(text)
		if len(urls) == 0 {
			// No URL — still record a basic entry so we don't re-process
			meta[key] = JetstreamPostMeta{AnalyzedAt: time.Now().Format(time.RFC3339)}
			continue
		}
		candidates = append(candidates, candidate{key: key, text: text, urls: urls})
	}

	if len(candidates) == 0 {
		saveJetstreamMeta(t, meta)
		return
	}

	// Process up to 10 posts per cycle
	limit := 10
	if len(candidates) < limit {
		limit = len(candidates)
	}
	candidates = candidates[:limit]

	fmt.Printf("[siki] Jetstream analyze: processing %d posts with URLs (%s)\n", limit, t.Format("01/02"))

	// Concurrent processing (max 5 parallel)
	sem := make(chan struct{}, 5)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, c := range candidates {
		wg.Add(1)
		go func(cand candidate) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			targetURL := cand.urls[0]
			ogpTitle, ogpDesc, ogpImage := fetchOGP(targetURL)

			// Evaluate with fast model
			var evaluation string
			var score int
			if ogpTitle != "" || ogpDesc != "" {
				evalPrompt := fmt.Sprintf(`以下のBlueskyポストとそのリンク先を10点満点で評価してください。
AI・テクノロジー・プログラミング・科学に関連する有益な情報ほど高得点。
雑談・広告・無関係な内容は低得点。

ポスト: %s
リンク先タイトル: %s
リンク先概要: %s

JSON形式で回答（他の文字不要）:
{"score": 7, "summary": "日本語1文の要約"}`,
					truncateStr(cand.text, 500),
					truncateStr(ogpTitle, 200),
					truncateStr(ogpDesc, 500))
				result, err := callFastModel(evalPrompt, config, 200)
				if err == nil {
					result = strings.TrimSpace(result)
					if idx := strings.Index(result, "{"); idx >= 0 {
						if end := strings.LastIndex(result, "}"); end > idx {
							var ev struct {
								Score   int    `json:"score"`
								Summary string `json:"summary"`
							}
							if json.Unmarshal([]byte(result[idx:end+1]), &ev) == nil {
								score = ev.Score
								evaluation = ev.Summary
							}
						}
					}
				}
			}

			mu.Lock()
			meta[cand.key] = JetstreamPostMeta{
				URL:        targetURL,
				OGPTitle:   ogpTitle,
				OGPDesc:    ogpDesc,
				OGPImage:   ogpImage,
				Evaluation: evaluation,
				Score:      score,
				AnalyzedAt: time.Now().Format(time.RFC3339),
				PostText:   truncateStr(cand.text, 500),
			}
			mu.Unlock()
		}(c)
	}

	wg.Wait()

	if err := saveJetstreamMeta(t, meta); err != nil {
		fmt.Printf("[siki] Jetstream analyze: save meta failed: %v\n", err)
	} else {
		analyzed := 0
		for _, m := range meta {
			if m.URL != "" {
				analyzed++
			}
		}
		fmt.Printf("[siki] Jetstream analyze: done (%d posts with URLs analyzed total)\n", analyzed)
	}
}

// evaluateDeliveryPriority performs deep evaluation of high-score posts
// to assign delivery priority (0-100) for the next email report.
// Runs each idle cycle, processing a few unevaluated posts at a time.
func (ws *WebServer) evaluateDeliveryPriority(config *Config) {
	now := time.Now()

	type evalCandidate struct {
		key  string
		meta JetstreamPostMeta
		day  time.Time
	}
	var candidates []evalCandidate

	for d := 0; d < 2; d++ {
		t := now.AddDate(0, 0, -d)
		meta := loadJetstreamMeta(t)
		for key, m := range meta {
			if m.Score >= 7 && !m.Reported && !m.DeepEvaluated && m.URL != "" {
				candidates = append(candidates, evalCandidate{key: key, meta: m, day: t})
			}
		}
	}

	if len(candidates) == 0 {
		return
	}

	// Process up to 5 per cycle to avoid overload
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].meta.Score > candidates[j].meta.Score
	})
	if len(candidates) > 5 {
		candidates = candidates[:5]
	}

	fmt.Printf("[siki] Delivery eval: evaluating %d posts for priority...\n", len(candidates))

	sem := make(chan struct{}, 3)
	var wg sync.WaitGroup

	for _, c := range candidates {
		wg.Add(1)
		go func(cand evalCandidate) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			// Fetch page content
			_, pageText, _, err := scraplingFetch(cand.meta.URL, 3000, false)
			if err != nil || len(pageText) < 50 {
				// Mark as evaluated with low priority
				meta := loadJetstreamMeta(cand.day)
				if m, ok := meta[cand.key]; ok {
					m.DeepEvaluated = true
					m.DeliveryPriority = cand.meta.Score * 5 // fallback: score * 5
					meta[cand.key] = m
					saveJetstreamMeta(cand.day, meta)
				}
				return
			}

			prompt := fmt.Sprintf(`以下の記事を0-100のスケールで配信優先度を評価してください。

## 評価基準
- 技術的な新規性・独創性 (30点)
- 実務への影響度・活用可能性 (25点)
- AI/プログラミング/科学の最先端トピックか (25点)
- 情報の信頼性・ソースの質 (10点)
- 話題性・タイムリーさ (10点)

## 低評価にすべきもの
- 既知の情報の焼き直し → 20以下
- 宣伝・マーケティング色が強い → 15以下
- 内容が薄い・表面的 → 25以下
- 単なるツール紹介で深みがない → 30以下

## Blueskyポスト
%s

## 記事タイトル: %s
## URL: %s

## 記事本文
%s

JSON形式で回答: {"priority": 数値0-100, "summary": "日本語で2-3文の要約", "reason": "優先度の根拠を1文で"}`,
				truncateStr(cand.meta.PostText, 300),
				cand.meta.OGPTitle,
				cand.meta.URL,
				truncateStr(pageText, 2500))

			resp, err := callFastModel(prompt, config, 500)
			if err != nil {
				return
			}

			// Parse response
			priority := cand.meta.Score * 5
			summary := cand.meta.Evaluation
			jsonStart := strings.Index(resp, "{")
			jsonEnd := strings.LastIndex(resp, "}")
			if jsonStart >= 0 && jsonEnd > jsonStart {
				var parsed struct {
					Priority int    `json:"priority"`
					Summary  string `json:"summary"`
					Reason   string `json:"reason"`
				}
				if err := json.Unmarshal([]byte(resp[jsonStart:jsonEnd+1]), &parsed); err == nil {
					if parsed.Priority > 0 && parsed.Priority <= 100 {
						priority = parsed.Priority
					}
					if parsed.Summary != "" {
						summary = parsed.Summary
						if parsed.Reason != "" {
							summary += " — " + parsed.Reason
						}
					}
				}
			}

			meta := loadJetstreamMeta(cand.day)
			if m, ok := meta[cand.key]; ok {
				m.DeepEvaluated = true
				m.DeliveryPriority = priority
				m.DeepSummary = summary
				meta[cand.key] = m
				saveJetstreamMeta(cand.day, meta)
			}
			fmt.Printf("[siki] Delivery eval: %s → priority %d\n", truncateStr(cand.meta.OGPTitle, 40), priority)
		}(c)
	}
	wg.Wait()
}

// generateScoreSVG creates an inline SVG bar chart of post scores for email embedding.
func generateScoreSVG(items []struct {
	Title string
	Score int
	URL   string
}) string {
	if len(items) == 0 {
		return ""
	}
	barH := 32
	gap := 8
	labelW := 280
	barMaxW := 300
	chartH := len(items)*(barH+gap) + 20
	totalW := labelW + barMaxW + 60

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">`, totalW, chartH, totalW, chartH))
	sb.WriteString(`<style>text{font-family:sans-serif;font-size:13px;fill:#333;} .score{font-weight:bold;font-size:14px;}</style>`)

	for i, item := range items {
		y := i*(barH+gap) + 10
		// Truncate title
		title := item.Title
		if len([]rune(title)) > 30 {
			title = string([]rune(title)[:28]) + "…"
		}
		barW := item.Score * barMaxW / 100
		if barW < 10 {
			barW = 10
		}
		// Color based on priority score (0-100)
		color := "#4caf50" // green
		if item.Score >= 80 {
			color = "#e53935" // red = top priority
		} else if item.Score >= 60 {
			color = "#ff9800" // orange = high
		}
		sb.WriteString(fmt.Sprintf(`<text x="0" y="%d" dominant-baseline="middle">%s</text>`, y+barH/2, title))
		sb.WriteString(fmt.Sprintf(`<rect x="%d" y="%d" width="%d" height="%d" rx="4" fill="%s" opacity="0.85"/>`, labelW, y, barW, barH, color))
		sb.WriteString(fmt.Sprintf(`<text x="%d" y="%d" dominant-baseline="middle" class="score" fill="white"> %d</text>`, labelW+barW/2-8, y+barH/2, item.Score))
	}
	sb.WriteString(`</svg>`)
	return sb.String()
}

var (
	lastJetstreamReportTime time.Time
	lastJetstreamReportMu   sync.Mutex
)

// jetstreamDeepDiveReport finds high-score unreported posts, deep-dives their URLs,
// takes screenshots, generates SVG chart and report, and emails it.
// Rate-limited to max 3 times per day (every 8 hours).
func (ws *WebServer) jetstreamDeepDiveReport() {
	// Rate limit: max 3 reports per day (8-hour interval)
	lastJetstreamReportMu.Lock()
	if time.Since(lastJetstreamReportTime) < 8*time.Hour {
		lastJetstreamReportMu.Unlock()
		return
	}
	lastJetstreamReportMu.Unlock()

	ws.mu.RLock()
	config := ws.config
	emailTo := config.EmailTo
	ws.mu.RUnlock()

	if emailTo == "" {
		return // no email configured
	}

	now := time.Now()
	// Collect deep-evaluated unreported posts from today and yesterday
	type reportCandidate struct {
		key  string
		meta JetstreamPostMeta
		day  time.Time
	}
	var candidates []reportCandidate

	for d := 0; d < 2; d++ {
		t := now.AddDate(0, 0, -d)
		meta := loadJetstreamMeta(t)
		for key, m := range meta {
			if m.Score >= 7 && !m.Reported && m.URL != "" && m.DeepEvaluated {
				candidates = append(candidates, reportCandidate{key: key, meta: m, day: t})
			}
		}
	}

	if len(candidates) < 3 {
		return // wait until enough evaluated posts accumulate
	}

	// Sort by DeliveryPriority descending (deep evaluation quality), then by Score
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].meta.DeliveryPriority != candidates[j].meta.DeliveryPriority {
			return candidates[i].meta.DeliveryPriority > candidates[j].meta.DeliveryPriority
		}
		return candidates[i].meta.Score > candidates[j].meta.Score
	})
	// Filter out low-priority posts (below 40/100)
	filtered := candidates[:0]
	for _, c := range candidates {
		if c.meta.DeliveryPriority >= 40 {
			filtered = append(filtered, c)
		}
	}
	candidates = filtered
	if len(candidates) < 3 {
		return
	}
	if len(candidates) > 10 {
		candidates = candidates[:10]
	}

	fmt.Printf("[siki] Jetstream report: deep-diving %d high-score posts...\n", len(candidates))

	// Deep-dive each URL with screenshot
	type deepDiveResult struct {
		meta       JetstreamPostMeta
		report     string // LLM-generated deep-dive
		screenshot []byte // PNG bytes (may be nil)
	}
	var results []deepDiveResult
	sem := make(chan struct{}, 3)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, c := range candidates {
		wg.Add(1)
		go func(cand reportCandidate) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			// Fetch full page content
			_, pageText, _, err := scraplingFetch(cand.meta.URL, 5000, false)
			if err != nil || len(pageText) < 100 {
				return
			}

			// Take screenshot (non-fatal if it fails)
			var ssData []byte
			ssData, err = scraplingScreenshot(cand.meta.URL)
			if err != nil {
				fmt.Printf("[siki] Jetstream report: screenshot failed for %s: %v\n", cand.meta.URL, err)
			}

			// Generate deep-dive report in Japanese
			reportPrompt := fmt.Sprintf(`以下のBlueskyポストで共有されたリンク先の内容を深掘りして、詳細なレポートを日本語で作成してください。

## Blueskyポスト
%s

## リンク先情報
タイトル: %s
URL: %s

## リンク先本文
%s

## レポート要件
- なぜこれが注目に値するか（1-2文）
- 技術的な要点・新規性（箇条書き3-5項目）
- 関連する技術動向や背景（1段落）
- 実務への影響・活用可能性（1段落）

必ず日本語で出力してください。
HTML形式（<h3>, <p>, <ul><li>, <strong>タグ使用）で出力。見出し・段落を使い読みやすく。`,
				truncateStr(cand.meta.PostText, 500),
				cand.meta.OGPTitle,
				cand.meta.URL,
				truncateStr(pageText, 4000))

			_, report, err := callSubModel(reportPrompt, config)
			if err != nil || len(report) < 50 {
				return
			}

			mu.Lock()
			results = append(results, deepDiveResult{
				meta:       cand.meta,
				report:     report,
				screenshot: ssData,
			})
			mu.Unlock()
		}(c)
	}
	wg.Wait()

	if len(results) == 0 {
		fmt.Println("[siki] Jetstream report: no deep-dives succeeded")
		return
	}

	// Sort results by DeliveryPriority descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].meta.DeliveryPriority > results[j].meta.DeliveryPriority
	})

	// Generate SVG score chart (show delivery priority)
	chartItems := make([]struct {
		Title string
		Score int
		URL   string
	}, len(results))
	for i, r := range results {
		chartItems[i].Title = r.meta.OGPTitle
		chartItems[i].Score = r.meta.DeliveryPriority
		chartItems[i].URL = r.meta.URL
	}
	svgChart := generateScoreSVG(chartItems)

	// Collect inline images for email
	var images []EmailImage

	// Build email HTML
	var htmlBuf strings.Builder
	htmlBuf.WriteString(fmt.Sprintf(`<div style="font-family:'Helvetica Neue',Arial,sans-serif;max-width:700px;margin:0 auto;color:#333;">
<h1 style="color:#1a73e8;border-bottom:3px solid #1a73e8;padding-bottom:0.5rem;">Bluesky 注目記事レポート</h1>
<p style="color:#666;font-size:0.95rem;">%s — Jetstream監視で発見した注目記事の深掘りレポート（%d件）</p>
`, now.Format("2006年01月02日 15:04"), len(results)))

	// Embed SVG chart directly in HTML (inline SVG works in most email clients better than as attachment)
	if svgChart != "" {
		htmlBuf.WriteString(`<div style="margin:1rem 0;padding:1rem;background:#f8f9fa;border-radius:8px;">`)
		htmlBuf.WriteString(`<h2 style="margin:0 0 0.8rem 0;color:#333;font-size:1.1rem;">スコア一覧</h2>`)
		htmlBuf.WriteString(svgChart)
		htmlBuf.WriteString(`</div>`)
	}

	htmlBuf.WriteString(`<hr style="border:none;border-top:2px solid #1a73e8;margin:1.5rem 0;">`)

	for i, r := range results {
		// Badge color by delivery priority
		badgeColor := "#4caf50"
		if r.meta.DeliveryPriority >= 80 {
			badgeColor = "#e53935"
		} else if r.meta.DeliveryPriority >= 60 {
			badgeColor = "#ff9800"
		}

		htmlBuf.WriteString(fmt.Sprintf(`
<div style="margin:1.5rem 0;padding:1.2rem;border:1px solid #e0e0e0;border-radius:12px;background:#fafafa;">
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
<span style="background:%s;color:white;padding:2px 10px;border-radius:12px;font-weight:bold;font-size:0.9rem;">優先度 %d</span>
<h2 style="margin:0;font-size:1.1rem;"><a href="%s" style="color:#1a73e8;text-decoration:none;">%d. %s</a></h2>
</div>`, badgeColor, r.meta.DeliveryPriority, r.meta.URL, i+1, r.meta.OGPTitle))

		// Screenshot as inline image
		if len(r.screenshot) > 0 {
			cid := fmt.Sprintf("screenshot_%d", i)
			images = append(images, EmailImage{
				CID:      cid,
				Data:     r.screenshot,
				MimeType: "image/png",
				Filename: fmt.Sprintf("screenshot_%d.png", i+1),
			})
			htmlBuf.WriteString(fmt.Sprintf(`<div style="margin:0.8rem 0;"><img src="cid:%s" style="max-width:100%%;border-radius:8px;border:1px solid #ddd;" alt="Screenshot"></div>`, cid))
		} else if r.meta.OGPImage != "" {
			// Fallback to OGP image
			htmlBuf.WriteString(fmt.Sprintf(`<div style="margin:0.8rem 0;"><img src="%s" style="max-width:100%%;border-radius:8px;" alt=""></div>`, r.meta.OGPImage))
		}

		// Deep summary (from priority evaluation) or original post text
		if r.meta.DeepSummary != "" {
			htmlBuf.WriteString(fmt.Sprintf(`<blockquote style="margin:0.5rem 0;padding:0.5rem 1rem;border-left:3px solid #1a73e8;background:#f0f4ff;color:#555;font-size:0.9rem;">%s</blockquote>`, r.meta.DeepSummary))
		} else if r.meta.PostText != "" {
			htmlBuf.WriteString(fmt.Sprintf(`<blockquote style="margin:0.5rem 0;padding:0.5rem 1rem;border-left:3px solid #1a73e8;background:#f0f4ff;color:#555;font-size:0.9rem;">%s</blockquote>`, truncateStr(r.meta.PostText, 200)))
		}

		// Deep-dive report
		htmlBuf.WriteString(`<div style="margin-top:0.8rem;">`)
		htmlBuf.WriteString(r.report)
		htmlBuf.WriteString(`</div>`)

		htmlBuf.WriteString(`</div>`)
	}

	htmlBuf.WriteString(`<hr style="border:none;border-top:1px solid #ddd;margin-top:2rem;">
<p style="color:#999;font-size:0.8rem;text-align:center;">このレポートは siki Jetstream Monitor が自動生成しました。</p>
</div>`)

	subject := fmt.Sprintf("Bluesky注目記事レポート (%d件) - %s", len(results), now.Format("01/02"))
	if err := sendEmailWithImages(config, subject, htmlBuf.String(), images); err != nil {
		fmt.Printf("[siki] Jetstream report: email send failed: %v\n", err)
		return
	}
	fmt.Printf("[siki] Jetstream report: sent %d articles (%d screenshots) to %s\n", len(results), len(images), emailTo)

	// Record send time for rate limiting
	lastJetstreamReportMu.Lock()
	lastJetstreamReportTime = time.Now()
	lastJetstreamReportMu.Unlock()

	// Mark as reported
	for _, c := range candidates {
		meta := loadJetstreamMeta(c.day)
		if m, ok := meta[c.key]; ok {
			m.Reported = true
			meta[c.key] = m
			saveJetstreamMeta(c.day, meta)
		}
	}
}

// Jetstream event types
type JetstreamEvent struct {
	DID    string           `json:"did"`
	TimeUS int64            `json:"time_us"`
	Kind   string           `json:"kind"`
	Commit *JetstreamCommit `json:"commit,omitempty"`
}

type JetstreamCommit struct {
	Rev        string          `json:"rev"`
	Operation  string          `json:"operation"`
	Collection string          `json:"collection"`
	RKey       string          `json:"rkey"`
	Record     json.RawMessage `json:"record"`
	CID        string          `json:"cid"`
}

type JetstreamPostRecord struct {
	Type      string   `json:"$type"`
	Text      string   `json:"text"`
	CreatedAt string   `json:"createdAt"`
	Langs     []string `json:"langs,omitempty"`
}

func jetstreamPostsDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".siki", "bluesky_posts")
}

// saveJetstreamPost appends a post to yyyy/mmdd.txt.
func saveJetstreamPost(did, rkey, text, createdAt string, keywords []string) error {
	now := time.Now()
	dir := filepath.Join(jetstreamPostsDir(), now.Format("2006"))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	filename := filepath.Join(dir, now.Format("0102")+".txt")

	// Find which keywords matched
	var matched []string
	for _, kw := range keywords {
		kwLower := strings.ToLower(kw)
		if len([]rune(kw)) <= 3 {
			re, err := regexp.Compile(`(?i)\b` + regexp.QuoteMeta(kw) + `\b`)
			if err == nil && re.MatchString(text) {
				matched = append(matched, kw)
			}
		} else {
			if strings.Contains(strings.ToLower(text), kwLower) {
				matched = append(matched, kw)
			}
		}
	}

	// Format: timestamp\tDID\trkey\tkeywords\ttext
	singleLine := strings.ReplaceAll(strings.ReplaceAll(text, "\n", " "), "\t", " ")
	line := fmt.Sprintf("%s\t%s\t%s\t[%s]\t%s\n",
		createdAt, did, rkey, strings.Join(matched, ","), singleLine)

	f, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line)
	return err
}

// searchJetstreamPosts searches saved Jetstream posts by keyword within recent N days.
func searchJetstreamPosts(query string, days int) (string, int) {
	if days <= 0 {
		days = 7
	}
	queryLower := strings.ToLower(query)
	var results []string
	total := 0

	for d := 0; d < days; d++ {
		t := time.Now().AddDate(0, 0, -d)
		filename := filepath.Join(jetstreamPostsDir(), t.Format("2006"), t.Format("0102")+".txt")
		data, err := os.ReadFile(filename)
		if err != nil {
			continue
		}
		lines := strings.Split(string(data), "\n")
		for _, line := range lines {
			if line == "" {
				continue
			}
			total++
			if strings.Contains(strings.ToLower(line), queryLower) {
				results = append(results, line)
			}
		}
	}

	if len(results) == 0 {
		return fmt.Sprintf("過去%d日間の保存済みポスト(%d件)に「%s」は見つかりませんでした。", days, total, query), 0
	}

	// Format results
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Jetstream監視ログ検索: 「%s」 (%d件ヒット / 過去%d日間 %d件中)\n\n", query, len(results), days, total))

	// Show newest first, max 50
	limit := 50
	if len(results) < limit {
		limit = len(results)
	}
	for i := len(results) - 1; i >= len(results)-limit; i-- {
		parts := strings.SplitN(results[i], "\t", 5)
		if len(parts) >= 5 {
			sb.WriteString(fmt.Sprintf("- **%s** `%s` %s: %s\n", parts[0], parts[3], parts[1], parts[4]))
		} else {
			sb.WriteString(fmt.Sprintf("- %s\n", results[i]))
		}
	}
	return sb.String(), len(results)
}

// jetstreamLoop runs the background Jetstream WebSocket consumer.
func (ws *WebServer) jetstreamLoop() {
	time.Sleep(10 * time.Second)
	backoff := 2 * time.Second
	maxBackoff := 5 * time.Minute

	for {
		ws.mu.RLock()
		enabled := ws.config.BlueskyEnabled
		keywords := ws.config.JetstreamKeywords
		ws.mu.RUnlock()

		if !enabled || len(keywords) == 0 {
			time.Sleep(30 * time.Second)
			continue
		}

		fmt.Printf("[siki] Jetstream: connecting (keywords: %v)...\n", keywords)
		err := ws.runJetstream(keywords)
		if err != nil {
			fmt.Printf("[siki] Jetstream: disconnected: %v\n", err)
		}

		fmt.Printf("[siki] Jetstream: reconnecting in %v...\n", backoff)
		time.Sleep(backoff)
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func (ws *WebServer) runJetstream(keywords []string) error {
	jc, err := jetstreamDial()
	if err != nil {
		return err
	}
	defer jc.close()

	fmt.Printf("[siki] Jetstream: connected, monitoring %d keywords\n", len(keywords))
	var saved, scanned int

	for {
		// Re-check keywords (may have been updated)
		ws.mu.RLock()
		currentKeywords := ws.config.JetstreamKeywords
		ws.mu.RUnlock()
		if len(currentKeywords) > 0 {
			keywords = currentKeywords
		}

		data, err := jc.readMessage()
		if err != nil {
			if saved > 0 || scanned > 0 {
				fmt.Printf("[siki] Jetstream: total scanned=%d saved=%d before disconnect\n", scanned, saved)
			}
			return err
		}

		var event JetstreamEvent
		if err := json.Unmarshal(data, &event); err != nil {
			continue
		}
		if event.Kind != "commit" || event.Commit == nil {
			continue
		}
		if event.Commit.Operation != "create" || event.Commit.Collection != "app.bsky.feed.post" {
			continue
		}

		var record JetstreamPostRecord
		if err := json.Unmarshal(event.Commit.Record, &record); err != nil {
			continue
		}

		scanned++
		if !matchJetstreamKeywords(record.Text, keywords) {
			continue
		}

		saved++
		if err := saveJetstreamPost(event.DID, event.Commit.RKey, record.Text, record.CreatedAt, keywords); err != nil {
			fmt.Printf("[siki] Jetstream: save error: %v\n", err)
		}

		if saved%10 == 0 {
			fmt.Printf("[siki] Jetstream: saved %d matching posts (scanned %d)\n", saved, scanned)
		}
	}
}

// filterBlueskyByIntent filters Bluesky posts by topic using LLM (like filterTweetsByIntent).
func filterBlueskyByIntent(posts []BlueskyPost, intent string, config *Config, sendEvent func(StreamEvent)) []BlueskyPost {
	if len(posts) == 0 {
		return nil
	}

	// エンゲージメント順にソートして上位を優先的に処理
	sorted := make([]BlueskyPost, len(posts))
	copy(sorted, posts)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].EngagementScore() > sorted[j].EngagementScore()
	})

	// バッチ処理: 200件ずつ処理（最大600件）
	var allFiltered []BlueskyPost
	batchSize := 200
	maxPosts := 600
	if len(sorted) > maxPosts {
		sorted = sorted[:maxPosts]
	}

	for batchStart := 0; batchStart < len(sorted); batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > len(sorted) {
			batchEnd = len(sorted)
		}
		batch := sorted[batchStart:batchEnd]

		var sb strings.Builder
		for i, p := range batch {
			name := p.AuthorName
			if name == "" {
				name = p.AuthorHandle
			}
			sb.WriteString(fmt.Sprintf("%d. [%s] @%s: %s\n", i, name, p.AuthorHandle, p.Text))
		}

		prompt := fmt.Sprintf(`以下のBluesky投稿リストから「%s」に関連する投稿の番号のみを出力してください。
技術、プログラミング、AI、機械学習、ソフトウェア開発に関する投稿を幅広く含めてください。
関連する投稿がない場合は「なし」と出力。番号はカンマ区切り。

投稿リスト:
%s

関連番号:`, intent, truncateStr(sb.String(), 6000))

		result, err := callFastModel(prompt, config)
		if err != nil || strings.TrimSpace(result) == "なし" {
			continue
		}

		indices := parseIntList(strings.TrimSpace(result))
		for _, idx := range indices {
			if idx >= 0 && idx < len(batch) {
				allFiltered = append(allFiltered, batch[idx])
			}
		}
	}

	if len(allFiltered) == 0 {
		// フォールバック: エンゲージメント上位20件を返す
		top := 20
		if len(sorted) < top {
			top = len(sorted)
		}
		return sorted[:top]
	}
	return allFiltered
}

// parseIntList parses a comma-separated list of integers.
func parseIntList(s string) []int {
	parts := strings.Split(s, ",")
	var result []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if n, err := strconv.Atoi(p); err == nil {
			result = append(result, n)
		}
	}
	return result
}

// BlueskyPostEvaluation holds per-post evaluation results from sub-agent.
type BlueskyPostEvaluation struct {
	Post       BlueskyPost
	Importance int    // 1-10
	Summary    string // sub-agent generated summary
	Relevant   bool
}

// evaluateBlueskyPostsConcurrently evaluates each post individually using sub-agents.
// Instead of batch filtering, each post is fetched (if URL present) and evaluated for
// relevance and importance by a sub-agent in parallel.
func evaluateBlueskyPostsConcurrently(posts []BlueskyPost, intent string, config *Config, sendEvent func(StreamEvent)) []BlueskyPostEvaluation {
	if len(posts) == 0 {
		return nil
	}

	// Sort by engagement, take top candidates
	sorted := make([]BlueskyPost, len(posts))
	copy(sorted, posts)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].EngagementScore() > sorted[j].EngagementScore()
	})
	maxPosts := 40
	if len(sorted) > maxPosts {
		sorted = sorted[:maxPosts]
	}

	if sendEvent != nil {
		sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("上位%d件をサブエージェントで個別評価中...", len(sorted))})
	}

	// Concurrent evaluation with semaphore (max 8 parallel)
	sem := make(chan struct{}, 8)
	var mu sync.Mutex
	var results []BlueskyPostEvaluation
	var wg sync.WaitGroup
	var evaluated int

	for _, p := range sorted {
		wg.Add(1)
		go func(post BlueskyPost) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			eval := evaluateSingleBlueskyPost(post, intent, config)
			mu.Lock()
			results = append(results, eval)
			evaluated++
			count := evaluated
			mu.Unlock()

			if sendEvent != nil {
				label := post.AuthorHandle
				if len(post.Text) > 30 {
					label = string([]rune(post.Text)[:30]) + "..."
				} else if post.Text != "" {
					label = post.Text
				}
				sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("ポスト評価中 %d/%d: @%s「%s」→ 重要度%d", count, len(sorted), post.AuthorHandle, strings.ReplaceAll(label, "\n", " "), eval.Importance)})
			}
		}(p)
	}

	wg.Wait()

	// Filter relevant posts and sort by importance
	var relevant []BlueskyPostEvaluation
	for _, r := range results {
		if r.Relevant && r.Importance >= 3 {
			relevant = append(relevant, r)
		}
	}
	sort.Slice(relevant, func(i, j int) bool {
		return relevant[i].Importance > relevant[j].Importance
	})

	if sendEvent != nil {
		sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("評価完了: %d件中%d件が関連", len(sorted), len(relevant))})
	}

	// Return top 20
	if len(relevant) > 20 {
		relevant = relevant[:20]
	}
	return relevant
}

// evaluateSingleBlueskyPost evaluates a single post: fetches URL content if present,
// then calls sub-agent to assess relevance, importance and generate summary.
func evaluateSingleBlueskyPost(post BlueskyPost, intent string, config *Config) BlueskyPostEvaluation {
	// Fetch URL content if available
	var urlContent string
	if post.ExternalURL != "" {
		text, _, _, err := scraplingFetch(post.ExternalURL, 2000, false)
		if err == nil && len(text) > 50 {
			urlContent = truncateStr(text, 2000)
		}
	}

	// Build evaluation prompt
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("以下のBluesky投稿を評価してください。テーマ: 「%s」\n\n", intent))
	name := post.AuthorName
	if name == "" {
		name = post.AuthorHandle
	}
	sb.WriteString(fmt.Sprintf("著者: %s (@%s)\n", name, post.AuthorHandle))
	sb.WriteString(fmt.Sprintf("投稿: %s\n", post.Text))
	sb.WriteString(fmt.Sprintf("エンゲージメント: ❤%d 🔁%d 💬%d\n", post.LikeCount, post.RepostCount, post.ReplyCount))
	if urlContent != "" {
		sb.WriteString(fmt.Sprintf("\nリンク先内容:\n%s\n", urlContent))
	}
	sb.WriteString(`
あなたはBluesky投稿の評価エージェントです。この投稿を精査し、以下を判定してください:

1. テーマとの関連性 - 投稿内容がテーマに直接関係するか？単なる雑談や無関係な話題ではないか？
2. 情報の価値 - 新しい情報・知見・ツール・論文・プロジェクトの紹介か？既知の一般論の繰り返しではないか？
3. リンク先の内容 - URLがある場合、リンク先の内容は有益か？

以下のJSON形式のみで回答（他の文字は一切不要）:
{"relevant": true, "importance": 7, "summary": "日本語で要約"}

- relevant: テーマに直接関連するならtrue、無関係・薄い関連ならfalse
- importance: 重要度1-10。新規性・技術的深さ・実用性・エンゲージメントを総合評価
- summary: 日本語2-3文。投稿の要点、リンク先の内容があればその要約も含む`)

	_, result, err := callSubModel(sb.String(), config)
	if err != nil {
		return BlueskyPostEvaluation{Post: post, Relevant: false}
	}

	// Extract JSON from response
	result = strings.TrimSpace(result)
	jsonStart := strings.Index(result, "{")
	jsonEnd := strings.LastIndex(result, "}")
	if jsonStart < 0 || jsonEnd < 0 || jsonEnd <= jsonStart {
		return BlueskyPostEvaluation{Post: post, Relevant: false}
	}
	jsonStr := result[jsonStart : jsonEnd+1]

	var eval struct {
		Relevant   bool   `json:"relevant"`
		Importance int    `json:"importance"`
		Summary    string `json:"summary"`
	}
	if err := json.Unmarshal([]byte(jsonStr), &eval); err != nil {
		return BlueskyPostEvaluation{Post: post, Relevant: false}
	}

	return BlueskyPostEvaluation{
		Post:       post,
		Importance: eval.Importance,
		Summary:    eval.Summary,
		Relevant:   eval.Relevant,
	}
}

// formatEvaluatedBlueskyPosts renders evaluated Bluesky posts with importance and summaries.
func formatEvaluatedBlueskyPosts(evals []BlueskyPostEvaluation, title string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# %s\n\n", title))
	for i, e := range evals {
		p := e.Post
		sb.WriteString(fmt.Sprintf("### %d. ", i+1))
		if p.AvatarURL != "" {
			sb.WriteString(fmt.Sprintf("<img src=\"%s\" width=\"32\" height=\"32\" style=\"border-radius:50%%;vertical-align:middle;margin-right:6px\"> ", p.AvatarURL))
		}
		name := p.AuthorName
		if name == "" {
			name = p.AuthorHandle
		}
		// Importance badge
		importanceLabel := "⭐"
		if e.Importance >= 8 {
			importanceLabel = "🔥"
		} else if e.Importance >= 6 {
			importanceLabel = "⭐⭐"
		}
		sb.WriteString(fmt.Sprintf("**%s** (@%s) %s 重要度:%d/10\n", name, p.AuthorHandle, importanceLabel, e.Importance))
		// Summary from sub-agent
		if e.Summary != "" {
			sb.WriteString(fmt.Sprintf("\n> %s\n", e.Summary))
		}
		sb.WriteString(fmt.Sprintf("\n%s\n", p.Text))
		for _, imgURL := range p.ImageURLs {
			sb.WriteString(fmt.Sprintf("\n![image](%s)\n", imgURL))
		}
		if p.ExternalURL != "" {
			linkTitle := p.ExternalTitle
			if linkTitle == "" {
				linkTitle = p.ExternalURL
			}
			sb.WriteString(fmt.Sprintf("\n🔗 [%s](%s)\n", linkTitle, p.ExternalURL))
		}
		if p.ReplyCount > 0 || p.LikeCount > 0 || p.RepostCount > 0 {
			sb.WriteString(fmt.Sprintf("\n💬%d 🔁%d ❤%d", p.ReplyCount, p.RepostCount, p.LikeCount))
			if p.QuoteCount > 0 {
				sb.WriteString(fmt.Sprintf(" 💭%d", p.QuoteCount))
			}
		}
		if p.CreatedAt != "" {
			sb.WriteString(fmt.Sprintf("\n*%s*", p.CreatedAt))
		}
		sb.WriteString(fmt.Sprintf(" [↗](https://bsky.app/profile/%s)\n\n---\n\n", p.AuthorHandle))
	}
	return sb.String()
}

// containsBlueskyKeywords checks if the message is about Bluesky.
func containsBlueskyKeywords(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	keywords := []string{"bluesky", "bsky", "ブルースカイ"}
	for _, kw := range keywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// isBlueskySearchRequest returns true if the message is a Bluesky search request (not just feed).
func isBlueskySearchRequest(userMsg string) bool {
	lower := strings.ToLower(userMsg)
	searchKeywords := []string{"検索", "search", "探して", "調べて", "について", "に関する", "の投稿", "の話題"}
	for _, kw := range searchKeywords {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

// cachedTwitterUserID caches the authenticated user's ID to avoid repeated /users/me calls.
var cachedTwitterUserID string

// hasTwitterOAuth1a returns true if OAuth 1.0a credentials are fully configured.
func hasTwitterOAuth1a(config *Config) bool {
	return config.TwitterConsumerKey != "" && config.TwitterConsumerSecret != "" &&
		config.TwitterAccessToken != "" && config.TwitterAccessSecret != ""
}

// percentEncode encodes a string per RFC 3986 (OAuth 1.0a requires this, NOT application/x-www-form-urlencoded).
func percentEncode(s string) string {
	// url.QueryEscape uses + for space; OAuth needs %20
	return strings.ReplaceAll(url.QueryEscape(s), "+", "%20")
}

// oauthSign generates an OAuth 1.0a Authorization header for a Twitter API request.
func oauthSign(method, rawURL string, queryParams url.Values, config *Config) string {
	nonce := fmt.Sprintf("%d%d", time.Now().UnixNano(), rand.Int63())
	timestamp := fmt.Sprintf("%d", time.Now().Unix())

	// Collect all params (OAuth + query) for signature
	type kv struct{ k, v string }
	var params []kv
	oauthKV := [][2]string{
		{"oauth_consumer_key", config.TwitterConsumerKey},
		{"oauth_nonce", nonce},
		{"oauth_signature_method", "HMAC-SHA1"},
		{"oauth_timestamp", timestamp},
		{"oauth_token", config.TwitterAccessToken},
		{"oauth_version", "1.0"},
	}
	for _, p := range oauthKV {
		params = append(params, kv{p[0], p[1]})
	}
	for k, vs := range queryParams {
		for _, v := range vs {
			params = append(params, kv{k, v})
		}
	}

	// Sort by key, then value (RFC 5849 Section 3.4.1.3.2)
	sort.Slice(params, func(i, j int) bool {
		if params[i].k == params[j].k {
			return params[i].v < params[j].v
		}
		return params[i].k < params[j].k
	})

	// Build parameter string with percent encoding
	var paramParts []string
	for _, p := range params {
		paramParts = append(paramParts, percentEncode(p.k)+"="+percentEncode(p.v))
	}
	paramString := strings.Join(paramParts, "&")

	// Base URL (strip query string)
	parsedURL, _ := url.Parse(rawURL)
	baseURL := fmt.Sprintf("%s://%s%s", parsedURL.Scheme, parsedURL.Host, parsedURL.Path)

	// Signature base string
	signatureBase := strings.ToUpper(method) + "&" + percentEncode(baseURL) + "&" + percentEncode(paramString)

	// Signing key
	signingKey := percentEncode(config.TwitterConsumerSecret) + "&" + percentEncode(config.TwitterAccessSecret)

	// HMAC-SHA1
	mac := hmac.New(sha1.New, []byte(signingKey))
	mac.Write([]byte(signatureBase))
	signature := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	// Build Authorization header
	return fmt.Sprintf(`OAuth oauth_consumer_key="%s", oauth_nonce="%s", oauth_signature="%s", oauth_signature_method="HMAC-SHA1", oauth_timestamp="%s", oauth_token="%s", oauth_version="1.0"`,
		percentEncode(config.TwitterConsumerKey),
		percentEncode(nonce),
		percentEncode(signature),
		percentEncode(timestamp),
		percentEncode(config.TwitterAccessToken))
}

// twitterAPIGet makes an authenticated GET request to Twitter API v2.
// Uses OAuth 1.0a if available (required for user-context endpoints like timeline),
// falls back to Bearer Token for app-only endpoints like search.
func twitterAPIGet(apiURL string, config *Config, requireUserContext bool) (*http.Response, error) {
	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, err
	}

	if requireUserContext && hasTwitterOAuth1a(config) {
		// OAuth 1.0a: parse query params for signature
		parsedURL, _ := url.Parse(apiURL)
		authHeader := oauthSign("GET", apiURL, parsedURL.Query(), config)
		req.Header.Set("Authorization", authHeader)
	} else if config.TwitterBearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+config.TwitterBearerToken)
	} else {
		return nil, fmt.Errorf("no Twitter authentication configured")
	}

	client := &http.Client{Timeout: 30 * time.Second}
	return client.Do(req)
}

// fetchTwitterUserID gets the authenticated user's ID.
func fetchTwitterUserID(config *Config) (string, error) {
	if cachedTwitterUserID != "" {
		return cachedTwitterUserID, nil
	}
	apiURL := "https://api.twitter.com/2/users/me"
	resp, err := twitterAPIGet(apiURL, config, true)
	if err != nil {
		return "", fmt.Errorf("Twitter API request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Twitter API error %d: %s", resp.StatusCode, truncateStr(string(body), 300))
	}
	var result struct {
		Data struct {
			ID       string `json:"id"`
			Username string `json:"username"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("Twitter API decode error: %w", err)
	}
	cachedTwitterUserID = result.Data.ID
	fmt.Printf("[siki] Twitter: authenticated as @%s (ID: %s)\n", result.Data.Username, result.Data.ID)
	return cachedTwitterUserID, nil
}

// fetchTwitterTimeline retrieves the authenticated user's home timeline (last 12 hours).
// Requires OAuth 1.0a (User Context authentication).
func fetchTwitterTimeline(config *Config) ([]TwitterTweet, error) {
	if !hasTwitterOAuth1a(config) {
		return nil, fmt.Errorf("Twitter OAuth 1.0a credentials not configured (Consumer Key/Secret + Access Token/Secret required for timeline)")
	}

	userID, err := fetchTwitterUserID(config)
	if err != nil {
		return nil, err
	}

	// Build request: reverse chronological timeline with media expansion
	startTime := time.Now().Add(-12 * time.Hour).UTC().Format(time.RFC3339)
	apiURL := fmt.Sprintf("https://api.twitter.com/2/users/%s/timelines/reverse_chronological?max_results=100&start_time=%s&tweet.fields=created_at,author_id,text,entities,attachments,public_metrics&expansions=author_id,attachments.media_keys&user.fields=username,name,profile_image_url&media.fields=type,url,preview_image_url,variants",
		userID, url.QueryEscape(startTime))

	resp, err := twitterAPIGet(apiURL, config, true)
	if err != nil {
		return nil, fmt.Errorf("Twitter timeline request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Twitter API error %d: %s", resp.StatusCode, truncateStr(string(body), 500))
	}

	body, _ := io.ReadAll(resp.Body)
	tweets := parseTwitterResponse(body)
	fmt.Printf("[siki] Twitter: fetched %d tweets from timeline\n", len(tweets))
	return tweets, nil
}

// parseTwitterResponse parses a Twitter API v2 response with user and media expansions.
func parseTwitterResponse(body []byte) []TwitterTweet {
	var apiResp struct {
		Data []struct {
			ID          string `json:"id"`
			Text        string `json:"text"`
			CreatedAt   string `json:"created_at"`
			AuthorID    string `json:"author_id"`
			Attachments struct {
				MediaKeys []string `json:"media_keys"`
			} `json:"attachments"`
			Entities struct {
				URLs []struct {
					ExpandedURL string `json:"expanded_url"`
					DisplayURL  string `json:"display_url"`
				} `json:"urls"`
			} `json:"entities"`
			PublicMetrics struct {
				ReplyCount   int `json:"reply_count"`
				RetweetCount int `json:"retweet_count"`
				LikeCount    int `json:"like_count"`
			} `json:"public_metrics"`
		} `json:"data"`
		Includes struct {
			Users []struct {
				ID              string `json:"id"`
				Name            string `json:"name"`
				Username        string `json:"username"`
				ProfileImageURL string `json:"profile_image_url"`
			} `json:"users"`
			Media []struct {
				MediaKey        string `json:"media_key"`
				Type            string `json:"type"`
				URL             string `json:"url"`
				PreviewImageURL string `json:"preview_image_url"`
				Variants        []struct {
					ContentType string `json:"content_type"`
					URL         string `json:"url"`
					BitRate     int    `json:"bit_rate"`
				} `json:"variants"`
			} `json:"media"`
		} `json:"includes"`
	}
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil
	}

	// Build user ID → display name map and profile image map
	userMap := map[string]string{}
	userIconMap := map[string]string{}
	for _, u := range apiResp.Includes.Users {
		userMap[u.ID] = fmt.Sprintf("%s (@%s)", u.Name, u.Username)
		if u.ProfileImageURL != "" {
			userIconMap[u.ID] = u.ProfileImageURL
		}
	}

	// Build media key → TwitterMedia map
	mediaMap := map[string]TwitterMedia{}
	for _, m := range apiResp.Includes.Media {
		tm := TwitterMedia{Type: m.Type, PreviewURL: m.PreviewImageURL}
		if m.URL != "" {
			tm.URL = m.URL
		} else if m.PreviewImageURL != "" {
			tm.URL = m.PreviewImageURL
		}
		// For video/animated_gif, find the best mp4 variant
		if m.Type == "video" || m.Type == "animated_gif" {
			bestBitrate := 0
			for _, v := range m.Variants {
				if v.ContentType == "video/mp4" && v.BitRate >= bestBitrate {
					tm.URL = v.URL
					bestBitrate = v.BitRate
				}
			}
		}
		mediaMap[m.MediaKey] = tm
	}

	// Build tweets
	var tweets []TwitterTweet
	for _, d := range apiResp.Data {
		t := TwitterTweet{
			ID:              d.ID,
			Text:            d.Text,
			CreatedAt:       d.CreatedAt,
			AuthorID:        d.AuthorID,
			Author:          userMap[d.AuthorID],
			ProfileImageURL: userIconMap[d.AuthorID],
			ReplyCount:      d.PublicMetrics.ReplyCount,
			RetweetCount:    d.PublicMetrics.RetweetCount,
			LikeCount:       d.PublicMetrics.LikeCount,
		}
		if t.Author == "" {
			t.Author = d.AuthorID
		}
		// Attach media
		for _, mk := range d.Attachments.MediaKeys {
			if m, ok := mediaMap[mk]; ok {
				t.Media = append(t.Media, m)
			}
		}
		// Collect expanded URLs (skip twitter internal pic/video URLs)
		for _, u := range d.Entities.URLs {
			if u.ExpandedURL != "" && !strings.Contains(u.ExpandedURL, "twitter.com/") && !strings.Contains(u.ExpandedURL, "x.com/") {
				t.URLs = append(t.URLs, u.ExpandedURL)
			}
		}
		tweets = append(tweets, t)
	}
	return tweets
}

// filterAndSummarizeTwitter uses LLM to extract AI-related tweets and summarize them as HTML.
// formatTweets renders tweets as markdown with media and URL embeds.
// filterTweetsByIntent uses LLM to judge each tweet's relevance to the user's intent.
// Processes tweets in batches for efficiency. Returns only relevant tweets.
func filterTweetsByIntent(tweets []TwitterTweet, intent string, config *Config, sendEvent ...func(StreamEvent)) []TwitterTweet {
	if len(tweets) == 0 {
		return nil
	}
	fmt.Printf("[siki] Filtering %d tweets by intent: %q\n", len(tweets), intent)

	var emit func(StreamEvent)
	if len(sendEvent) > 0 {
		emit = sendEvent[0]
	}

	truncText := func(s string, n int) string {
		r := []rune(s)
		if len(r) > n {
			return string(r[:n]) + "..."
		}
		return s
	}

	// Phase 1: LLMでintentからキーワードを抽出（1回だけ）
	if emit != nil {
		emit(StreamEvent{Type: "progress", Content: "キーワード抽出中..."})
	}
	kwPrompt := fmt.Sprintf(`Generate 20 search keywords for filtering tweets about: %s
IMPORTANT: Output BOTH Japanese AND English keywords. At least 8 Japanese, at least 8 English.
Example for "AI news": AI, 人工知能, machine learning, 機械学習, deep learning, 深層学習, LLM, 大規模言語モデル, GPT, ニューラルネット, robot, ロボット, 自動化, automation, データ, data science, 開発, research, モデル, tech
Output comma-separated keywords only:`, intent)
	kwResp, err := callFastModel(kwPrompt, config)
	var keywords []string
	if err == nil {
		for _, kw := range strings.Split(kwResp, ",") {
			kw = strings.TrimSpace(strings.ToLower(kw))
			if kw != "" && len([]rune(kw)) >= 2 {
				keywords = append(keywords, kw)
			}
		}
	}
	// intentの単語自体もキーワードに含める（LLMが出さなかった場合の保険）
	for _, w := range strings.Fields(strings.ToLower(intent)) {
		if len([]rune(w)) >= 2 {
			found := false
			for _, kw := range keywords {
				if kw == w {
					found = true
					break
				}
			}
			if !found {
				keywords = append(keywords, w)
			}
		}
	}
	if len(keywords) == 0 {
		keywords = []string{strings.ToLower(intent)}
	}
	if emit != nil {
		emit(StreamEvent{Type: "progress", Content: fmt.Sprintf("キーワード: %s", strings.Join(keywords, ", "))})
	}
	fmt.Printf("[siki] Filter keywords: %v\n", keywords)

	// Phase 2: 全ツイートをキーワードマッチで即座に判定（1件ずつ表示）
	var relevant []TwitterTweet
	for i, tw := range tweets {
		lower := strings.ToLower(tw.Text + " " + tw.Author)
		matched := ""
		for _, kw := range keywords {
			if strings.Contains(lower, kw) {
				matched = kw
				break
			}
		}
		preview := truncText(strings.ReplaceAll(tw.Text, "\n", " "), 40)
		if matched != "" {
			relevant = append(relevant, tw)
			if emit != nil {
				emit(StreamEvent{Type: "progress", Content: fmt.Sprintf("(%d/%d) ✓ %s [%s]", i+1, len(tweets), preview, matched)})
			}
		} else {
			if emit != nil {
				emit(StreamEvent{Type: "progress", Content: fmt.Sprintf("(%d/%d) ✗ %s", i+1, len(tweets), preview)})
			}
		}
	}

	if emit != nil {
		emit(StreamEvent{Type: "progress", Content: fmt.Sprintf("選別完了: %d/%d件を抽出", len(relevant), len(tweets))})
	}
	fmt.Printf("[siki] Filtered: %d/%d tweets matched intent\n", len(relevant), len(tweets))
	return relevant
}

// fetchConversationThread retrieves reply threads for a tweet using Twitter API.
// Uses conversation_id search if available (requires Basic tier), otherwise returns nil.
func fetchConversationThread(tweetID string, authorHandle string, config *Config) ([]TwitterTweet, error) {
	// Get the tweet's conversation_id
	infoURL := fmt.Sprintf("https://api.twitter.com/2/tweets/%s?tweet.fields=conversation_id", tweetID)
	resp, err := twitterAPIGet(infoURL, config, hasTwitterOAuth1a(config))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("tweet info error %d: %s", resp.StatusCode, truncateStr(string(body), 200))
	}
	var infoResp struct {
		Data struct {
			ConversationID string `json:"conversation_id"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&infoResp)
	convID := infoResp.Data.ConversationID
	if convID == "" {
		convID = tweetID
	}

	// Try search for conversation replies (requires Basic tier)
	searchURL := fmt.Sprintf("https://api.twitter.com/2/tweets/search/recent?query=conversation_id:%s&max_results=100&tweet.fields=created_at,author_id,text,entities,attachments,public_metrics&expansions=author_id,attachments.media_keys&user.fields=username,name,profile_image_url&media.fields=type,url,preview_image_url,variants&sort_order=chronological",
		convID)

	resp2, err := twitterAPIGet(searchURL, config, hasTwitterOAuth1a(config))
	if err != nil {
		return nil, err
	}
	defer resp2.Body.Close()
	if resp2.StatusCode != 200 {
		// Search API not available (free tier) — return nil gracefully
		fmt.Printf("[siki] Thread search unavailable (status %d) for tweet %s — skipping\n", resp2.StatusCode, tweetID)
		return nil, nil
	}
	body, _ := io.ReadAll(resp2.Body)
	thread := parseTwitterResponse(body)

	// Filter out the original tweet to avoid duplication
	var replies []TwitterTweet
	for _, t := range thread {
		if t.ID != tweetID {
			replies = append(replies, t)
		}
	}
	fmt.Printf("[siki] Thread for tweet %s: %d replies\n", tweetID, len(replies))
	return replies, nil
}

// selectTweetsForDeepDive asks LLM which filtered tweets deserve thread expansion.
func selectTweetsForDeepDive(tweets []TwitterTweet, intent string, config *Config) []int {
	if len(tweets) == 0 {
		return nil
	}

	var sb strings.Builder
	for i, tw := range tweets {
		sb.WriteString(fmt.Sprintf("[%d] @%s: %s\n", i, tw.Author, tw.Text))
	}

	prompt := fmt.Sprintf(`Select up to 3 tweets that are most important and worth reading thread replies for topic "%s".
Output only the numbers comma-separated. If none worth deep-diving, output "none".

Tweets:
%s

Numbers:`, intent, sb.String())

	resp, err := callFastModel(prompt, config)
	if err != nil {
		return nil
	}
	resp = strings.TrimSpace(resp)
	if resp == "なし" || resp == "none" || resp == "" {
		return nil
	}

	var indices []int
	for _, numStr := range strings.Split(resp, ",") {
		cleaned := ""
		for _, c := range strings.TrimSpace(numStr) {
			if c >= '0' && c <= '9' {
				cleaned += string(c)
			}
		}
		if cleaned == "" {
			continue
		}
		idx := 0
		for _, c := range cleaned {
			idx = idx*10 + int(c-'0')
		}
		if idx >= 0 && idx < len(tweets) {
			indices = append(indices, idx)
		}
		if len(indices) >= 3 {
			break
		}
	}
	return indices
}

// evaluateAndSelectDeepDive evaluates each filtered tweet's importance and
// whether it's worth deep-diving, emitting per-tweet progress events.
func evaluateAndSelectDeepDive(tweets []TwitterTweet, intent string, config *Config, sendEvent func(StreamEvent)) []int {
	if len(tweets) == 0 {
		return nil
	}

	truncText := func(s string, n int) string {
		r := []rune(strings.ReplaceAll(s, "\n", " "))
		if len(r) > n {
			return string(r[:n]) + "..."
		}
		return string(r)
	}

	// Build tweet list for LLM (include reply count as signal)
	var sb strings.Builder
	for i, tw := range tweets {
		text := strings.ReplaceAll(tw.Text, "\n", " ")
		metrics := ""
		if tw.ReplyCount > 0 || tw.LikeCount > 0 {
			metrics = fmt.Sprintf(" [💬%d ❤%d]", tw.ReplyCount, tw.LikeCount)
		}
		sb.WriteString(fmt.Sprintf("[%d] @%s: %s%s\n", i, tw.Author, text, metrics))
	}

	prompt := fmt.Sprintf(`Rate each tweet's importance for topic "%s" and decide if thread replies should be fetched.
Tweets with many replies (💬) are good DIVE candidates.
Format per line: number|HIGH or MID or LOW|DIVE or SKIP|reason(10 words max)
Example:
0|HIGH|DIVE|Major AI model release announcement
1|MID|SKIP|General opinion about tech
2|LOW|SKIP|Only tangentially related

Tweets:
%s

Ratings:`, intent, sb.String())

	fmt.Printf("[siki] evaluateAndSelectDeepDive: %d tweets, intent=%q\n", len(tweets), intent)
	// Need ~20 tokens per tweet line output
	maxTokens := len(tweets)*25 + 50
	if maxTokens < 300 {
		maxTokens = 300
	}
	resp, err := callFastModel(prompt, config, maxTokens)
	if err != nil {
		fmt.Printf("[siki] evaluateAndSelectDeepDive: callFastModel error: %v\n", err)
		if sendEvent != nil {
			sendEvent(StreamEvent{Type: "progress", Content: "評価失敗、デフォルト選定に切替..."})
		}
		return selectTweetsForDeepDive(tweets, intent, config)
	}
	fmt.Printf("[siki] evaluateAndSelectDeepDive: LFM2.5 response (%d bytes):\n%s\n", len(resp), resp)

	var diveIndices []int
	for _, line := range strings.Split(resp, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 4)
		if len(parts) < 3 {
			continue
		}
		// Parse index
		cleaned := ""
		for _, c := range strings.TrimSpace(parts[0]) {
			if c >= '0' && c <= '9' {
				cleaned += string(c)
			}
		}
		if cleaned == "" {
			continue
		}
		idx := 0
		for _, c := range cleaned {
			idx = idx*10 + int(c-'0')
		}
		if idx < 0 || idx >= len(tweets) {
			continue
		}

		importance := strings.TrimSpace(strings.ToUpper(parts[1]))
		dive := strings.TrimSpace(strings.ToUpper(parts[2]))
		reason := ""
		if len(parts) >= 4 {
			reason = strings.TrimSpace(parts[3])
		}

		tw := tweets[idx]
		preview := truncText(tw.Text, 30)

		// Build progress message
		var icon string
		switch {
		case strings.Contains(importance, "HIGH"):
			icon = "🔴"
		case strings.Contains(importance, "MID"):
			icon = "🟡"
		default:
			icon = "⚪"
		}
		diveLabel := ""
		if strings.Contains(dive, "DIVE") {
			diveLabel = " → 深掘り"
			diveIndices = append(diveIndices, idx)
		}
		msg := fmt.Sprintf("%s %s @%s: %s", icon, importance, tw.Author, preview)
		if reason != "" {
			msg += " (" + reason + ")"
		}
		msg += diveLabel
		if sendEvent != nil {
			sendEvent(StreamEvent{Type: "progress", Content: msg})
		}
	}

	// Also auto-DIVE tweets with many replies (>= 3) that LLM didn't flag
	diveSet := map[int]bool{}
	for _, idx := range diveIndices {
		diveSet[idx] = true
	}
	for i, tw := range tweets {
		if !diveSet[i] && tw.ReplyCount >= 3 {
			diveIndices = append(diveIndices, i)
			diveSet[i] = true
			if sendEvent != nil {
				preview := truncText(strings.ReplaceAll(tw.Text, "\n", " "), 30)
				sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("💬 %d件のリプライ → 深掘り追加: @%s: %s", tw.ReplyCount, tw.Author, preview)})
			}
		}
	}

	if sendEvent != nil {
		if len(diveIndices) > 0 {
			sendEvent(StreamEvent{Type: "progress", Content: fmt.Sprintf("深掘り対象: %d件", len(diveIndices))})
		} else {
			sendEvent(StreamEvent{Type: "progress", Content: "深掘り対象なし"})
		}
	}
	return diveIndices
}

// formatOneTweet renders a single tweet as markdown.
func formatOneTweet(tw TwitterTweet, indent string) string {
	var sb strings.Builder
	if tw.ProfileImageURL != "" {
		sb.WriteString(fmt.Sprintf("%s<img src=\"%s\" width=\"32\" height=\"32\" style=\"border-radius:50%%;vertical-align:middle;margin-right:6px\"> **%s**\n%s%s\n", indent, tw.ProfileImageURL, tw.Author, indent, tw.Text))
	} else {
		sb.WriteString(fmt.Sprintf("%s**%s**\n%s%s\n", indent, tw.Author, indent, tw.Text))
	}
	for _, m := range tw.Media {
		switch m.Type {
		case "photo":
			sb.WriteString(fmt.Sprintf("\n%s![image](%s)\n", indent, m.URL))
		case "video", "animated_gif":
			if m.PreviewURL != "" {
				sb.WriteString(fmt.Sprintf("\n%s[![video](%s)](%s)\n", indent, m.PreviewURL, m.URL))
			} else {
				sb.WriteString(fmt.Sprintf("\n%s[▶ 動画](%s)\n", indent, m.URL))
			}
		}
	}
	for _, u := range tw.URLs {
		sb.WriteString(fmt.Sprintf("\n%s🔗 %s\n", indent, u))
	}
	// Show engagement metrics if available
	if tw.ReplyCount > 0 || tw.LikeCount > 0 || tw.RetweetCount > 0 {
		sb.WriteString(fmt.Sprintf("\n%s💬%d 🔁%d ❤%d", indent, tw.ReplyCount, tw.RetweetCount, tw.LikeCount))
	}
	if tw.CreatedAt != "" {
		sb.WriteString(fmt.Sprintf("\n%s*%s*", indent, tw.CreatedAt))
	}
	sb.WriteString(fmt.Sprintf(" [↗](https://x.com/i/status/%s)\n", tw.ID))
	return sb.String()
}

// ThreadData holds thread replies for a specific tweet.
type ThreadData struct {
	TweetIndex int
	Replies    []TwitterTweet
}

func formatTweets(tweets []TwitterTweet, title string) string {
	return formatTweetsWithThreads(tweets, title, nil)
}

func formatTweetsWithThreads(tweets []TwitterTweet, title string, threads []ThreadData) string {
	// Build thread map: tweet index -> replies
	threadMap := map[int][]TwitterTweet{}
	for _, td := range threads {
		threadMap[td.TweetIndex] = td.Replies
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# %s\n\n", title))
	for i, tw := range tweets {
		sb.WriteString(fmt.Sprintf("### %d. ", i+1))
		sb.WriteString(formatOneTweet(tw, ""))

		// Render thread if available
		if replies, ok := threadMap[i]; ok && len(replies) > 0 {
			sb.WriteString(fmt.Sprintf("\n<details><summary>🧵 スレッド (%d件の返信)</summary>\n\n", len(replies)))
			for _, r := range replies {
				sb.WriteString(formatOneTweet(r, "> "))
				sb.WriteString("\n")
			}
			sb.WriteString("</details>\n")
		}
		sb.WriteString("\n---\n\n")
	}
	return sb.String()
}

func filterAndSummarizeTwitter(tweets []TwitterTweet, config *Config) (string, error) {
	if len(tweets) == 0 {
		return "", nil
	}

	// Build tweet text for LLM
	var sb strings.Builder
	for _, t := range tweets {
		sb.WriteString(fmt.Sprintf("[%s] %s: %s\n\n", t.CreatedAt, t.Author, t.Text))
	}

	prompt := fmt.Sprintf(`以下はTwitterタイムラインの%d件のツイートです。
AI・機械学習・LLM・大規模言語モデル・テクノロジー・プログラミングに関連するツイートのみ抽出し、重要度順にまとめてください。

## ツイート:
%s

## ルール:
- AI/ML/LLM/テクノロジーに関連するもののみ抽出
- 各項目に元のツイート著者名を含めること
- ツイートに含まれるURLはそのまま<a href>で残すこと
- HTML形式（<h3>, <p>, <a>タグ使用）で出力
- 関連ツイートが1件もない場合は「該当なし」とだけ出力
- 日本語で出力すること`, len(tweets), truncateStr(sb.String(), 8000))

	_, result, err := callSubModel(prompt, config)
	if err != nil {
		return "", fmt.Errorf("Twitter summary LLM failed: %w", err)
	}

	result = strings.TrimSpace(result)
	if result == "該当なし" || len(result) < 20 {
		return "", nil
	}
	return result, nil
}

// twitterSearch searches Twitter using the v2 Recent Search API and returns formatted results.
func twitterSearch(query string, maxResults int, config *Config) (string, error) {
	if config.TwitterBearerToken == "" && !hasTwitterOAuth1a(config) {
		return "", fmt.Errorf("Twitter認証が設定されていません。Settings → Twitter で設定してください。")
	}
	if maxResults <= 0 || maxResults > 100 {
		maxResults = 30
	}
	if maxResults < 10 {
		maxResults = 10
	}

	apiURL := fmt.Sprintf("https://api.twitter.com/2/tweets/search/recent?query=%s&max_results=%d&tweet.fields=created_at,author_id,text,public_metrics,entities,attachments&expansions=author_id,attachments.media_keys&user.fields=username,name,profile_image_url&media.fields=type,url,preview_image_url,variants",
		url.QueryEscape(query), maxResults)

	// Search endpoint works with both Bearer Token and OAuth 1.0a
	resp, err := twitterAPIGet(apiURL, config, hasTwitterOAuth1a(config))
	if err != nil {
		return "", fmt.Errorf("Twitter API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Twitter API error %d: %s", resp.StatusCode, truncateStr(string(body), 500))
	}

	body, _ := io.ReadAll(resp.Body)
	tweets := parseTwitterResponse(body)
	if len(tweets) == 0 {
		return fmt.Sprintf("「%s」に関するツイートは見つかりませんでした。", query), nil
	}

	return formatTweets(tweets, fmt.Sprintf("Twitter検索結果: \"%s\" (%d件)", query, len(tweets))), nil
}

// EmailImage represents an inline image attachment for emails.
type EmailImage struct {
	CID      string // Content-ID (referenced in HTML as cid:xxx)
	Data     []byte // Raw image bytes
	MimeType string // e.g. "image/png"
	Filename string
}

func sendEmail(config *Config, subject, htmlBody string) error {
	return sendEmailWithImages(config, subject, htmlBody, nil)
}

func sendEmailWithImages(config *Config, subject, htmlBody string, images []EmailImage) error {
	from := config.EmailFrom
	if from == "" {
		from = config.SMTPUser
	}
	if from == "" {
		from = "siki@localhost"
	}
	host := config.SMTPHost
	if host == "" {
		host = "smtp.gmail.com"
	}
	port := config.SMTPPort
	if port == 0 {
		port = 587
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "From: %s\r\n", from)
	fmt.Fprintf(&buf, "To: %s\r\n", config.EmailTo)
	fmt.Fprintf(&buf, "Subject: =?UTF-8?B?%s?=\r\n", base64.StdEncoding.EncodeToString([]byte(subject)))
	fmt.Fprintf(&buf, "MIME-Version: 1.0\r\n")

	if len(images) == 0 {
		// Simple HTML email
		fmt.Fprintf(&buf, "Content-Type: text/html; charset=UTF-8\r\n\r\n")
		buf.WriteString(htmlBody)
	} else {
		// Multipart related email (HTML + inline images)
		mw := multipart.NewWriter(&buf)
		fmt.Fprintf(&buf, "Content-Type: multipart/related; boundary=%s\r\n\r\n", mw.Boundary())

		// HTML part (base64 encoded for reliable delivery)
		htmlHeader := make(textproto.MIMEHeader)
		htmlHeader.Set("Content-Type", "text/html; charset=UTF-8")
		htmlHeader.Set("Content-Transfer-Encoding", "base64")
		htmlPart, err := mw.CreatePart(htmlHeader)
		if err != nil {
			return fmt.Errorf("failed to create HTML part: %w", err)
		}
		htmlB64 := base64.StdEncoding.EncodeToString([]byte(htmlBody))
		for i := 0; i < len(htmlB64); i += 76 {
			end := i + 76
			if end > len(htmlB64) {
				end = len(htmlB64)
			}
			htmlPart.Write([]byte(htmlB64[i:end] + "\r\n"))
		}

		// Image parts
		for _, img := range images {
			imgHeader := make(textproto.MIMEHeader)
			imgHeader.Set("Content-Type", img.MimeType)
			imgHeader.Set("Content-Transfer-Encoding", "base64")
			imgHeader.Set("Content-ID", fmt.Sprintf("<%s>", img.CID))
			imgHeader.Set("Content-Disposition", fmt.Sprintf("inline; filename=%q", img.Filename))
			imgPart, err := mw.CreatePart(imgHeader)
			if err != nil {
				continue
			}
			encoded := base64.StdEncoding.EncodeToString(img.Data)
			// Write base64 in 76-char lines per RFC 2045
			for i := 0; i < len(encoded); i += 76 {
				end := i + 76
				if end > len(encoded) {
					end = len(encoded)
				}
				imgPart.Write([]byte(encoded[i:end] + "\r\n"))
			}
		}
		mw.Close()
	}

	addr := fmt.Sprintf("%s:%d", host, port)

	conn, err := net.DialTimeout("tcp", addr, 10*time.Second)
	if err != nil {
		return fmt.Errorf("SMTP connect failed: %w", err)
	}

	c, err := smtp.NewClient(conn, host)
	if err != nil {
		conn.Close()
		return fmt.Errorf("SMTP client failed: %w", err)
	}
	defer c.Close()

	tlsConfig := &tls.Config{ServerName: host}
	if err = c.StartTLS(tlsConfig); err != nil {
		return fmt.Errorf("STARTTLS failed: %w", err)
	}

	if config.SMTPUser != "" {
		auth := smtp.PlainAuth("", config.SMTPUser, config.SMTPPass, host)
		if err = c.Auth(auth); err != nil {
			return fmt.Errorf("SMTP auth failed: %w", err)
		}
	}

	if err = c.Mail(from); err != nil {
		return fmt.Errorf("SMTP MAIL FROM failed: %w", err)
	}
	if err = c.Rcpt(config.EmailTo); err != nil {
		return fmt.Errorf("SMTP RCPT TO failed: %w", err)
	}
	w, err := c.Data()
	if err != nil {
		return fmt.Errorf("SMTP DATA failed: %w", err)
	}
	_, err = w.Write(buf.Bytes())
	if err != nil {
		return fmt.Errorf("SMTP write failed: %w", err)
	}
	return w.Close()
}

// formatDigestEmailBody wraps raw HTML content into a complete email HTML document.
func formatDigestEmailBody(subject, body string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>%s</title></head>
<body style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
<h1 style="color: #6f42c1; border-bottom: 2px solid #6f42c1; padding-bottom: 10px;">%s</h1>
%s
<hr style="margin-top: 30px; border: 1px solid #eee;">
<p style="color: #999; font-size: 12px;">Generated by siki (式神)</p>
</body>
</html>`, subject, subject, body)
}

// ---- Self-Improvement Loop ----

// tryPromptImprovement autonomously attempts to improve the system prompt
// and evaluates response quality before/after with a test question.
func (ws *WebServer) tryPromptImprovement() {
	// Only attempt every ~3 cycles (randomized)
	if rand.Intn(3) != 0 {
		return
	}

	fmt.Println("[siki] Auto-improvement: evaluating system prompt...")

	// 1. Get current prompt
	selfMu.RLock()
	currentPrompt := ""
	if currentSelf != nil && currentSelf.Prompt != "" {
		currentPrompt = currentSelf.Prompt
	} else {
		currentPrompt = ws.config.SystemPrompt
	}
	version := 0
	if currentSelf != nil {
		version = currentSelf.Version
	}
	selfMu.RUnlock()

	if len(currentPrompt) < 50 {
		return
	}

	// 2. Collect recent user dissatisfaction patterns
	threads, err := listThreads()
	if err != nil {
		return
	}
	cutoff := time.Now().Add(-24 * time.Hour)
	var issues strings.Builder
	for _, t := range threads {
		if !t.UpdatedAt.After(cutoff) {
			continue
		}
		thread, err := loadThread(t.ID)
		if err != nil {
			continue
		}
		for _, m := range thread.Messages {
			if m.Role == "user" {
				lower := strings.ToLower(m.Content)
				if strings.Contains(lower, "違う") || strings.Contains(lower, "ダメ") ||
					strings.Contains(lower, "できてない") || strings.Contains(lower, "間違") ||
					strings.Contains(lower, "もういい") || strings.Contains(lower, "使えない") {
					issues.WriteString("- " + truncateString(m.Content, 100) + "\n")
				}
			}
		}
	}

	// 3. Ask LLM to suggest prompt improvements
	improvementPrompt := fmt.Sprintf(`あなたはAIシステムの改良エンジニアです。
以下の現在のシステムプロンプト（一部）を分析し、改善案を提案せよ。

## 現在のプロンプト（先頭1000文字）:
%s

## ユーザーからの不満（最近24時間）:
%s

## タスク
1. プロンプトの問題点を1〜2個指摘
2. 改善内容をappendする短いテキスト（200文字以内）を提案
3. 改善不要なら空文字

以下のJSON形式で出力:
{"needs_change": true/false, "append_text": "追加するテキスト", "reason": "理由"}`,
		truncateString(currentPrompt, 1000),
		issues.String())

	_, resp, err := callSubModel(improvementPrompt, ws.config)
	if err != nil {
		fmt.Printf("[siki] Auto-improvement: LLM failed: %v\n", err)
		return
	}

	start := strings.Index(resp, "{")
	end := strings.LastIndex(resp, "}")
	if start < 0 || end <= start {
		return
	}

	var suggestion struct {
		NeedsChange bool   `json:"needs_change"`
		AppendText  string `json:"append_text"`
		Reason      string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(resp[start:end+1]), &suggestion); err != nil {
		return
	}

	if !suggestion.NeedsChange || suggestion.AppendText == "" {
		fmt.Println("[siki] Auto-improvement: no changes needed")
		return
	}

	// 4. Test the improvement with a sample question
	testQuestion := "最新のAIニュースを教えて"
	fmt.Printf("[siki] Auto-improvement: testing proposed change: %s\n", truncateString(suggestion.Reason, 100))

	// Before: generate response with current prompt
	beforePrompt := fmt.Sprintf("%s\n\nユーザー: %s\n短く回答せよ（100文字以内）。", currentPrompt, testQuestion)
	_, beforeResp, err := callSubModel(beforePrompt, ws.config)
	if err != nil {
		return
	}

	// After: generate response with improved prompt
	newPrompt := currentPrompt + "\n" + suggestion.AppendText
	afterPrompt := fmt.Sprintf("%s\n\nユーザー: %s\n短く回答せよ（100文字以内）。", newPrompt, testQuestion)
	_, afterResp, err := callSubModel(afterPrompt, ws.config)
	if err != nil {
		return
	}

	// 5. Judge which is better
	judgePrompt := fmt.Sprintf(`2つのAI応答を比較し、どちらが良いか判定せよ。

質問: %s

回答A（変更前）: %s
回答B（変更後）: %s

JSON形式で出力: {"winner": "A" or "B", "reason": "理由"}`, testQuestion,
		truncateString(beforeResp, 300), truncateString(afterResp, 300))

	_, judgeResp, err := callSubModel(judgePrompt, ws.config)
	if err != nil {
		return
	}

	jStart := strings.Index(judgeResp, "{")
	jEnd := strings.LastIndex(judgeResp, "}")
	if jStart < 0 || jEnd <= jStart {
		return
	}
	var judgment struct {
		Winner string `json:"winner"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(judgeResp[jStart:jEnd+1]), &judgment); err != nil {
		return
	}

	if judgment.Winner != "B" {
		fmt.Printf("[siki] Auto-improvement: proposed change did not improve quality (winner=%s), skipping\n", judgment.Winner)
		return
	}

	// 6. Apply the improvement
	fmt.Printf("[siki] Auto-improvement: applying improvement (v%d → v%d): %s\n", version, version+1, suggestion.Reason)
	selfMu.Lock()
	if currentSelf == nil {
		selfMu.Unlock()
		return
	}
	// Snapshot before modification
	createSnapshot(currentSelf, "auto-improvement: "+suggestion.Reason)
	currentSelf.Version++
	if currentSelf.Prompt == "" {
		currentSelf.Prompt = ws.config.SystemPrompt
	}
	currentSelf.Prompt += "\n" + suggestion.AppendText
	saveSelfState(currentSelf)
	selfMu.Unlock()

	fmt.Printf("[siki] Auto-improvement: system prompt updated to v%d. Reason: %s\n", currentSelf.Version, suggestion.Reason)
}

func (ws *WebServer) selfImproveLoop() {
	ticker := time.NewTicker(3 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ws.mu.RLock()
		idle := time.Since(ws.lastActivity) > 3*time.Minute
		ws.mu.RUnlock()

		if !idle {
			continue
		}

		if !ws.improveMu.TryLock() {
			continue
		}

		fmt.Println("[siki] Self-improvement loop: analyzing recent conversations...")
		ws.runSelfImprovement()
		ws.updateUserProfile()
		ws.tryPromptImprovement()
		ws.analyzeJetstreamPosts()
		ws.runAutonomousThinking()
		ws.runProactiveExecution()
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

// ---- Autonomous Idle Thinking ----

// broadcastIdleEvent sends an event to all connected idle SSE clients (non-blocking).
func (ws *WebServer) broadcastIdleEvent(event StreamEvent) {
	ws.idleClientMu.Lock()
	defer ws.idleClientMu.Unlock()
	for ch := range ws.idleClients {
		select {
		case ch <- event:
		default: // skip slow clients
		}
	}
}

// handleIdleStream is an SSE endpoint for idle thinking events.
func (ws *WebServer) handleIdleStream(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	ch := make(chan StreamEvent, 16)
	ws.idleClientMu.Lock()
	ws.idleClients[ch] = true
	ws.idleClientMu.Unlock()

	defer func() {
		ws.idleClientMu.Lock()
		delete(ws.idleClients, ch)
		ws.idleClientMu.Unlock()
	}()

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-ch:
			data, _ := json.Marshal(event)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

// autonomousTasks defines the pool of idle thinking tasks.
var autonomousTasks = []struct {
	Name       string
	PromptFunc func(summary string) string
}{
	{
		Name: "ユーザー分析",
		PromptFunc: func(summary string) string {
			return fmt.Sprintf(`以下の会話履歴を分析し、ユーザーの特徴を簡潔にまとめてください。
- 職業・技術レベルの推定
- 主な興味・関心分野
- コミュニケーションスタイルの特徴

会話サマリ:
%s

日本語で3-5行で回答してください。`, summary)
		},
	},
	{
		Name: "ニーズ予測",
		PromptFunc: func(summary string) string {
			return fmt.Sprintf(`以下の会話履歴から、ユーザーが次に求めそうなことを予測してください。
- 直近の会話の流れから推測される次のアクション
- まだ解決されていない潜在的な課題
- 提案できそうなこと

会話サマリ:
%s

日本語で3-5行で回答してください。`, summary)
		},
	},
	{
		Name: "興味分野リサーチ",
		PromptFunc: func(summary string) string {
			return fmt.Sprintf(`以下の会話履歴からユーザーの興味分野を特定し、関連する有用な情報や最新トレンドを調べてください。

会話サマリ:
%s

ユーザーに役立ちそうな情報を日本語で3-5行で提供してください。`, summary)
		},
	},
	{
		Name: "会話パターン分析",
		PromptFunc: func(summary string) string {
			return fmt.Sprintf(`以下の会話履歴を分析し、パターンを見つけてください。
- よく使われるツールや機能
- 繰り返し出てくる質問やテーマ
- 改善提案

会話サマリ:
%s

日本語で3-5行で回答してください。`, summary)
		},
	},
}

const idleThreadTitle = "sikiの思考ログ"
const idleThreadIDPrefix = "idle-thoughts-"

// getOrCreateIdleThread returns the thread ID for today's idle thoughts thread.
// Each day gets a separate thread to prevent unbounded growth.
func getOrCreateIdleThread() string {
	if err := initThreadDir(); err != nil {
		return ""
	}

	today := time.Now().Format("2006-01-02")
	todayID := idleThreadIDPrefix + today

	// Look for today's idle thread
	threads, err := listThreads()
	if err == nil {
		for _, t := range threads {
			if t.ID == todayID {
				return t.ID
			}
		}
	}

	// Create today's idle thread
	t := &Thread{
		ID:        todayID,
		Title:     fmt.Sprintf("%s (%s)", idleThreadTitle, today),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	if err := saveThreadMeta(t); err != nil {
		fmt.Printf("[siki] Failed to create idle thread: %v\n", err)
		return ""
	}
	fmt.Printf("[siki] Created idle thoughts thread: %s\n", todayID)

	// Prune old idle threads (keep last 7 days)
	if threads != nil {
		cutoff := time.Now().AddDate(0, 0, -7)
		for _, t := range threads {
			if strings.HasPrefix(t.ID, idleThreadIDPrefix) && t.UpdatedAt.Before(cutoff) {
				os.Remove(filepath.Join(threadDir, t.ID+".json"))
				os.Remove(filepath.Join(threadDir, t.ID+".jsonl"))
				fmt.Printf("[siki] Pruned old idle thread: %s\n", t.ID)
			}
		}
	}

	return todayID
}

// runAutonomousThinking executes one idle thinking task and broadcasts results.
func (ws *WebServer) runAutonomousThinking() {
	// Skip if no one is watching
	ws.idleClientMu.Lock()
	clientCount := len(ws.idleClients)
	ws.idleClientMu.Unlock()
	if clientCount == 0 {
		return
	}

	// Gather recent conversation summaries (last 24h)
	threads, err := listThreads()
	if err != nil || len(threads) == 0 {
		return
	}
	cutoff := time.Now().Add(-24 * time.Hour)
	var summaryBuilder strings.Builder
	for _, t := range threads {
		if t.UpdatedAt.After(cutoff) {
			summaryBuilder.WriteString(fmt.Sprintf("- '%s' (%d messages)\n", t.Title, t.MessageCount))
		}
	}
	summary := summaryBuilder.String()
	if summary == "" {
		return
	}

	// Pick a random task (different from last one)
	var task struct {
		Name       string
		PromptFunc func(string) string
	}
	for attempts := 0; attempts < 10; attempts++ {
		idx := rand.Intn(len(autonomousTasks))
		candidate := autonomousTasks[idx]
		if candidate.Name != ws.lastIdleTask || len(autonomousTasks) == 1 {
			task = candidate
			break
		}
	}
	if task.Name == "" {
		task = autonomousTasks[0]
	}
	ws.lastIdleTask = task.Name

	// Create cancellable context
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	ws.mu.Lock()
	ws.idleCancel = cancel
	ws.mu.Unlock()
	defer func() {
		cancel()
		ws.mu.Lock()
		ws.idleCancel = nil
		ws.mu.Unlock()
	}()

	model := ws.config.SubModel
	if model == "" {
		model = ws.config.ModelName
	}

	fmt.Printf("[siki] Autonomous thinking: starting '%s' with model %s\n", task.Name, model)
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_start", Content: task.Name, Model: model})

	// Build prompt and call LLM with context
	prompt := task.PromptFunc(summary)
	result, err := callOllamaGenerateWithCtx(ctx, model, prompt, 1024, 2*time.Minute, ws.config)
	if err != nil {
		if ctx.Err() != nil {
			fmt.Printf("[siki] Autonomous thinking: '%s' interrupted\n", task.Name)
			ws.broadcastIdleEvent(StreamEvent{Type: "idle_interrupted"})
			return
		}
		fmt.Printf("[siki] Autonomous thinking: '%s' failed: %v\n", task.Name, err)
		return
	}

	fmt.Printf("[siki] Autonomous thinking: '%s' complete (%d bytes)\n", task.Name, len(result))
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_result", Content: result, Name: task.Name})

	// Save result to the dedicated idle thoughts thread
	idleThreadID := getOrCreateIdleThread()
	if idleThreadID != "" {
		appendToLog(idleThreadID, ThreadMessage{
			Role:      "assistant",
			Content:   fmt.Sprintf("【自律思考: %s】\n%s", task.Name, result),
			Timestamp: time.Now().Unix(),
		})
		// Update thread metadata
		metaPath := filepath.Join(threadDir, idleThreadID+".json")
		var thread Thread
		if data, err := os.ReadFile(metaPath); err == nil {
			json.Unmarshal(data, &thread)
		}
		thread.MessageCount++
		thread.UpdatedAt = time.Now()
		saveThreadMeta(&thread)
		fmt.Printf("[siki] Autonomous thinking: saved to idle thread %s\n", idleThreadID)

		// Also log high-engagement Bluesky posts if available
		if ws.config.BlueskyEnabled {
			feed := loadBlueskyFeed()
			recentPosts := filterRecentBlueskyPosts(feed.Posts, 6*time.Hour)
			var notable []BlueskyPost
			for _, p := range recentPosts {
				if p.EngagementScore() >= 50 {
					notable = append(notable, p)
				}
			}
			if len(notable) > 0 {
				if len(notable) > 5 {
					notable = notable[:5]
				}
				var bskySb strings.Builder
				bskySb.WriteString("【Bluesky注目ポスト】\n")
				for _, p := range notable {
					name := p.AuthorName
					if name == "" {
						name = p.AuthorHandle
					}
					bskySb.WriteString(fmt.Sprintf("- %s (@%s): %s [score=%d]\n", name, p.AuthorHandle, truncateStr(p.Text, 100), p.EngagementScore()))
				}
				appendToLog(idleThreadID, ThreadMessage{
					Role:      "assistant",
					Content:   bskySb.String(),
					Timestamp: time.Now().Unix(),
				})
			}
		}
	}
}

// callOllamaGenerateWithCtx is like callOllamaGenerate but accepts a context for cancellation.
func callOllamaGenerateWithCtx(ctx context.Context, model, prompt string, maxTokens int, timeout time.Duration, config *Config) (string, error) {
	endpoint := subModelEndpoint(config)

	if isSubModelVLLM(config) {
		// For vllm, use OpenAI-compatible API with context
		reqBody := map[string]interface{}{
			"model":      model,
			"prompt":     prompt,
			"max_tokens": maxTokens,
		}
		body, err := json.Marshal(reqBody)
		if err != nil {
			return "", fmt.Errorf("marshal error: %w", err)
		}
		req, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/completions", bytes.NewReader(body))
		if err != nil {
			return "", fmt.Errorf("create request error: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: timeout}
		resp, err := client.Do(req)
		if err != nil {
			return "", fmt.Errorf("request error: %w", err)
		}
		defer resp.Body.Close()
		var result struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return "", fmt.Errorf("decode error: %w", err)
		}
		if len(result.Choices) > 0 {
			return strings.TrimSpace(result.Choices[0].Text), nil
		}
		return "", fmt.Errorf("no choices returned")
	}

	// Ollama native API with context
	reqBody := map[string]interface{}{
		"model":      model,
		"prompt":     prompt,
		"stream":     false,
		"keep_alive": -1,
		"options": map[string]interface{}{
			"num_predict": maxTokens,
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("generate request error: %w", err)
	}
	defer resp.Body.Close()

	var genResp struct {
		Response string `json:"response"`
		Thinking string `json:"thinking"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}

	content := genResp.Response
	if content == "" && genResp.Thinking != "" {
		content = genResp.Thinking
	}
	// Strip inline <think> tags
	if idx := strings.Index(content, "</think>"); idx >= 0 {
		content = strings.TrimSpace(content[idx+len("</think>"):])
	}
	return content, nil
}

// ---- Proactive Execution (predict and pre-execute user's next request) ----

const proactiveThreadIDPrefix = "proactive-"

// runProactiveExecution predicts what the user might ask next and executes it in a new thread.
func (ws *WebServer) runProactiveExecution() {
	// Check if still idle
	ws.mu.RLock()
	idle := time.Since(ws.lastActivity) > 3*time.Minute
	ws.mu.RUnlock()
	if !idle {
		return
	}

	// Gather recent conversation summaries
	threads, err := listThreads()
	if err != nil || len(threads) == 0 {
		return
	}

	// Skip if we already have a recent proactive thread (avoid spam)
	for _, t := range threads {
		if t.Proactive && t.Unread && time.Since(t.UpdatedAt) < 30*time.Minute {
			return // already have a recent unread proactive thread
		}
	}

	cutoff := time.Now().Add(-24 * time.Hour)
	var summaryBuilder strings.Builder
	for _, t := range threads {
		if t.UpdatedAt.After(cutoff) && !strings.HasPrefix(t.ID, idleThreadIDPrefix) && !strings.HasPrefix(t.ID, proactiveThreadIDPrefix) {
			summaryBuilder.WriteString(fmt.Sprintf("- '%s' (%d messages)\n", t.Title, t.MessageCount))
		}
	}
	summary := summaryBuilder.String()
	if summary == "" {
		return
	}

	// Create cancellable context
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	ws.mu.Lock()
	ws.idleCancel = cancel
	ws.mu.Unlock()
	defer func() {
		cancel()
		ws.mu.Lock()
		ws.idleCancel = nil
		ws.mu.Unlock()
	}()

	model := ws.config.SubModel
	if model == "" {
		model = ws.config.ModelName
	}

	// Step 1: Predict what the user might ask next
	predictPrompt := fmt.Sprintf(`あなたはユーザーの行動を予測するAIです。以下の会話履歴から、ユーザーが次に聞きそうな質問やリクエストを1つだけ予測してください。

## 最近の会話:
%s

## ルール:
- ユーザーの興味・関心に基づいて、自然な次のリクエストを予測すること
- 具体的で実行可能な質問/リクエストにすること
- 回答は予測される質問/リクエストのみを出力すること（説明不要）
- 日本語で回答すること`, summary)

	fmt.Printf("[siki] Proactive: predicting next user request...\n")
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_start", Content: "次のリクエストを予測中...", Model: model})

	predictedTask, err := callOllamaGenerateWithCtx(ctx, model, predictPrompt, 256, 2*time.Minute, ws.config)
	if err != nil {
		if ctx.Err() != nil {
			ws.broadcastIdleEvent(StreamEvent{Type: "idle_interrupted"})
		}
		fmt.Printf("[siki] Proactive: prediction failed: %v\n", err)
		return
	}
	predictedTask = strings.TrimSpace(predictedTask)
	if predictedTask == "" || len(predictedTask) < 5 {
		return
	}

	fmt.Printf("[siki] Proactive: predicted task: %s\n", predictedTask)

	// Check if still idle before executing
	ws.mu.RLock()
	idle = time.Since(ws.lastActivity) > 3*time.Minute
	ws.mu.RUnlock()
	if !idle {
		ws.broadcastIdleEvent(StreamEvent{Type: "idle_interrupted"})
		return
	}

	// Step 2: Execute the predicted task
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_start", Content: fmt.Sprintf("先行実行: %s", predictedTask), Model: model})

	executePrompt := fmt.Sprintf(`以下のユーザーのリクエストに対して、詳しく回答してください。

リクエスト: %s

## 注意:
- 詳しく有用な回答を提供すること
- 日本語で回答すること
- 具体的な情報や例を含めること`, predictedTask)

	result, err := callOllamaGenerateWithCtx(ctx, model, executePrompt, 2048, 3*time.Minute, ws.config)
	if err != nil {
		if ctx.Err() != nil {
			ws.broadcastIdleEvent(StreamEvent{Type: "idle_interrupted"})
		}
		fmt.Printf("[siki] Proactive: execution failed: %v\n", err)
		return
	}
	result = strings.TrimSpace(result)
	if result == "" {
		return
	}

	// Step 3: Create a new thread with the result
	threadID := proactiveThreadIDPrefix + fmt.Sprintf("%d", time.Now().UnixMilli())
	title := predictedTask
	if len(title) > 50 {
		title = title[:50] + "..."
	}
	t := &Thread{
		ID:        threadID,
		Title:     "💡 " + title,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Unread:    true,
		Proactive: true,
	}
	if err := saveThreadMeta(t); err != nil {
		fmt.Printf("[siki] Proactive: failed to create thread: %v\n", err)
		return
	}

	// Save the predicted question as user message and result as assistant message
	appendToLog(threadID, ThreadMessage{
		Role:      "user",
		Content:   predictedTask,
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "assistant",
		Content:   result,
		Timestamp: time.Now().Unix(),
	})
	t.MessageCount = 2
	saveThreadMeta(t)

	fmt.Printf("[siki] Proactive: created thread '%s' with result (%d bytes)\n", threadID, len(result))
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_result", Content: fmt.Sprintf("先行実行完了: %s", title), Name: "proactive"})

	// Notify idle clients about new unread thread
	ws.broadcastIdleEvent(StreamEvent{Type: "new_thread", Content: threadID, Name: t.Title})
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

// Zeroboot sandbox integration
var (
	zerobootEndpoint = "https://api.zeroboot.dev"
	zerobootAPIKey   = "zb_demo_hn2026"
)

type ZerobootResult struct {
	ID          string  `json:"id"`
	Stdout      string  `json:"stdout"`
	Stderr      string  `json:"stderr"`
	ExitCode    int     `json:"exit_code"`
	ForkTimeMs  float64 `json:"fork_time_ms"`
	ExecTimeMs  float64 `json:"exec_time_ms"`
	TotalTimeMs float64 `json:"total_time_ms"`
	Error       string  `json:"error,omitempty"`
}

func zerobootExec(code, language string, timeoutSec int) (*ZerobootResult, error) {
	if language == "" {
		language = "python"
	}
	if timeoutSec <= 0 {
		timeoutSec = 30
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"code":            code,
		"language":        language,
		"timeout_seconds": timeoutSec,
	})

	req, err := http.NewRequest("POST", zerobootEndpoint+"/v1/exec", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+zerobootAPIKey)

	client := &http.Client{Timeout: time.Duration(timeoutSec+10) * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("zeroboot request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("zeroboot error (status %d): %s", resp.StatusCode, string(body))
	}

	var result ZerobootResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("zeroboot parse error: %w", err)
	}
	return &result, nil
}

// zerobootExecWithFiles executes code in zeroboot and extracts base64-encoded file outputs.
// The code should print files as: __FILE:filename:base64data__
func zerobootExecWithFiles(code, language string, timeoutSec int) (*ZerobootResult, map[string][]byte, error) {
	result, err := zerobootExec(code, language, timeoutSec)
	if err != nil {
		return nil, nil, err
	}

	files := make(map[string][]byte)
	cleanStdout := ""
	for _, line := range strings.Split(result.Stdout, "\n") {
		if strings.HasPrefix(line, "__FILE:") && strings.HasSuffix(line, "__") {
			inner := line[7 : len(line)-2]
			parts := strings.SplitN(inner, ":", 2)
			if len(parts) == 2 {
				data, err := base64.StdEncoding.DecodeString(parts[1])
				if err == nil {
					files[parts[0]] = data
				}
			}
		} else {
			cleanStdout += line + "\n"
		}
	}
	result.Stdout = strings.TrimSpace(cleanStdout)
	return result, files, nil
}

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
RUN pip install --no-cache-dir openai-whisper transformers diffusers accelerate safetensors sentencepiece protobuf imageio imageio-ffmpeg ftfy
RUN mkdir -p /workspace /workspace/output
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

	// Start new container with GPU access, HF cache, shared memory
	home, _ := os.UserHomeDir()
	hfCacheDir := filepath.Join(home, ".cache", "huggingface")
	os.MkdirAll(hfCacheDir, 0755)
	fmt.Println("[siki] Starting Docker container siki-worker with GPU access...")
	startCmd := exec.Command("docker", "run", "-d",
		"--name", dockerContainerName,
		"--gpus", "all",
		"--shm-size=16g",
		"-v", dockerWorkspaceDir+":/workspace",
		"-v", hfCacheDir+":/root/.cache/huggingface",
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

// ============================================================================
// Ollama VRAM Management (unload/reload for CUDA-heavy tasks)
// ============================================================================

// unloadOllamaModels queries ollama for loaded models, unloads them, and returns the list.
func unloadOllamaModels(config *Config) []string {
	ollamaEndpoint := "http://localhost:11434"
	if config != nil && config.SubModelBackend == "ollama" {
		ep := subModelEndpoint(config)
		if ep != "" && strings.Contains(ep, "11434") {
			ollamaEndpoint = strings.TrimSuffix(ep, "/v1")
		}
	}

	// Get loaded models via /api/ps
	var loadedModels []string
	resp, err := http.Get(ollamaEndpoint + "/api/ps")
	if err != nil {
		fmt.Printf("[siki] Could not query ollama models: %v\n", err)
		return nil
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var psResp struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if json.Unmarshal(body, &psResp) == nil {
		for _, m := range psResp.Models {
			loadedModels = append(loadedModels, m.Name)
		}
	}

	if len(loadedModels) == 0 {
		fmt.Println("[siki] No ollama models loaded, nothing to unload")
		return nil
	}

	fmt.Printf("[siki] Unloading %d ollama model(s) to free VRAM: %v\n", len(loadedModels), loadedModels)
	for _, model := range loadedModels {
		reqBody, _ := json.Marshal(map[string]interface{}{
			"model":    model,
			"keep_alive": 0,
		})
		req, _ := http.NewRequest("POST", ollamaEndpoint+"/api/generate", bytes.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: 10 * time.Second}
		r, err := client.Do(req)
		if err != nil {
			fmt.Printf("[siki] Failed to unload %s: %v\n", model, err)
		} else {
			io.ReadAll(r.Body)
			r.Body.Close()
			fmt.Printf("[siki] Unloaded ollama model: %s\n", model)
		}
	}
	time.Sleep(2 * time.Second) // Allow VRAM to stabilize
	return loadedModels
}

// reloadOllamaModels reloads previously unloaded models in the background.
func reloadOllamaModels(models []string, config *Config) {
	if len(models) == 0 {
		return
	}
	ollamaEndpoint := "http://localhost:11434"
	if config != nil && config.SubModelBackend == "ollama" {
		ep := subModelEndpoint(config)
		if ep != "" && strings.Contains(ep, "11434") {
			ollamaEndpoint = strings.TrimSuffix(ep, "/v1")
		}
	}

	fmt.Printf("[siki] Reloading %d ollama model(s): %v\n", len(models), models)
	for _, model := range models {
		reqBody, _ := json.Marshal(map[string]interface{}{
			"model":      model,
			"prompt":     "Say OK.",
			"keep_alive": -1,
			"stream":     false,
			"options":    map[string]interface{}{"num_predict": 1},
		})
		req, _ := http.NewRequest("POST", ollamaEndpoint+"/api/generate", bytes.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: 120 * time.Second}
		r, err := client.Do(req)
		if err != nil {
			fmt.Printf("[siki] Failed to reload %s: %v\n", model, err)
		} else {
			io.ReadAll(r.Body)
			r.Body.Close()
			fmt.Printf("[siki] Reloaded ollama model: %s\n", model)
		}
	}
}

// ============================================================================
// Docker Run Model — Run HuggingFace/GitHub models in Docker GPU environment
// ============================================================================

// dockerRunModel fetches a model's README, generates a Python script, and runs it in Docker.
// Flow: README取得 → ollama解放 → Docker起動 → 環境構築 → 利用可能クラス取得 → コード生成 → 実行(リトライ付き)
func dockerRunModel(modelURL, userPrompt string, config *Config, sendEvent ...func(StreamEvent)) (string, error) {
	emit := func(msg string) {
		fmt.Printf("[siki] docker_run_model: %s\n", msg)
		if len(sendEvent) > 0 && sendEvent[0] != nil {
			sendEvent[0](StreamEvent{Type: "thinking", Content: msg})
		}
	}
	emitContent := func(msg string) {
		if len(sendEvent) > 0 && sendEvent[0] != nil {
			sendEvent[0](StreamEvent{Type: "content", Content: msg})
		}
	}

	// ========================================================================
	// Phase 1: README取得（LLM不要、ネットワークのみ）
	// ========================================================================
	emit(fmt.Sprintf("🔗 モデルページを取得中: %s", modelURL))

	title, readmeText, _, err := scraplingFetch(modelURL, 15000, false)
	if err != nil || len(readmeText) < 50 {
		emit("Scrapling失敗、直接HTTPで再取得中...")
		a := &Agent{config: config}
		readmeText, err = a.webFetchQuick(modelURL)
		if err != nil {
			return "", fmt.Errorf("モデルページの取得に失敗: %w", err)
		}
		title = modelURL
	}
	if len(readmeText) > 6000 {
		readmeText = readmeText[:6000]
	}
	emit(fmt.Sprintf("📄 取得完了: %s (%d文字)", title, len(readmeText)))

	// ========================================================================
	// Phase 2: VRAM確保 + Docker環境構築（コード生成の前にやる）
	// ========================================================================
	emit("🧹 VRAM確保のためollamaモデルをアンロード中...")
	unloadedModels := unloadOllamaModels(config)
	if len(unloadedModels) > 0 {
		emit(fmt.Sprintf("✅ %d個のモデルをアンロード: %s", len(unloadedModels), strings.Join(unloadedModels, ", ")))
	} else {
		emit("ℹ️ アンロード対象のollamaモデルなし")
	}

	emit("🐳 DockerコンテナをGPU付きで起動中...")
	if err := ensureDockerContainer(); err != nil {
		go reloadOllamaModels(unloadedModels, config)
		return "", fmt.Errorf("Dockerコンテナが利用不可: %w", err)
	}
	emit("✅ Dockerコンテナ準備完了")

	// Create output directory
	dockerExec("mkdir -p /workspace/output", 30)

	// ========================================================================
	// Phase 3: Docker内のML環境を最新化し、利用可能クラスを取得
	// ========================================================================
	emit("📦 diffusersを最新版にアップグレード中...")
	upgradeOut, upgradeErr := dockerExec("pip install --no-cache-dir --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate 2>&1 | tail -5", 600)
	if upgradeErr != nil {
		emit(fmt.Sprintf("⚠️ diffusersアップグレード警告: %v", upgradeErr))
	} else {
		emit(fmt.Sprintf("✅ diffusersアップグレード完了: %s", truncateStr(upgradeOut, 200)))
	}

	// Query available Pipeline classes — this gives the LLM accurate info for code gen
	emit("🔍 利用可能なPipelineクラスを確認中...")
	availClasses, _ := dockerExec(`python3 -c "import diffusers; print([x for x in dir(diffusers) if 'Pipeline' in x])" 2>&1`, 30)
	availClasses = strings.TrimSpace(availClasses)
	emit(fmt.Sprintf("📋 利用可能クラス: %s", truncateStr(availClasses, 500)))

	// Also check for model-specific classes (e.g. LTX for LTX-2.3)
	modelHint := ""
	lowerURL := strings.ToLower(modelURL)
	if strings.Contains(lowerURL, "ltx") {
		ltxClasses, _ := dockerExec(`python3 -c "import diffusers; print([x for x in dir(diffusers) if 'LTX' in x])" 2>&1`, 30)
		if strings.TrimSpace(ltxClasses) != "" && !strings.Contains(ltxClasses, "Error") {
			modelHint = fmt.Sprintf("\n\n## このモデルに関連するクラス:\n%s", strings.TrimSpace(ltxClasses))
		}
	} else if strings.Contains(lowerURL, "flux") {
		fluxClasses, _ := dockerExec(`python3 -c "import diffusers; print([x for x in dir(diffusers) if 'Flux' in x])" 2>&1`, 30)
		if strings.TrimSpace(fluxClasses) != "" && !strings.Contains(fluxClasses, "Error") {
			modelHint = fmt.Sprintf("\n\n## このモデルに関連するクラス:\n%s", strings.TrimSpace(fluxClasses))
		}
	} else if strings.Contains(lowerURL, "stable") || strings.Contains(lowerURL, "sdxl") {
		sdClasses, _ := dockerExec(`python3 -c "import diffusers; print([x for x in dir(diffusers) if 'Stable' in x or 'SDXL' in x])" 2>&1`, 30)
		if strings.TrimSpace(sdClasses) != "" && !strings.Contains(sdClasses, "Error") {
			modelHint = fmt.Sprintf("\n\n## このモデルに関連するクラス:\n%s", strings.TrimSpace(sdClasses))
		}
	}

	// Check installed torch/CUDA version for the prompt
	torchInfo, _ := dockerExec(`python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1`, 15)
	torchInfo = strings.TrimSpace(torchInfo)

	// ========================================================================
	// Phase 4: 環境情報を含めてPythonコード生成
	// ========================================================================
	emit("🧠 READMEと環境情報を元にPythonスクリプトを生成中...")
	codePrompt := fmt.Sprintf(`以下のモデルページの情報とDocker実行環境の情報を参考に、Pythonスクリプトを生成せよ。

## モデルURL: %s
## ユーザーリクエスト: %s

## モデルページの内容:
%s

## Docker実行環境の情報:
- %s
- 利用可能なdiffusers Pipelineクラス一覧: %s%s

## 要件:
- 出力ファイルは必ず /workspace/output/ ディレクトリに保存すること
- 画像の場合: /workspace/output/result.png
- 動画の場合: /workspace/output/result.mp4 （exportメソッドやimageio等で保存）
- テキストの場合: /workspace/output/result.txt
- CUDAを使用すること（torch.device("cuda")）
- メモリ節約のため torch.bfloat16 または torch.float16 を使うこと
- enable_model_cpu_offload() を使ってVRAM節約すること
- import文から始めて、完全に動作するスクリプトにすること
- os.makedirs("/workspace/output", exist_ok=True) を最初に呼ぶこと
- 必要なpipパッケージがあれば、スクリプト冒頭にコメントで # pip: package1 package2 と書くこと

重要:
- 上記の「利用可能なPipelineクラス一覧」に含まれるクラス名のみを使うこと！存在しないクラスをimportしないこと！
- READMEにdiffusersの使用例コードがあれば、それを優先して参考にすること
- Pythonコードのみ出力。説明不要、コードフェンスも不要。`, modelURL, userPrompt, readmeText, torchInfo, availClasses, modelHint)

	_, pythonCode, err := callSubModelWith(codePrompt, config, "", 5*time.Minute)
	if err != nil {
		altModel := pickRetryModel(config, 1)
		emit(fmt.Sprintf("⚠️ デフォルトモデルがタイムアウト。%s で再試行中...", altModel))
		_, pythonCode, err = callSubModelWith(codePrompt, config, altModel, 5*time.Minute)
		if err != nil {
			go reloadOllamaModels(unloadedModels, config)
			return "", fmt.Errorf("Pythonスクリプトの生成に失敗（両モデル）: %w", err)
		}
	}

	pythonCode = cleanCodeFences(pythonCode)

	codeLines := strings.Count(pythonCode, "\n") + 1
	emit(fmt.Sprintf("✅ スクリプト生成完了 (%d行, %dバイト)", codeLines, len(pythonCode)))
	preview := pythonCode
	if lines := strings.SplitN(preview, "\n", 8); len(lines) > 7 {
		preview = strings.Join(lines[:7], "\n") + "\n..."
	}
	emitContent("\n```python\n" + preview + "\n```\n")

	// ========================================================================
	// Phase 5: 依存パッケージのインストール + 実行（リトライ付き）
	// ========================================================================

	// Extract pip deps from # pip: comments
	var pipDeps []string
	for _, line := range strings.Split(pythonCode, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# pip:") {
			deps := strings.TrimPrefix(line, "# pip:")
			for _, d := range strings.Fields(deps) {
				pipDeps = append(pipDeps, d)
			}
		}
	}

	// Write script
	scriptPath := filepath.Join(dockerWorkspaceDir, "run_model.py")
	if err := os.WriteFile(scriptPath, []byte(pythonCode), 0644); err != nil {
		go reloadOllamaModels(unloadedModels, config)
		return "", fmt.Errorf("スクリプト書き込み失敗: %w", err)
	}

	// Install pip dependencies
	if len(pipDeps) > 0 {
		emit(fmt.Sprintf("📦 依存パッケージをインストール中: %s", strings.Join(pipDeps, ", ")))
		installCmd := "pip install --no-cache-dir " + strings.Join(pipDeps, " ")
		out, installErr := dockerExec(installCmd, 300)
		if installErr != nil {
			emit(fmt.Sprintf("⚠️ pip install 警告: %v", installErr))
			fmt.Printf("[siki] pip install output: %s\n", out)
		} else {
			emit("✅ 依存パッケージインストール完了")
		}
	}

	// Run script with aggressive auto-fix retries
	const maxRetries = 10
	currentCode := pythonCode
	var output string

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt == 0 {
			emit("🚀 Docker内でPythonスクリプトを実行中（最大10分）...")
		} else {
			emit(fmt.Sprintf("🚀 修正版スクリプトを実行中（リトライ %d/%d）...", attempt, maxRetries))
		}

		output, err = dockerExec("cd /workspace && python3 run_model.py 2>&1", 600)

		if len(output) > 0 {
			outPreview := truncateStr(output, 1000)
			emit(fmt.Sprintf("📋 実行出力:\n%s", outPreview))
		}

		// Check success
		hasTraceback := strings.Contains(output, "Traceback") || strings.Contains(output, "Error:")
		hasOutput := fileExistsInDocker("/workspace/output/")
		if err == nil && (!hasTraceback || hasOutput) {
			emit("✅ スクリプト実行成功!")
			break
		}

		if attempt >= maxRetries {
			emit(fmt.Sprintf("❌ %d回リトライしたが成功しなかった", maxRetries))
			break
		}

		emit(fmt.Sprintf("❌ エラー発生（試行 %d/%d）。エラーを分析して修正中...", attempt+1, maxRetries+1))

		// Extract pip install commands from error output
		pkgSet := map[string]bool{}
		for _, line := range strings.Split(output, "\n") {
			line = strings.TrimSpace(line)

			// "pip install xxx yyy zzz"
			if idx := strings.Index(line, "pip install"); idx >= 0 {
				pkgStr := line[idx+len("pip install"):]
				pkgStr = strings.TrimPrefix(pkgStr, " --no-cache-dir")
				for _, pkg := range strings.Fields(pkgStr) {
					pkg = strings.Trim(pkg, "`,.'\"")
					if pkg != "" && pkg != "pip" && !strings.HasPrefix(pkg, "-") {
						pkgSet[pkg] = true
					}
				}
			}

			// "No module named 'xxx'"
			if strings.Contains(line, "No module named") {
				if qi := strings.Index(line, "'"); qi >= 0 {
					mod := line[qi+1:]
					if qe := strings.Index(mod, "'"); qe >= 0 {
						mod = strings.Split(mod[:qe], ".")[0]
						pkgSet[mod] = true
					}
				}
			}

			// "cannot import name 'Xxx' from 'diffusers'" → upgrade diffusers
			if strings.Contains(line, "cannot import name") && strings.Contains(line, "from 'diffusers'") {
				pkgSet["git+https://github.com/huggingface/diffusers.git"] = true
			}

			// CUDA OOM → add cpu_offload hint for retry
			if strings.Contains(line, "CUDA out of memory") || strings.Contains(line, "cudaErrorMemoryAllocation") {
				// Will be handled in retry prompt below
			}
		}

		// Run pip install for detected packages
		if len(pkgSet) > 0 {
			var pkgs []string
			for p := range pkgSet {
				pkgs = append(pkgs, p)
			}
			installCmd := "pip install --no-cache-dir " + strings.Join(pkgs, " ")
			emit(fmt.Sprintf("📦 エラーから検出したパッケージをインストール: %s", strings.Join(pkgs, ", ")))
			pipOut, pipErr := dockerExec(installCmd, 300)
			if pipErr != nil {
				emit(fmt.Sprintf("⚠️ pip install エラー: %v", pipErr))
			}
			if len(pipOut) > 0 {
				emit(fmt.Sprintf("📦 pip出力: %s", truncateStr(pipOut, 300)))
			}

			// Re-run same script after installing packages
			emit("🔄 パッケージインストール後、同じスクリプトを再実行...")
			output, err = dockerExec("cd /workspace && python3 run_model.py 2>&1", 600)
			if len(output) > 0 {
				emit(fmt.Sprintf("📋 再実行出力:\n%s", truncateStr(output, 1000)))
			}
			hasTraceback = strings.Contains(output, "Traceback") || strings.Contains(output, "Error:")
			hasOutput = fileExistsInDocker("/workspace/output/")
			if err == nil && (!hasTraceback || hasOutput) {
				emit("✅ パッケージインストール後に成功!")
				break
			}
		}

		// Re-query available classes (may have changed after pip install)
		freshClasses, _ := dockerExec(`python3 -c "import diffusers; print([x for x in dir(diffusers) if 'Pipeline' in x])" 2>&1`, 30)

		// Build CUDA OOM hint if applicable
		oomHint := ""
		if strings.Contains(output, "CUDA out of memory") || strings.Contains(output, "cudaErrorMemoryAllocation") {
			oomHint = `
- CUDA OOMが発生している。以下の対策を全て適用せよ:
  - pipe.enable_model_cpu_offload() を使え（pipe.to("cuda")の代わりに）
  - torch.bfloat16 を使え
  - enable_attention_slicing() があれば使え
  - 解像度/フレーム数を下げよ（画像: 512x512以下、動画: 短く）`
		}

		// Ask alternate LLM to fix the code
		altModel := pickRetryModel(config, attempt+1)
		emit(fmt.Sprintf("🔄 %s にエラーを渡してスクリプトを修正中...", altModel))

		retryPrompt := fmt.Sprintf(`Pythonスクリプトがエラーになった。修正した完全なスクリプトを出力せよ。

## エラー出力:
%s

## 元のスクリプト:
%s

## Docker環境で利用可能なdiffusersのPipelineクラス:
%s

修正ルール:
- エラーの原因を特定して修正せよ
- ImportErrorの場合、上記の利用可能クラス一覧から正しいクラス名を使え
- 存在しないクラスを使うな！利用可能クラス一覧にあるもののみ使え！%s
- 出力先は /workspace/output/ を維持
- os.makedirs("/workspace/output", exist_ok=True) を最初に呼べ
- Pythonコードのみ出力。説明不要、コードフェンスも不要。`, truncateStr(output, 2000), currentCode, truncateStr(freshClasses, 500), oomHint)

		_, retryCode, retryErr := callSubModelWith(retryPrompt, config, altModel, 5*time.Minute)
		if retryErr != nil || len(retryCode) < 50 {
			emit(fmt.Sprintf("⚠️ %s でのコード修正に失敗: %v", altModel, retryErr))
			continue
		}

		retryCode = cleanCodeFences(retryCode)
		currentCode = retryCode
		os.WriteFile(scriptPath, []byte(retryCode), 0644)

		retryLines := strings.Count(retryCode, "\n") + 1
		emit(fmt.Sprintf("✅ 修正版スクリプト生成完了 (%d行)", retryLines))
		retryPreview := retryCode
		if lines := strings.SplitN(retryPreview, "\n", 6); len(lines) > 5 {
			retryPreview = strings.Join(lines[:5], "\n") + "\n..."
		}
		emitContent("\n```python\n" + retryPreview + "\n```\n")
	}

	// Reload ollama models in background
	emit("🔄 ollamaモデルをバックグラウンドでリロード中...")
	go reloadOllamaModels(unloadedModels, config)

	// Collect output files
	emit("📂 出力ファイルを回収中...")
	result, collectErr := collectDockerOutput(output)
	if collectErr != nil {
		emit(fmt.Sprintf("⚠️ 出力回収エラー: %v", collectErr))
		return result, collectErr
	}
	emit("✅ 完了!")
	return result, nil
}

// fileExistsInDocker checks if any result file exists in the Docker output directory.
func fileExistsInDocker(prefix string) bool {
	out, _ := dockerExec("ls /workspace/output/ 2>/dev/null", 10)
	return strings.TrimSpace(out) != ""
}

// truncateStr truncates a string to maxLen characters.
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// cleanCodeFences strips markdown code fences from LLM-generated code.
func cleanCodeFences(code string) string {
	code = strings.TrimSpace(code)
	if strings.HasPrefix(code, "```python") {
		code = code[len("```python"):]
	} else if strings.HasPrefix(code, "```") {
		code = code[3:]
	}
	if strings.HasSuffix(code, "```") {
		code = code[:len(code)-3]
	}
	return strings.TrimSpace(code)
}

// collectDockerOutput copies output files from Docker workspace to playground and returns display text.
func collectDockerOutput(scriptOutput string) (string, error) {
	// List output files
	listing, _ := dockerExec("ls -la /workspace/output/ 2>/dev/null", 10)
	if strings.TrimSpace(listing) == "" || strings.Contains(listing, "No such file") {
		// No output files — return script output as text
		if scriptOutput != "" {
			return "Script output:\n```\n" + scriptOutput + "\n```", nil
		}
		return "", fmt.Errorf("no output produced")
	}

	// Copy files from Docker workspace to playground
	home, _ := os.UserHomeDir()
	srcDir := filepath.Join(home, ".siki", "workspace", "output")
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return "Script output:\n```\n" + scriptOutput + "\n```", nil
	}

	var results []string
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		srcPath := filepath.Join(srcDir, entry.Name())
		dstName := fmt.Sprintf("model_%d_%s", time.Now().UnixMilli(), entry.Name())
		dstPath := filepath.Join(playgroundDir, dstName)

		data, err := os.ReadFile(srcPath)
		if err != nil {
			continue
		}
		if err := os.WriteFile(dstPath, data, 0644); err != nil {
			continue
		}

		urlPath := "/playground/" + dstName
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		switch ext {
		case ".png", ".jpg", ".jpeg", ".webp", ".gif":
			results = append(results, fmt.Sprintf("![Generated Image](%s)", urlPath))
		case ".mp4", ".webm", ".avi", ".mov":
			results = append(results, fmt.Sprintf("<video src=\"%s\" controls autoplay loop style=\"max-width:100%%\"></video>", urlPath))
		case ".txt", ".json", ".csv":
			text := string(data)
			if len(text) > 5000 {
				text = text[:5000] + "\n... (truncated)"
			}
			results = append(results, fmt.Sprintf("Output (%s):\n```\n%s\n```", entry.Name(), text))
		default:
			results = append(results, fmt.Sprintf("[%s](%s)", entry.Name(), urlPath))
		}

		fmt.Printf("[siki] Output file: %s → %s\n", entry.Name(), urlPath)
	}

	// Clean up output directory for next run
	dockerExec("rm -rf /workspace/output/*", 10)

	if len(results) == 0 {
		return "Script output:\n```\n" + scriptOutput + "\n```", nil
	}

	result := strings.Join(results, "\n\n")
	if scriptOutput != "" && len(scriptOutput) < 1000 {
		result += "\n\nScript log:\n```\n" + scriptOutput + "\n```"
	}
	return result, nil
}

// ============================================================================
// Flux Image Generation Server Management
// ============================================================================

const imageServerScript = `#!/usr/bin/env python3
"""Flux Klein 4B Image Generation Server for siki"""
import os, sys, io, base64, json, time, signal
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing dependencies...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn[standard]", "pydantic"])
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

pipe = None
model_loaded = False

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    num_steps: int = 4
    guidance_scale: float = 3.5

@asynccontextmanager
async def lifespan(app):
    yield
    print("Shutting down image server...")

app = FastAPI(lifespan=lifespan)

def load_model():
    global pipe, model_loaded
    if model_loaded:
        return
    import torch
    from diffusers import DiffusionPipeline
    model_id = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")
    print(f"Loading {model_id}...")
    dtype = torch.bfloat16

    # Use DiffusionPipeline.from_pretrained for auto-detection of pipeline class
    # This handles FLUX.1, FLUX.2, and other variants automatically
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)

    use_fp8 = os.environ.get("FLUX_FP8", "0") == "1"
    if use_fp8:
        try:
            from torchao.quantization import float8_weight_only, quantize_
            quantize_(pipe.transformer, float8_weight_only())
            print("Using FP8 quantization")
        except ImportError:
            print("torchao not available, using bfloat16")

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    model_loaded = True
    print(f"Model loaded: {model_id}")

@app.get("/health")
async def health():
    return {"status": "ready" if model_loaded else "loading", "model_loaded": model_loaded}

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not model_loaded:
        try:
            load_model()
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": f"Model not loaded: {e}"})
    try:
        w = max(256, min(1024, (req.width // 8) * 8))
        h = max(256, min(1024, (req.height // 8) * 8))
        image = pipe(
            prompt=req.prompt,
            width=w,
            height=h,
            num_inference_steps=req.num_steps,
            guidance_scale=req.guidance_scale,
        ).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": b64, "width": w, "height": h}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/load")
async def load():
    """Eagerly load model"""
    try:
        load_model()
        return {"status": "loaded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("IMAGE_PORT", "8100"))
    if "--preload" in sys.argv:
        load_model()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
`

var (
	imageServerProcess *exec.Cmd
	imageServerReady   bool
	imageServerMu      sync.Mutex
	imageServerDir     string
)

// detectVRAM returns free VRAM in MB using nvidia-smi
func detectVRAM() int {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	// Take first GPU's free memory
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return 0
	}
	val := strings.TrimSpace(lines[0])
	// Some GPUs (Jetson/GB10) return "[N/A]" or "Not Supported"
	if strings.Contains(val, "N/A") || strings.Contains(val, "Not") {
		// GPU exists but can't report free memory — try total memory
		cmd2 := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits")
		out2, err2 := cmd2.Output()
		if err2 == nil {
			val2 := strings.TrimSpace(strings.Split(string(out2), "\n")[0])
			if mb2, err3 := strconv.Atoi(val2); err3 == nil {
				return mb2
			}
		}
		// Can't query total either (shared memory GPU like Jetson) — assume capable
		return 16000
	}
	mb, err := strconv.Atoi(val)
	if err != nil {
		return 0
	}
	return mb
}

// canRunImageServer checks if GPU exists and python3 is available
func canRunImageServer() bool {
	if _, err := exec.LookPath("python3"); err != nil {
		return false
	}
	// Check if nvidia-smi exists (GPU present)
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return false
	}
	return true
}

// initImageServerDir creates ~/.siki/image_server/ and writes server.py
func initImageServerDir() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	imageServerDir = filepath.Join(home, ".siki", "image_server")
	if err := os.MkdirAll(imageServerDir, 0755); err != nil {
		return err
	}
	// Also create output directory for generated images
	outputDir := filepath.Join(home, ".siki", "image_server", "output")
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}
	scriptPath := filepath.Join(imageServerDir, "server.py")
	return os.WriteFile(scriptPath, []byte(imageServerScript), 0644)
}

// startImageServer launches the Python subprocess and waits for health
func startImageServer(config *Config) error {
	imageServerMu.Lock()
	defer imageServerMu.Unlock()

	if imageServerReady {
		return nil
	}

	if err := initImageServerDir(); err != nil {
		return fmt.Errorf("failed to init image server dir: %w", err)
	}

	scriptPath := filepath.Join(imageServerDir, "server.py")
	endpoint := config.ImageEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8100"
	}
	// Extract port from endpoint
	port := "8100"
	if idx := strings.LastIndex(endpoint, ":"); idx >= 0 {
		port = endpoint[idx+1:]
	}

	modelName := config.ImageModel
	if modelName == "" {
		modelName = "black-forest-labs/FLUX.2-klein-4B"
	}

	// Find Python with CUDA torch support
	// Priority: 1) existing venvs with torch+CUDA, 2) system python3
	pythonPath := "python3"
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, "knowledgeCore", "venv", "bin", "python3"),
		filepath.Join(home, ".egox_venv", "bin", "python"),
	}
	for _, cand := range candidates {
		if _, serr := os.Stat(cand); serr == nil {
			checkCmd := exec.Command(cand, "-c", "import torch, diffusers; assert torch.cuda.is_available()")
			if err := checkCmd.Run(); err == nil {
				pythonPath = cand
				fmt.Printf("[siki] Using Python with CUDA torch: %s\n", pythonPath)
				break
			}
		}
	}
	if pythonPath == "python3" {
		fmt.Println("[siki] WARNING: No CUDA-enabled Python found. Image generation may fail.")
	}

	cmd := exec.Command(pythonPath, scriptPath)
	cmd.Env = append(os.Environ(),
		"IMAGE_PORT="+port,
		"FLUX_MODEL="+modelName,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// Set process group so we can kill children
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start image server: %w", err)
	}
	imageServerProcess = cmd

	// Poll for health with timeout
	healthURL := endpoint + "/health"
	deadline := time.Now().Add(120 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(2 * time.Second)
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				imageServerReady = true
				fmt.Println("[siki] Image server is ready")
				return nil
			}
		}
	}

	// Timed out - kill the process
	stopImageServer()
	return fmt.Errorf("image server failed to start within 120 seconds")
}

// ensureImageServer performs lazy startup on first use
func ensureImageServer(config *Config) error {
	if imageServerReady {
		return nil
	}

	// Check if server is already running (external process)
	endpoint := config.ImageEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8100"
	}
	resp, err := http.Get(endpoint + "/health")
	if err == nil {
		resp.Body.Close()
		if resp.StatusCode == 200 {
			imageServerReady = true
			return nil
		}
	}

	fmt.Println("[siki] Starting Flux image server (first use, this may take a while)...")
	return startImageServer(config)
}

// stopImageServer sends SIGTERM then SIGKILL
func stopImageServer() {
	imageServerMu.Lock()
	defer imageServerMu.Unlock()

	if imageServerProcess == nil {
		return
	}

	fmt.Println("[siki] Stopping image server...")
	// Send SIGTERM to process group
	if imageServerProcess.Process != nil {
		syscall.Kill(-imageServerProcess.Process.Pid, syscall.SIGTERM)
		done := make(chan error, 1)
		go func() { done <- imageServerProcess.Wait() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			syscall.Kill(-imageServerProcess.Process.Pid, syscall.SIGKILL)
			<-done
		}
	}
	imageServerProcess = nil
	imageServerReady = false
}

// generateImage calls the image server API and saves the PNG
func generateImage(prompt string, width, height int, config *Config) (string, error) {
	if err := ensureImageServer(config); err != nil {
		return "", fmt.Errorf("image server not available: %w", err)
	}

	endpoint := config.ImageEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8100"
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"prompt": prompt,
		"width":  width,
		"height": height,
	})

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post(endpoint+"/generate", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("image generation request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("image generation failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		ImageBase64 string `json:"image_base64"`
		Width       int    `json:"width"`
		Height      int    `json:"height"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode image response: %w", err)
	}

	// Decode base64 and save to playground directory
	imgData, err := base64.StdEncoding.DecodeString(result.ImageBase64)
	if err != nil {
		return "", fmt.Errorf("failed to decode image base64: %w", err)
	}

	// Save to playground dir with unique name
	filename := fmt.Sprintf("image_%d.png", time.Now().UnixNano())
	filePath := filepath.Join(playgroundDir, filename)
	if err := os.WriteFile(filePath, imgData, 0644); err != nil {
		return "", fmt.Errorf("failed to save image: %w", err)
	}

	urlPath := "/playground/" + filename
	return urlPath, nil
}

// ============================================================================
// Helios Video Generation Server Management
// ============================================================================

const videoServerScript = `#!/usr/bin/env python3
"""Helios Video Generation Server for siki"""
import os, sys, base64, json, time

# Ensure all dependencies
def ensure_deps():
    deps = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn[standard]",
        "pydantic": "pydantic",
        "ftfy": "ftfy",
        "imageio": "imageio",
    }
    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["--quiet"])
    # Ensure imageio-ffmpeg separately (hyphen in name)
    try:
        import imageio_ffmpeg
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg", "--quiet"])

ensure_deps()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

pipe = None
model_loaded = False

class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = 33
    height: int = 384
    width: int = 640
    seed: int = 42

app = FastAPI()

def ensure_diffusers_latest():
    """Ensure diffusers has HeliosPyramidPipeline (requires latest source)."""
    try:
        from diffusers import HeliosPyramidPipeline
        print("HeliosPyramidPipeline available")
    except ImportError:
        print("HeliosPyramidPipeline not available, upgrading diffusers from source...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps",
            "git+https://github.com/huggingface/diffusers.git", "--quiet"])
        print("diffusers updated from source")

def free_vram_for_video():
    """Unload ollama models to free VRAM for video generation."""
    import urllib.request
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        for model in ["gpt-oss:latest", "qwen3.5:latest"]:
            data = json.dumps({"model": model, "keep_alive": 0}).encode()
            req = urllib.request.Request(f"{ollama_url}/api/generate", data=data,
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        print("Ollama models unloaded to free VRAM")
        time.sleep(2)
    except Exception as e:
        print(f"Warning: could not unload ollama models: {e}")

def load_model():
    global pipe, model_loaded
    if model_loaded:
        return
    ensure_diffusers_latest()
    free_vram_for_video()
    import torch
    from diffusers import HeliosPyramidPipeline
    model_id = os.environ.get("VIDEO_MODEL", "BestWishYSH/Helios-Distilled")
    print(f"Loading video model {model_id}...")

    pipe = HeliosPyramidPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")

    model_loaded = True
    print(f"Video model loaded: {model_id}")

@app.get("/health")
async def health():
    return {"status": "ready" if model_loaded else "loading", "model_loaded": model_loaded}

@app.post("/generate")
async def generate(req: GenerateRequest):
    if not model_loaded:
        try:
            load_model()
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": f"Model not loaded: {e}"})
    try:
        import torch
        from diffusers.utils import export_to_video

        # Round num_frames to nearest multiple of 33
        nf = max(33, ((req.num_frames + 16) // 33) * 33)
        print(f"Generating video: {req.prompt[:80]}... frames={nf}")

        output = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            num_frames=nf,
            generator=torch.Generator("cuda").manual_seed(req.seed),
        )

        # Save to temp file
        output_dir = os.environ.get("VIDEO_OUTPUT_DIR", "/tmp")
        ts = int(time.time() * 1000)
        output_path = os.path.join(output_dir, f"video_{ts}.mp4")
        export_to_video(output.frames[0], output_path, fps=24)

        # Read and base64 encode
        with open(output_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.remove(output_path)

        return {"video_base64": video_b64, "num_frames": nf, "fps": 24}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/load")
async def load():
    try:
        load_model()
        return {"status": "loaded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("VIDEO_PORT", "8101"))
    if "--preload" in sys.argv:
        load_model()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
`

// ============================================================================
// Scrapling Web Fetch Server
// ============================================================================

const scraplingServerScript = `#!/usr/bin/env python3
"""Scrapling-based web fetch server for siki"""
import os, sys, json

def ensure_deps():
    deps = {"fastapi": "fastapi", "uvicorn": "uvicorn[standard]", "pydantic": "pydantic", "scrapling": "scrapling[all]", "curl_cffi": "curl_cffi"}
    missing = []
    for mod, pkg in deps.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        import subprocess
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["--quiet"])

ensure_deps()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class FetchRequest(BaseModel):
    url: str
    max_length: int = 15000
    stealth: bool = False

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10

@app.get("/health")
async def health():
    return {"status": "ready"}

@app.post("/fetch")
async def fetch(req: FetchRequest):
    try:
        if req.stealth:
            from scrapling.fetchers import StealthyFetcher
            page = StealthyFetcher.fetch(req.url, headless=True, timeout=20)
        else:
            from scrapling.fetchers import Fetcher
            page = Fetcher.get(req.url, timeout=20)

        title = ""
        title_els = page.css("title")
        if title_els:
            title = title_els[0].text

        # Try article/main content first, fallback to body
        text = ""
        for selector in ["article", "main", "[role=main]", ".post-content", ".entry-content", ".article-body"]:
            els = page.css(selector)
            if els:
                t = els[0].get_all_text()
                if len(t) > 100:
                    text = t
                    break
        if not text:
            body_els = page.css("body")
            if body_els:
                text = body_els[0].get_all_text()

        # Clean up: collapse blank lines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        text = "\n".join(lines)

        if len(text) > req.max_length:
            text = text[:req.max_length] + "\n\n... (truncated)"

        # Extract links
        links = []
        for a in page.css("a")[:30]:
            href = a.attrib.get("href", "")
            link_text = a.text.strip() if a.text else ""
            if href and link_text and href.startswith("http"):
                links.append({"text": link_text[:100], "url": href})
            if len(links) >= 20:
                break

        return {"title": title, "text": text, "links": links, "url": req.url}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

class ScreenshotRequest(BaseModel):
    url: str
    width: int = 1280
    height: int = 800

@app.post("/screenshot")
async def screenshot(req: ScreenshotRequest):
    try:
        import base64
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright", "--quiet"])
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": req.width, "height": req.height})
            await page.goto(req.url, timeout=20000, wait_until="networkidle")
            png_bytes = await page.screenshot(full_page=False)
            await browser.close()
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            return {"image": b64, "width": req.width, "height": req.height}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/search")
async def search(req: SearchRequest):
    try:
        from scrapling.fetchers import Fetcher
        import urllib.parse

        encoded = urllib.parse.quote_plus(req.query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded}"
        page = Fetcher.get(search_url, timeout=20)

        results = []
        for item in page.css(".result"):
            title_els = item.css(".result__a")
            snippet_els = item.css(".result__snippet")
            if not title_els:
                continue

            title = title_els[0].text.strip() if title_els[0].text else ""
            url = title_els[0].attrib.get("href", "")

            # Resolve DuckDuckGo redirect
            if "uddg=" in url:
                import re
                m = re.search(r"uddg=([^&]+)", url)
                if m:
                    url = urllib.parse.unquote(m.group(1))

            snippet = snippet_els[0].get_all_text().strip() if snippet_els else ""
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= req.max_results:
                break

        return {"query": req.query, "results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("SCRAPLING_PORT", "8102"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
`

var (
	scraplingServerProcess *exec.Cmd
	scraplingServerReady   bool
	scraplingServerMu      sync.Mutex
	scraplingServerDir     string
	scraplingEndpoint      = "http://localhost:8102"
)

func initScraplingServerDir() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	scraplingServerDir = filepath.Join(home, ".siki", "scrapling_server")
	if err := os.MkdirAll(scraplingServerDir, 0755); err != nil {
		return err
	}
	scriptPath := filepath.Join(scraplingServerDir, "server.py")
	return os.WriteFile(scriptPath, []byte(scraplingServerScript), 0644)
}

func startScraplingServer() error {
	scraplingServerMu.Lock()
	defer scraplingServerMu.Unlock()

	if scraplingServerReady {
		return nil
	}

	if err := initScraplingServerDir(); err != nil {
		return fmt.Errorf("failed to init scrapling server dir: %w", err)
	}

	scriptPath := filepath.Join(scraplingServerDir, "server.py")

	// Find python3 (no CUDA needed for scrapling)
	pythonPath := "python3"
	home, _ := os.UserHomeDir()
	// Prefer existing venv that likely has pip packages
	venvPython := filepath.Join(home, "knowledgeCore", "venv", "bin", "python3")
	if _, serr := os.Stat(venvPython); serr == nil {
		pythonPath = venvPython
	}

	fmt.Printf("[siki] Starting scrapling server with %s...\n", pythonPath)
	cmd := exec.Command(pythonPath, scriptPath)
	cmd.Env = append(os.Environ(), "SCRAPLING_PORT=8102")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start scrapling server: %w", err)
	}
	scraplingServerProcess = cmd

	// Wait for server to be ready
	client := &http.Client{Timeout: 5 * time.Second}
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(1 * time.Second)
		resp, err := client.Get(scraplingEndpoint + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				scraplingServerReady = true
				fmt.Println("[siki] Scrapling server ready")
				return nil
			}
		}
	}
	return fmt.Errorf("scrapling server startup timeout")
}

func ensureScraplingServer() error {
	if scraplingServerReady {
		// Quick health check
		client := &http.Client{Timeout: 3 * time.Second}
		resp, err := client.Get(scraplingEndpoint + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return nil
			}
		}
		scraplingServerReady = false
	}
	return startScraplingServer()
}

func stopScraplingServer() {
	scraplingServerMu.Lock()
	defer scraplingServerMu.Unlock()
	if scraplingServerProcess != nil && scraplingServerProcess.Process != nil {
		scraplingServerProcess.Process.Kill()
		scraplingServerProcess = nil
	}
	scraplingServerReady = false
}

// scraplingFetch fetches a URL using the Scrapling server, returns extracted text.
func scraplingFetch(targetURL string, maxLength int, stealth bool) (string, string, []map[string]string, error) {
	if err := ensureScraplingServer(); err != nil {
		return "", "", nil, err
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"url":        targetURL,
		"max_length": maxLength,
		"stealth":    stealth,
	})

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(scraplingEndpoint+"/fetch", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return "", "", nil, fmt.Errorf("scrapling fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", "", nil, fmt.Errorf("scrapling fetch error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Title string              `json:"title"`
		Text  string              `json:"text"`
		Links []map[string]string `json:"links"`
		URL   string              `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", "", nil, err
	}
	return result.Title, result.Text, result.Links, nil
}

// scraplingScreenshot takes a screenshot of a URL and returns PNG bytes.
func scraplingScreenshot(targetURL string) ([]byte, error) {
	if err := ensureScraplingServer(); err != nil {
		return nil, err
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"url":    targetURL,
		"width":  1280,
		"height": 800,
	})

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(scraplingEndpoint+"/screenshot", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("scrapling screenshot failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("scrapling screenshot error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Image string `json:"image"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return base64.StdEncoding.DecodeString(result.Image)
}

// scraplingSearch performs a web search using the Scrapling server.
func scraplingSearch(query string, maxResults int) ([]map[string]string, error) {
	if err := ensureScraplingServer(); err != nil {
		return nil, err
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"query":       query,
		"max_results": maxResults,
	})

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(scraplingEndpoint+"/search", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("scrapling search failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("scrapling search error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Query   string              `json:"query"`
		Results []map[string]string `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Results, nil
}

// ============================================================================
// Helios Video Generation Server Management
// ============================================================================

var (
	videoServerProcess *exec.Cmd
	videoServerReady   bool
	videoServerMu      sync.Mutex
	videoServerDir     string
)

func initVideoServerDir() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	videoServerDir = filepath.Join(home, ".siki", "video_server")
	if err := os.MkdirAll(videoServerDir, 0755); err != nil {
		return err
	}
	scriptPath := filepath.Join(videoServerDir, "server.py")
	return os.WriteFile(scriptPath, []byte(videoServerScript), 0644)
}

func startVideoServer(config *Config) error {
	videoServerMu.Lock()
	defer videoServerMu.Unlock()

	if videoServerReady {
		return nil
	}

	if err := initVideoServerDir(); err != nil {
		return fmt.Errorf("failed to init video server dir: %w", err)
	}

	scriptPath := filepath.Join(videoServerDir, "server.py")
	endpoint := config.VideoEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8101"
	}
	port := "8101"
	if idx := strings.LastIndex(endpoint, ":"); idx >= 0 {
		port = endpoint[idx+1:]
	}

	modelName := config.VideoModel
	if modelName == "" {
		modelName = "BestWishYSH/Helios-Distilled"
	}

	// Find Python with CUDA torch + diffusers
	pythonPath := "python3"
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, "knowledgeCore", "venv", "bin", "python3"),
		filepath.Join(home, ".egox_venv", "bin", "python"),
	}
	for _, cand := range candidates {
		if _, serr := os.Stat(cand); serr == nil {
			checkCmd := exec.Command(cand, "-c", "import torch, diffusers; assert torch.cuda.is_available()")
			if err := checkCmd.Run(); err == nil {
				pythonPath = cand
				fmt.Printf("[siki] Video server using Python: %s\n", pythonPath)
				break
			}
		}
	}

	cmd := exec.Command(pythonPath, scriptPath)
	cmd.Env = append(os.Environ(),
		"VIDEO_PORT="+port,
		"VIDEO_MODEL="+modelName,
		"VIDEO_OUTPUT_DIR="+playgroundDir,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	fmt.Printf("[siki] Starting Helios video server on port %s...\n", port)
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start video server: %w", err)
	}
	videoServerProcess = cmd

	// Poll for health with timeout (video model takes longer to load)
	healthURL := endpoint + "/health"
	deadline := time.Now().Add(300 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(3 * time.Second)
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				videoServerReady = true
				fmt.Println("[siki] Video server is ready")
				return nil
			}
		}
	}

	stopVideoServer()
	return fmt.Errorf("video server failed to start within 300 seconds")
}

func ensureVideoServer(config *Config) error {
	if videoServerReady {
		return nil
	}

	endpoint := config.VideoEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8101"
	}
	resp, err := http.Get(endpoint + "/health")
	if err == nil {
		resp.Body.Close()
		if resp.StatusCode == 200 {
			videoServerReady = true
			return nil
		}
	}

	fmt.Println("[siki] Starting Helios video server (first use, this may take a while)...")
	return startVideoServer(config)
}

func stopVideoServer() {
	videoServerMu.Lock()
	defer videoServerMu.Unlock()

	if videoServerProcess == nil {
		return
	}

	fmt.Println("[siki] Stopping video server...")
	if videoServerProcess.Process != nil {
		syscall.Kill(-videoServerProcess.Process.Pid, syscall.SIGTERM)
		done := make(chan error, 1)
		go func() { done <- videoServerProcess.Wait() }()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			syscall.Kill(-videoServerProcess.Process.Pid, syscall.SIGKILL)
			<-done
		}
	}
	videoServerProcess = nil
	videoServerReady = false
}

func generateVideo(prompt string, numFrames int, config *Config) (string, error) {
	if err := ensureVideoServer(config); err != nil {
		return "", fmt.Errorf("video server not available: %w", err)
	}

	endpoint := config.VideoEndpoint
	if endpoint == "" {
		endpoint = "http://localhost:8101"
	}

	reqBody, _ := json.Marshal(map[string]interface{}{
		"prompt":     prompt,
		"num_frames": numFrames,
	})

	client := &http.Client{Timeout: 600 * time.Second} // Video generation can be slow
	resp, err := client.Post(endpoint+"/generate", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("video generation request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("video generation failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		VideoBase64 string `json:"video_base64"`
		NumFrames   int    `json:"num_frames"`
		FPS         int    `json:"fps"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode video response: %w", err)
	}

	videoData, err := base64.StdEncoding.DecodeString(result.VideoBase64)
	if err != nil {
		return "", fmt.Errorf("failed to decode video base64: %w", err)
	}

	filename := fmt.Sprintf("video_%d.mp4", time.Now().UnixNano())
	filePath := filepath.Join(playgroundDir, filename)
	if err := os.WriteFile(filePath, videoData, 0644); err != nil {
		return "", fmt.Errorf("failed to save video: %w", err)
	}

	urlPath := "/playground/" + filename
	return urlPath, nil
}

type Thread struct {
	ID           string          `json:"id"`
	Title        string          `json:"title"`
	CreatedAt    time.Time       `json:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at"`
	Messages     []ThreadMessage `json:"messages,omitempty"` // omitempty: metadata JSON has no messages
	MessageCount int             `json:"message_count,omitempty"`
	Summary      string          `json:"summary,omitempty"`
	Unread       bool            `json:"unread,omitempty"`    // true if user hasn't viewed this thread
	Proactive    bool            `json:"proactive,omitempty"` // true if auto-created by siki
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
	EventType  string     `json:"event_type,omitempty"` // display-only: "thinking", "tool_start", "plan_progress", "suggestions"
	Model      string     `json:"model,omitempty"`      // model name for thinking events
}

type ThreadListItem struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
	MessageCount int       `json:"message_count"`
	Summary      string    `json:"summary,omitempty"`
	Unread       bool      `json:"unread,omitempty"`
	Proactive    bool      `json:"proactive,omitempty"`
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
		Unread:       t.Unread,
		Proactive:    t.Proactive,
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
					Unread:       t.Unread,
					Proactive:    t.Proactive,
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
			Unread:       t.Unread,
			Proactive:    t.Proactive,
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

	// Extract keywords from query (split by spaces, filter short words)
	queryLower := strings.ToLower(query)
	words := strings.Fields(queryLower)
	var keywords []string
	for _, w := range words {
		// Skip common particles and short words
		if len(w) >= 3 && w != "して" && w != "ください" && w != "について" && w != "から" && w != "ので" && w != "です" && w != "ます" {
			keywords = append(keywords, w)
		}
	}

	var results []string
	for _, item := range items {
		thread, err := loadThread(item.ID)
		if err != nil {
			continue
		}
		matchCount := 0
		var lastMatch string
		for _, m := range thread.Messages {
			if m.EventType != "" || m.Role == "system" {
				continue
			}
			contentLower := strings.ToLower(m.Content)
			// Match any keyword
			for _, kw := range keywords {
				if strings.Contains(contentLower, kw) {
					matchCount++
					lastMatch = truncateString(m.Content, 200)
					break
				}
			}
		}
		if matchCount > 0 {
			results = append(results, fmt.Sprintf("**Thread: %s** (%s) - %d matches\nLast match: %s", item.Title, item.ID, matchCount, lastMatch))
		}
	}

	// If keyword search found nothing, return summaries of recent threads
	if len(results) == 0 {
		var summaries []string
		limit := 20
		if len(items) < limit {
			limit = len(items)
		}
		for i := 0; i < limit; i++ {
			item := items[i]
			thread, err := loadThread(item.ID)
			if err != nil {
				continue
			}
			// Collect user messages as thread summary
			var userMsgs []string
			for _, m := range thread.Messages {
				if m.Role == "user" && m.Content != "" && m.EventType == "" {
					userMsgs = append(userMsgs, truncateString(m.Content, 100))
				}
			}
			if len(userMsgs) == 0 {
				continue
			}
			summary := strings.Join(userMsgs, " / ")
			if len(summary) > 300 {
				summary = summary[:300] + "..."
			}
			ts := item.UpdatedAt.Format("2006-01-02 15:04")
			summaries = append(summaries, fmt.Sprintf("**%s** (%s) [%s]\nユーザー発言: %s", item.Title, item.ID, ts, summary))
		}
		if len(summaries) == 0 {
			return "過去のスレッドが見つかりません。", nil
		}
		return fmt.Sprintf("最近の%dスレッド一覧（キーワード検索結果なし、代わりに概要を表示）:\n\n%s", len(summaries), strings.Join(summaries, "\n\n---\n")), nil
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
  --sub-model <name>           Set sub-model name (default: gpt-oss:20b)
  --sub-backend <backend>      Set sub-model backend: ollama or vllm (default: ollama)
  --sub-endpoint <url>         Set sub-model endpoint (default: same as main endpoint)

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

// LastToolExecution tracks the previous tool execution for dissatisfaction recovery.
type LastToolExecution struct {
	UserMsg    string
	ToolName   string
	Args       map[string]interface{}
	ToolResult string
	Response   string
	UsedAgent  bool // true if sub-agent was used for summarization
}

type WebServer struct {
	config        *Config
	conversations map[string]*Agent
	mu            sync.RWMutex
	lastActivity  time.Time
	improveMu     sync.Mutex
	lastExec      map[string]*LastToolExecution // keyed by conversation ID
	idleCancel    context.CancelFunc            // cancel autonomous thinking
	idleClients   map[chan StreamEvent]bool      // idle SSE clients
	idleClientMu  sync.Mutex
	lastIdleTask  string // previous idle task name (avoid repeats)
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
	Orchestrator     string     `json:"orchestrator,omitempty"`
	OrchestratorBackend string  `json:"orchestrator_backend,omitempty"`
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
		lastExec:      make(map[string]*LastToolExecution),
		idleClients:   make(map[chan StreamEvent]bool),
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

	// Inject user profile if available
	if profile := loadUserProfile(); profile != nil {
		sb.WriteString("\n\n## ユーザー情報\n")
		if profile.Name != "" {
			sb.WriteString(fmt.Sprintf("- 名前: %s\n", profile.Name))
		}
		if profile.Company != "" {
			sb.WriteString(fmt.Sprintf("- 勤務先: %s\n", profile.Company))
		}
		if profile.Role != "" {
			sb.WriteString(fmt.Sprintf("- 役職: %s\n", profile.Role))
		}
		if profile.Location != "" {
			sb.WriteString(fmt.Sprintf("- 所在地: %s\n", profile.Location))
		}
		if len(profile.Skills) > 0 {
			sb.WriteString(fmt.Sprintf("- スキル: %s\n", strings.Join(profile.Skills, ", ")))
		}
		if len(profile.Clients) > 0 {
			sb.WriteString(fmt.Sprintf("- 取引先: %s\n", strings.Join(profile.Clients, ", ")))
		}
		if profile.Bio != "" {
			sb.WriteString(fmt.Sprintf("- 自己紹介: %s\n", profile.Bio))
		}
		if len(profile.Interests) > 0 {
			sb.WriteString(fmt.Sprintf("- 興味: %s\n", strings.Join(profile.Interests, ", ")))
		}
		if profile.Occupation != "" {
			sb.WriteString(fmt.Sprintf("- 職業: %s\n", profile.Occupation))
		}
		if profile.TechLevel != "" {
			sb.WriteString(fmt.Sprintf("- 技術レベル: %s\n", profile.TechLevel))
		}
		if len(profile.Preferences) > 0 {
			sb.WriteString(fmt.Sprintf("- 傾向: %s\n", strings.Join(profile.Preferences, ", ")))
		}

		// 未入力フィールドを1つだけ自然に聞く指示
		missing := profileMissingFields(profile)
		if len(missing) > 0 {
			// まだ聞いていない & 24時間以上経過 → 1項目だけ質問
			asked := make(map[string]bool)
			for _, f := range profile.AskedFields {
				asked[f] = true
			}
			var toAsk string
			for _, f := range missing {
				if !asked[f] {
					toAsk = f
					break
				}
			}
			if toAsk != "" && time.Since(profile.LastAsked) > 24*time.Hour {
				fieldLabels := map[string]string{
					"name": "名前", "occupation": "職業", "age": "年齢",
					"company": "勤務先", "role": "役職", "location": "所在地",
				}
				label := fieldLabels[toAsk]
				sb.WriteString(fmt.Sprintf("\n\n## プロフィール質問指示\nユーザーの「%s」がまだ不明です。会話の自然な流れの中で、さりげなく聞いてください。押し付けがましくならないように。\n", label))
				// AskedFieldsに追加して保存
				profile.AskedFields = append(profile.AskedFields, toAsk)
				profile.LastAsked = time.Now()
				saveUserProfile(profile)
			}
		}
	}

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

	// Inject installed skills
	skillsMu.RLock()
	if len(loadedSkills) > 0 {
		sb.WriteString("\n\n## Installed Skills\n")
		sb.WriteString("Use `use_skill` tool to activate a skill when appropriate. Skills provide structured workflows for development tasks.\n\n")
		for _, s := range loadedSkills {
			sb.WriteString(fmt.Sprintf("- **%s**: %s\n", s.Name, s.Description))
		}
		sb.WriteString("\nスキルはuse_skillツールで起動できます。ブレインストーミング、デバッグ、コードレビューなどの構造化されたワークフローを提供します。\n")
	}
	skillsMu.RUnlock()

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
			// Skip display-only event messages (not part of LLM context)
			if tm.EventType != "" {
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
	// Only serve index.html for root path
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	content, err := webFS.ReadFile("web/index.html")
	if err != nil {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
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

	// Allow .html, .png, .jpg, .jpeg, .gif, .svg files
	ext := strings.ToLower(filepath.Ext(filename))
	allowedExts := map[string]string{
		".html": "text/html; charset=utf-8",
		".png":  "image/png",
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".gif":  "image/gif",
		".svg":  "image/svg+xml",
		".mp4":  "video/mp4",
		".webm": "video/webm",
	}
	contentType, ok := allowedExts[ext]
	if !ok {
		http.Error(w, "Invalid file type", http.StatusBadRequest)
		return
	}

	filePath := filepath.Join(playgroundDir, filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		http.Error(w, "Playground not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(data)))
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Write(data)
}

// resolvedStaticDir is set at startup to the actual static directory path
var resolvedStaticDir string

func initStaticDir() {
	// Check CWD first (most common dev scenario)
	if cwd, err := os.Getwd(); err == nil {
		dir := filepath.Join(cwd, "static")
		if _, err := os.Stat(dir); err == nil {
			resolvedStaticDir = dir
			fmt.Printf("[siki] Static dir: %s\n", dir)
			return
		}
	}
	// Try next to executable
	if exe, err := os.Executable(); err == nil {
		if real, err := filepath.EvalSymlinks(exe); err == nil {
			dir := filepath.Join(filepath.Dir(real), "static")
			if _, err := os.Stat(dir); err == nil {
				resolvedStaticDir = dir
				fmt.Printf("[siki] Static dir: %s\n", dir)
				return
			}
		}
	}
	// Last resort: home dir
	home, _ := os.UserHomeDir()
	resolvedStaticDir = filepath.Join(home, ".siki", "static")
	fmt.Printf("[siki] Static dir (fallback): %s\n", resolvedStaticDir)
}

func (ws *WebServer) handleJS(w http.ResponseWriter, r *http.Request) {
	// Serve JS/WASM files from static/ directory next to the binary
	filename := strings.TrimPrefix(r.URL.Path, "/js/")
	if filename == "" || strings.Contains(filename, "..") || strings.Contains(filename, "/") {
		http.Error(w, "Invalid filename", http.StatusBadRequest)
		return
	}
	data, err := os.ReadFile(filepath.Join(resolvedStaticDir, filename))
	if err != nil {
		http.Error(w, "Not found", http.StatusNotFound)
		return
	}
	if strings.HasSuffix(filename, ".js") {
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
	} else if strings.HasSuffix(filename, ".wasm") {
		w.Header().Set("Content-Type", "application/wasm")
	}
	w.Header().Set("Cache-Control", "public, max-age=86400")
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

func (ws *WebServer) handleDigestSettings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case http.MethodGet:
		hours := ws.config.DigestHours
		if hours == nil {
			hours = []int{9, 18}
		}
		// Return raw values — type="password" fields hide them in the browser
		json.NewEncoder(w).Encode(map[string]interface{}{
			"email_to":                ws.config.EmailTo,
			"email_from":              ws.config.EmailFrom,
			"smtp_host":               ws.config.SMTPHost,
			"smtp_port":               ws.config.SMTPPort,
			"smtp_user":               ws.config.SMTPUser,
			"digest_enabled":          ws.config.DigestEnabled,
			"digest_hours":            hours,
			"twitter_enabled":         ws.config.TwitterEnabled,
			"twitter_bearer_token":    ws.config.TwitterBearerToken,
			"twitter_consumer_key":    ws.config.TwitterConsumerKey,
			"twitter_consumer_secret": ws.config.TwitterConsumerSecret,
			"twitter_access_token":    ws.config.TwitterAccessToken,
			"twitter_access_secret":   ws.config.TwitterAccessSecret,
			"bluesky_enabled":         ws.config.BlueskyEnabled,
			"bluesky_identifier":      ws.config.BlueskyIdentifier,
			"jetstream_keywords":      ws.config.JetstreamKeywords,
			"brave_api_key":           ws.config.BraveAPIKey,
			"groq_api_key":            ws.config.GroqAPIKey,
		})
	case http.MethodPost:
		var req struct {
			EmailTo               string `json:"email_to"`
			EmailFrom             string `json:"email_from"`
			SMTPHost              string `json:"smtp_host"`
			SMTPPort              int    `json:"smtp_port"`
			SMTPUser              string `json:"smtp_user"`
			SMTPPass              string `json:"smtp_pass"`
			DigestEnabled         bool   `json:"digest_enabled"`
			DigestHours           []int  `json:"digest_hours"`
			TwitterEnabled        bool   `json:"twitter_enabled"`
			TwitterBearerToken    string `json:"twitter_bearer_token"`
			TwitterConsumerKey    string `json:"twitter_consumer_key"`
			TwitterConsumerSecret string `json:"twitter_consumer_secret"`
			TwitterAccessToken    string `json:"twitter_access_token"`
			TwitterAccessSecret   string `json:"twitter_access_secret"`
			BlueskyEnabled        bool     `json:"bluesky_enabled"`
			BlueskyIdentifier     string   `json:"bluesky_identifier"`
			BlueskyAppPassword    string   `json:"bluesky_app_password"`
			JetstreamKeywords     []string `json:"jetstream_keywords"`
			BraveAPIKey           string   `json:"brave_api_key"`
			GroqAPIKey            string   `json:"groq_api_key"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ws.mu.Lock()
		ws.config.EmailTo = req.EmailTo
		ws.config.EmailFrom = req.EmailFrom
		ws.config.SMTPHost = req.SMTPHost
		if req.SMTPPort > 0 {
			ws.config.SMTPPort = req.SMTPPort
		}
		ws.config.SMTPUser = req.SMTPUser
		if req.SMTPPass != "" {
			ws.config.SMTPPass = req.SMTPPass
		}
		ws.config.DigestEnabled = req.DigestEnabled
		if len(req.DigestHours) > 0 {
			ws.config.DigestHours = req.DigestHours
		}
		ws.config.TwitterEnabled = req.TwitterEnabled
		ws.config.TwitterBearerToken = req.TwitterBearerToken
		ws.config.TwitterConsumerKey = req.TwitterConsumerKey
		ws.config.TwitterConsumerSecret = req.TwitterConsumerSecret
		ws.config.TwitterAccessToken = req.TwitterAccessToken
		ws.config.TwitterAccessSecret = req.TwitterAccessSecret
		ws.config.BlueskyEnabled = req.BlueskyEnabled
		if req.BlueskyIdentifier != "" {
			ws.config.BlueskyIdentifier = req.BlueskyIdentifier
		}
		if req.BlueskyAppPassword != "" {
			ws.config.BlueskyAppPassword = req.BlueskyAppPassword
		}
		if req.JetstreamKeywords != nil {
			ws.config.JetstreamKeywords = req.JetstreamKeywords
		}
		if req.BraveAPIKey != "" {
			ws.config.BraveAPIKey = req.BraveAPIKey
			os.Setenv("BRAVE_API_KEY", req.BraveAPIKey)
		}
		if req.GroqAPIKey != "" {
			ws.config.GroqAPIKey = req.GroqAPIKey
			os.Setenv("GROQ_API_KEY", req.GroqAPIKey)
		}
		cachedTwitterUserID = ""
		// Persist to file
		dc := &DigestConfig{
			EmailTo:               ws.config.EmailTo,
			EmailFrom:             ws.config.EmailFrom,
			SMTPHost:              ws.config.SMTPHost,
			SMTPPort:              ws.config.SMTPPort,
			SMTPUser:              ws.config.SMTPUser,
			SMTPPass:              ws.config.SMTPPass,
			DigestEnabled:         ws.config.DigestEnabled,
			DigestHours:           ws.config.DigestHours,
			TwitterBearerToken:    ws.config.TwitterBearerToken,
			TwitterEnabled:        ws.config.TwitterEnabled,
			TwitterConsumerKey:    ws.config.TwitterConsumerKey,
			TwitterConsumerSecret: ws.config.TwitterConsumerSecret,
			TwitterAccessToken:    ws.config.TwitterAccessToken,
			TwitterAccessSecret:   ws.config.TwitterAccessSecret,
			BlueskyEnabled:        ws.config.BlueskyEnabled,
			BlueskyIdentifier:     ws.config.BlueskyIdentifier,
			BlueskyAppPassword:    ws.config.BlueskyAppPassword,
			JetstreamKeywords:     ws.config.JetstreamKeywords,
			BraveAPIKey:           ws.config.BraveAPIKey,
			GroqAPIKey:            ws.config.GroqAPIKey,
		}
		ws.mu.Unlock()
		if err := saveDigestConfig(dc); err != nil {
			fmt.Printf("[siki] Failed to save digest config: %v\n", err)
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleDigestTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if ws.config.EmailTo == "" {
		http.Error(w, "Email address not configured", http.StatusBadRequest)
		return
	}
	go ws.sendDigestEmail()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "sending"})
}

func (ws *WebServer) handleJetstreamKeywords(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch r.Method {
	case http.MethodGet:
		ws.mu.RLock()
		kw := ws.config.JetstreamKeywords
		ws.mu.RUnlock()
		json.NewEncoder(w).Encode(map[string]interface{}{"keywords": kw})
	case http.MethodPost:
		var req struct {
			Keywords []string `json:"keywords"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		// Clean up empty strings
		var clean []string
		for _, kw := range req.Keywords {
			kw = strings.TrimSpace(kw)
			if kw != "" {
				clean = append(clean, kw)
			}
		}
		ws.mu.Lock()
		ws.config.JetstreamKeywords = clean
		ws.mu.Unlock()
		// Persist
		dc := loadDigestConfig()
		if dc == nil {
			dc = &DigestConfig{}
		}
		dc.JetstreamKeywords = clean
		saveDigestConfig(dc)
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "keywords": clean})
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (ws *WebServer) handleJetstreamStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	// Count saved posts for recent days
	type DayStat struct {
		Date  string `json:"date"`
		Count int    `json:"count"`
	}
	var stats []DayStat
	for d := 0; d < 7; d++ {
		t := time.Now().AddDate(0, 0, -d)
		filename := filepath.Join(jetstreamPostsDir(), t.Format("2006"), t.Format("0102")+".txt")
		count := 0
		if data, err := os.ReadFile(filename); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if line != "" {
					count++
				}
			}
		}
		stats = append(stats, DayStat{Date: t.Format("2006-01-02"), Count: count})
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"stats": stats})
}

func (ws *WebServer) handleJetstreamPosts(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	query := r.URL.Query().Get("q")
	days := 1
	if d, err := strconv.Atoi(r.URL.Query().Get("days")); err == nil && d > 0 {
		days = d
	}
	limit := 200
	if l, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && l > 0 {
		limit = l
	}

	type Post struct {
		Time       string   `json:"time"`
		DID        string   `json:"did"`
		RKey       string   `json:"rkey"`
		Keywords   []string `json:"keywords"`
		Text       string   `json:"text"`
		OGPTitle   string   `json:"ogp_title,omitempty"`
		OGPDesc    string   `json:"ogp_desc,omitempty"`
		OGPImage   string   `json:"ogp_image,omitempty"`
		URL        string   `json:"url,omitempty"`
		Evaluation string   `json:"evaluation,omitempty"`
		Score      int      `json:"score,omitempty"`
	}

	queryLower := strings.ToLower(query)
	var posts []Post
	for d := 0; d < days; d++ {
		t := time.Now().AddDate(0, 0, -d)
		filename := filepath.Join(jetstreamPostsDir(), t.Format("2006"), t.Format("0102")+".txt")
		data, err := os.ReadFile(filename)
		if err != nil {
			continue
		}
		meta := loadJetstreamMeta(t)
		lines := strings.Split(string(data), "\n")
		// Read in reverse (newest first)
		for i := len(lines) - 1; i >= 0; i-- {
			line := lines[i]
			if line == "" {
				continue
			}
			if query != "" && !strings.Contains(strings.ToLower(line), queryLower) {
				continue
			}
			parts := strings.SplitN(line, "\t", 5)
			if len(parts) < 5 {
				continue
			}
			kwStr := strings.Trim(parts[3], "[]")
			var kws []string
			if kwStr != "" {
				kws = strings.Split(kwStr, ",")
			}
			p := Post{
				Time:     parts[0],
				DID:      parts[1],
				RKey:     parts[2],
				Keywords: kws,
				Text:     parts[4],
			}
			// Attach metadata if available
			key := parts[1] + ":" + parts[2]
			if m, ok := meta[key]; ok {
				p.OGPTitle = m.OGPTitle
				p.OGPDesc = m.OGPDesc
				p.OGPImage = m.OGPImage
				p.URL = m.URL
				p.Evaluation = m.Evaluation
				p.Score = m.Score
			}
			posts = append(posts, p)
			if len(posts) >= limit {
				break
			}
		}
		if len(posts) >= limit {
			break
		}
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"posts": posts, "count": len(posts)})
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
			// List threads with optional type filter: ?type=user|autonomous|all (default: all)
			items, err := listThreads()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if items == nil {
				items = []ThreadListItem{}
			}
			typeFilter := r.URL.Query().Get("type")
			if typeFilter == "user" {
				var filtered []ThreadListItem
				for _, t := range items {
					if !strings.HasPrefix(t.ID, idleThreadIDPrefix) && !strings.HasPrefix(t.ID, proactiveThreadIDPrefix) {
						filtered = append(filtered, t)
					}
				}
				if filtered == nil {
					filtered = []ThreadListItem{}
				}
				items = filtered
			} else if typeFilter == "autonomous" {
				var filtered []ThreadListItem
				for _, t := range items {
					if strings.HasPrefix(t.ID, idleThreadIDPrefix) || strings.HasPrefix(t.ID, proactiveThreadIDPrefix) {
						filtered = append(filtered, t)
					}
				}
				if filtered == nil {
					filtered = []ThreadListItem{}
				}
				items = filtered
			}
			// Apply limit (default 100) to prevent heavy responses
			limitStr := r.URL.Query().Get("limit")
			limit := 100
			if limitStr != "" {
				if n, err := strconv.Atoi(limitStr); err == nil && n > 0 {
					limit = n
				}
			}
			// Sort by updated_at descending (most recent first)
			sort.Slice(items, func(i, j int) bool {
				return items[i].UpdatedAt.After(items[j].UpdatedAt)
			})
			if len(items) > limit {
				items = items[:limit]
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
			// Mark as read when user views the thread
			if t.Unread {
				t.Unread = false
				saveThreadMeta(t)
			}
			// Trim excessive messages to prevent browser freeze
			{
				var trimmedMsgs []ThreadMessage
				thinkingCount := 0
				for _, m := range t.Messages {
					if m.EventType == "thinking" {
						thinkingCount++
						if thinkingCount > 20 {
							continue
						}
					}
					// Truncate large content in tool results and tool_call events
					if m.EventType == "tool_call" || m.Role == "tool" {
						runes := []rune(m.Content)
						if len(runes) > 3000 {
							m.Content = string(runes[:3000]) + "\n...(truncated)"
						}
					}
					trimmedMsgs = append(trimmedMsgs, m)
				}
				t.Messages = trimmedMsgs
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
		Model:               pp.Model,
		Backend:             pp.Backend,
		Endpoint:            pp.Endpoint,
		Version:             Version,
		HasAPIKey:           pp.APIKey != "",
		Providers:           ws.config.Providers,
		DockerAvailable:     isDockerAvailable(),
		DockerImageReady:    isDockerAvailable() && isDockerImageAvailable(),
		VisionModel:         ws.config.VisionModel,
		Orchestrator:        ws.config.orchestratorModel(),
		OrchestratorBackend: ws.config.orchestratorBackend(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleOrchestrator allows GET (current orchestrator info) and POST (change orchestrator model).
func (ws *WebServer) handleOrchestrator(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method == http.MethodGet {
		ws.mu.RLock()
		info := map[string]string{
			"model":    ws.config.orchestratorModel(),
			"backend":  ws.config.orchestratorBackend(),
			"endpoint": ws.config.orchestratorEndpoint(),
		}
		ws.mu.RUnlock()
		json.NewEncoder(w).Encode(info)
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Model    string `json:"model"`
		Backend  string `json:"backend"`
		Endpoint string `json:"endpoint"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	ws.mu.Lock()
	if req.Model != "" {
		ws.config.Orchestrator = req.Model
	}
	if req.Backend != "" {
		ws.config.OrchestratorBackend = req.Backend
	}
	if req.Endpoint != "" {
		ws.config.OrchestratorEndpoint = req.Endpoint
	}
	newModel := ws.config.orchestratorModel()
	newBackend := ws.config.orchestratorBackend()
	ws.mu.Unlock()

	fmt.Printf("[siki] Orchestrator changed to: %s (backend: %s)\n", newModel, newBackend)

	// Pre-warm the new orchestrator model
	if newBackend == "ollama" {
		go func() {
			fmt.Printf("[siki] Pre-warming orchestrator: %s ...\n", newModel)
			start := time.Now()
			_, err := callOrchestratorGenerate("Say OK.", 5, 600*time.Second, ws.config)
			if err != nil {
				fmt.Printf("[siki] Orchestrator warm-up failed: %v\n", err)
			} else {
				fmt.Printf("[siki] Orchestrator ready (%.1fs)\n", time.Since(start).Seconds())
			}
		}()
	}

	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"model":  newModel,
	})
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

			// Override args for small orchestrator models
			userReq := agent.lastUserMessage()
			args = overrideToolArgs(toolName, userReq, args, ws.config, nil)

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
	Type        string   `json:"type"`
	Content     string   `json:"content,omitempty"`
	Name        string   `json:"name,omitempty"`
	Result      string   `json:"result,omitempty"`
	Error       string   `json:"error,omitempty"`
	Suggestions []string `json:"suggestions,omitempty"`
	Model       string   `json:"model,omitempty"`
}

// modelThinkingEvent creates a thinking StreamEvent with the appropriate model name.
func modelThinkingEvent(content string, config *Config, useSubAgent bool) StreamEvent {
	model := ""
	if useSubAgent && hasSubAgent(config) {
		model = config.SubAgent
	} else if config.SubModel != "" {
		model = config.SubModel
	}
	return StreamEvent{Type: "thinking", Content: content, Model: model}
}

// orchestratorThinkingEvent creates a thinking event showing the orchestrator model name.
func orchestratorThinkingEvent(content string, config *Config) StreamEvent {
	return StreamEvent{Type: "thinking", Content: content, Model: config.orchestratorModel()}
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
		_, err := fmt.Fprintf(w, "data: %s\n\n", data)
		if err != nil {
			fmt.Printf("[siki] SSE write error: %v\n", err)
			return
		}
		flusher.Flush()
	}

	// Track activity for self-improvement idle detection
	ws.mu.Lock()
	ws.lastActivity = time.Now()
	// Cancel autonomous thinking immediately
	if ws.idleCancel != nil {
		ws.idleCancel()
		ws.idleCancel = nil
	}
	ws.mu.Unlock()
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_interrupted"})

	agent := ws.getOrCreateAgent(req.ConversationID)

	// Helper: save a message to thread log immediately
	threadID := req.ConversationID

	// Accumulate thinking and content for dedup with saved messages
	var thinkingBuf strings.Builder
	var contentBuf strings.Builder

	// flushContentBuf saves accumulated content as an event_type message
	// (for intermediate content like quick ack that precedes tool/thinking events)
	flushContentBuf := func() {
		if contentBuf.Len() > 0 {
			text := strings.TrimSpace(contentBuf.String())
			if text != "" {
				appendToLog(threadID, ThreadMessage{
					EventType: "content",
					Role:      "assistant",
					Content:   text,
					Timestamp: time.Now().Unix(),
				})
			}
			contentBuf.Reset()
		}
	}

	// Wrap sendEvent to persist ALL events for thread replay
	origSendEvent := sendEvent
	sendEvent = func(event StreamEvent) {
		origSendEvent(event)
		switch event.Type {
		case "content":
			// Accumulate content; will be flushed when non-content event arrives
			contentBuf.WriteString(event.Content)
		case "thinking":
			// Flush any preceding content (e.g. quick ack) before thinking
			flushContentBuf()
			if event.Content != "" {
				thinkingBuf.WriteString(event.Content)
				appendToLog(threadID, ThreadMessage{
					EventType: "thinking",
					Role:      "assistant",
					Content:   event.Content,
					Model:     event.Model,
					Timestamp: time.Now().Unix(),
				})
			}
		case "tool_start":
			flushContentBuf()
			appendToLog(threadID, ThreadMessage{
				EventType: "tool_start",
				Role:      "assistant",
				ToolName:  event.Name,
				Content:   event.Name,
				Timestamp: time.Now().Unix(),
			})
		case "tool_call":
			flushContentBuf()
			appendToLog(threadID, ThreadMessage{
				EventType: "tool_call",
				Role:      "assistant",
				ToolName:  event.Name,
				Content:   event.Result,
				Timestamp: time.Now().Unix(),
			})
		case "plan_progress":
			flushContentBuf()
			appendToLog(threadID, ThreadMessage{
				EventType: "plan_progress",
				Role:      "assistant",
				Content:   event.Content,
				Timestamp: time.Now().Unix(),
			})
		case "suggestions":
			flushContentBuf()
			if len(event.Suggestions) > 0 {
				data, _ := json.Marshal(event.Suggestions)
				appendToLog(threadID, ThreadMessage{
					EventType: "suggestions",
					Role:      "assistant",
					Content:   string(data),
					Timestamp: time.Now().Unix(),
				})
			}
		}
	}

	// Wrap saveMsg to clear duplicates (already saved as events)
	saveMsg := func(msg Message, toolName string) {
		if msg.Role == "assistant" {
			// Thinking was already saved as event_type messages
			if thinkingBuf.Len() > 0 {
				msg.Thinking = ""
				thinkingBuf.Reset()
			}
			// Content was streamed; clear contentBuf (the message has the final content)
			contentBuf.Reset()
		}
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
		sendEvent(StreamEvent{Type: "thinking", Content: "画像を解析中...", Model: ws.config.VisionModel})
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

	// 10 minutes for the full pipeline — sub-model can be slow on first load
	ctx, cancel := context.WithTimeout(context.Background(), 600*time.Second)
	var hitTimeout bool

	if ws.config.SubModel != "" {
		// Dual-model pipeline: lfm (quick ack) + gpt-oss (real orchestration)
		lastAssistantReply = ws.dualModelPipeline(ctx, agent, req.Message, sendEvent, saveMsg, req.ConversationID)
	} else {
		// Fallback: single model agent loop (lfm only, with tool overrides)
		for turn := 0; turn < ws.config.MaxTurns; turn++ {
			response, err := agent.chatStream(ctx, StreamCallbacks{
				OnContent: func(content string) {
					sendEvent(StreamEvent{Type: "content", Content: content})
				},
				OnThinking: func(thinking string) {
					sendEvent(StreamEvent{Type: "thinking", Content: thinking, Model: ws.config.ModelName})
				},
			})

			if err != nil {
				if response != nil && (response.Content != "" || len(response.ToolCalls) > 0) {
					agent.messages = append(agent.messages, *response)
					saveMsg(*response, "")
					for _, tc := range response.ToolCalls {
						placeholder := Message{
							Role:       "tool",
							Content:    fmt.Sprintf("[%s の結果は取得できませんでした]", tc.Function.Name),
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
				break
			}

			agent.messages = append(agent.messages, *response)
			saveMsg(*response, "")

			if len(response.ToolCalls) == 0 {
				if turn == 0 {
					if fallbackResult := autoToolFallback(agent, req.Message, response.Content, sendEvent, saveMsg); fallbackResult != "" {
						lastAssistantReply = fallbackResult
						break
					}
				}
				lastAssistantReply = response.Content
				break
			}

			for _, tc := range response.ToolCalls {
				toolName := tc.Function.Name
				if toolName == "diagram" && needsRunCode(agent.lastUserMessage()) {
					toolName = "run_code"
				}
				sendEvent(StreamEvent{Type: "tool_start", Name: toolName})

				var args map[string]interface{}
				if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
					sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: fmt.Sprintf("Error: %v", err)})
					agent.messages = append(agent.messages, Message{Role: "tool", Content: fmt.Sprintf("Error: %v", err), ToolCallID: tc.ID})
					continue
				}
				args = overrideToolArgs(toolName, agent.lastUserMessage(), args, ws.config, sendEvent)

				result, err := agent.executeTool(toolName, args)
				if err != nil {
					result = fmt.Sprintf("Error: %v", err)
				}
				displayResult := result
				if len(displayResult) > 2000 {
					displayResult = displayResult[:2000] + "\n... (truncated)"
				}
				sendEvent(StreamEvent{Type: "tool_call", Name: toolName, Result: displayResult})

				toolMsg := Message{Role: "tool", Content: result, ToolCallID: tc.ID}
				agent.messages = append(agent.messages, toolMsg)
				saveMsg(toolMsg, toolName)
			}
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
	// Load persisted digest settings
	applyDigestConfig(config)

	// Apply Zeroboot config
	if config.ZerobootEndpoint != "" {
		zerobootEndpoint = config.ZerobootEndpoint
	}
	if config.ZerobootAPIKey != "" {
		zerobootAPIKey = config.ZerobootAPIKey
	}

	ws := NewWebServer(config)

	initStaticDir()

	http.HandleFunc("/", ws.handleIndex)
	http.HandleFunc("/api/status", ws.handleStatus)
	http.HandleFunc("/api/settings", ws.handleSettings)
	http.HandleFunc("/api/chat", ws.handleChat)
	http.HandleFunc("/api/chat/stream", ws.handleChatStream)
	http.HandleFunc("/api/images", ws.handleImages)
	http.HandleFunc("/js/", ws.handleJS)
	http.HandleFunc("/diagrams/", ws.handleDiagrams)
	http.HandleFunc("/playground/", ws.handlePlayground)
	http.HandleFunc("/api/plugins/ui", ws.handlePluginsUI)
	http.HandleFunc("/api/plugins", ws.handlePlugins)
	http.HandleFunc("/api/threads/", ws.handleThreads)
	http.HandleFunc("/api/threads", ws.handleThreads)
	http.HandleFunc("/api/docker/exec", ws.handleDockerExec)
	http.HandleFunc("/api/docker/status", ws.handleDockerStatus)
	http.HandleFunc("/api/upload", ws.handleUpload)
	http.HandleFunc("/api/idle/stream", ws.handleIdleStream)
	http.HandleFunc("/api/orchestrator", ws.handleOrchestrator)
	http.HandleFunc("/api/digest/settings", ws.handleDigestSettings)
	http.HandleFunc("/api/digest/test", ws.handleDigestTest)
	http.HandleFunc("/api/profile", ws.handleProfile)
	http.HandleFunc("/api/jetstream/keywords", ws.handleJetstreamKeywords)
	http.HandleFunc("/api/jetstream/stats", ws.handleJetstreamStats)
	http.HandleFunc("/api/jetstream/posts", ws.handleJetstreamPosts)
	http.HandleFunc("/api/jetstream/analyze", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		go ws.analyzeJetstreamPosts()
		json.NewEncoder(w).Encode(map[string]string{"status": "started"})
	})

	// Skills API
	http.HandleFunc("/api/skills", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		skillsMu.RLock()
		defer skillsMu.RUnlock()
		// Return skill list (without full content for lighter response)
		type skillInfo struct {
			Name        string   `json:"name"`
			Description string   `json:"description"`
			Files       []string `json:"files,omitempty"`
			Source      string   `json:"source,omitempty"`
		}
		var list []skillInfo
		for _, s := range loadedSkills {
			list = append(list, skillInfo{
				Name:        s.Name,
				Description: s.Description,
				Files:       s.Files,
				Source:       s.Source,
			})
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"skills": list, "count": len(list)})
	})

	http.HandleFunc("/api/skills/install", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST required", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")

		var req struct {
			RepoURL string `json:"repo_url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.RepoURL == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "repo_url required"})
			return
		}

		// Clone repo to temp dir
		tmpDir, err := os.MkdirTemp("", "siki-skills-*")
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		defer os.RemoveAll(tmpDir)

		cmd := exec.Command("git", "clone", "--depth", "1", req.RepoURL, tmpDir)
		if out, err := cmd.CombinedOutput(); err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": fmt.Sprintf("git clone failed: %s", string(out))})
			return
		}

		// Detect skill directory (look for skills/ subdir)
		skillSrcDir := filepath.Join(tmpDir, "skills")
		if _, err := os.Stat(skillSrcDir); err != nil {
			// Maybe skills are at root level
			skillSrcDir = tmpDir
		}

		// Detect source name from repo URL
		source := filepath.Base(strings.TrimSuffix(req.RepoURL, ".git"))

		count, err := installSkillsFromDir(skillSrcDir, source)
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "installed",
			"count":     count,
			"source":    source,
		})
	})

	http.HandleFunc("/api/skills/delete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "POST required", 405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		var req struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Name == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "name required"})
			return
		}
		skillDir := filepath.Join(skillsDir(), req.Name)
		if err := os.RemoveAll(skillDir); err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		skillsMu.Lock()
		loadedSkills = loadSkills()
		skillsMu.Unlock()
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "name": req.Name})
	})

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
	// Load skills
	os.MkdirAll(skillsDir(), 0755)
	skillsMu.Lock()
	loadedSkills = loadSkills()
	skillsMu.Unlock()
	fmt.Printf("[siki] Loaded %d skills\n", len(loadedSkills))
	if err := initDockerWorkspace(); err != nil {
		fmt.Printf("Warning: failed to initialize docker workspace: %v\n", err)
	}
	// Initialize image server directory (write server.py)
	if config.ImageEnabled {
		if err := initImageServerDir(); err != nil {
			fmt.Printf("Warning: failed to initialize image server dir: %v\n", err)
		}
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
		fmt.Printf("Sub Model: %s (%s)\n", config.SubModel, config.SubModelBackend)
	}
	if config.SubAgent != "" {
		backend := config.SubAgentBackend
		if backend == "" {
			backend = "vllm"
		}
		endpoint := config.SubAgentEndpoint
		if endpoint == "" {
			endpoint = "(same as sub-model)"
		}
		fmt.Printf("Sub Agent: %s (%s, %s)\n", config.SubAgent, backend, endpoint)
	}
	// Orchestrator model info
	orchModel := config.orchestratorModel()
	orchBackend := config.orchestratorBackend()
	if config.Orchestrator != "" {
		fmt.Printf("Orchestrator: %s (%s)\n", orchModel, orchBackend)
	} else {
		fmt.Printf("Orchestrator: %s (%s) [= sub-model]\n", orchModel, orchBackend)
	}
	// Image generation status
	if config.ImageEnabled {
		vram := detectVRAM()
		imgModel := config.ImageModel
		if imgModel == "" {
			imgModel = "Flux Klein 4B"
		}
		if canRunImageServer() {
			fmt.Printf("Image Generation: %s (available, lazy load, VRAM: %dMB free)\n", imgModel, vram)
		} else if vram > 0 {
			fmt.Printf("Image Generation: %s (insufficient VRAM: %dMB free, need 8000MB)\n", imgModel, vram)
		} else {
			fmt.Printf("Image Generation: disabled (no GPU or python3 not found)\n")
		}
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

	// Pre-warm sub-model (gpt-oss) in background so first request is fast
	preWarmSubModel(config)

	// Start self-improvement background loop
	go ws.selfImproveLoop()

	// Start email digest loop
	go ws.digestLoop()

	// Start Bluesky feed background loop
	go ws.blueskyFeedLoop()

	// Start Bluesky Jetstream monitoring loop
	go ws.jetstreamLoop()

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
		stopImageServer()
		stopVideoServer()
		stopScraplingServer()
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
		stopImageServer()
		stopVideoServer()
		stopScraplingServer()
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
	webHost := "0.0.0.0"

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
		case "--sub-backend":
			if i+1 < len(args) {
				config.SubModelBackend = args[i+1]
				i += 2
				continue
			}
		case "--sub-endpoint":
			if i+1 < len(args) {
				config.SubModelEndpoint = args[i+1]
				i += 2
				continue
			}
		case "--sub-agent":
			if i+1 < len(args) {
				config.SubAgent = args[i+1]
				i += 2
				continue
			}
		case "--sub-agent-backend":
			if i+1 < len(args) {
				config.SubAgentBackend = args[i+1]
				i += 2
				continue
			}
		case "--sub-agent-endpoint":
			if i+1 < len(args) {
				config.SubAgentEndpoint = args[i+1]
				i += 2
				continue
			}
		case "--orchestrator":
			if i+1 < len(args) {
				config.Orchestrator = args[i+1]
				i += 2
				continue
			}
		case "--orchestrator-backend":
			if i+1 < len(args) {
				config.OrchestratorBackend = args[i+1]
				i += 2
				continue
			}
		case "--orchestrator-endpoint":
			if i+1 < len(args) {
				config.OrchestratorEndpoint = args[i+1]
				i += 2
				continue
			}
		case "-h", "--help":
			printHelp()
			return
		default:
			// Non-flag argument (subcommand like "web", "chat") — just skip and keep parsing
			i++
			continue
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

	// Get remaining args: collect non-flag arguments, skipping flag values
	var remaining []string
	flagsWithValue := map[string]bool{
		"--backend": true, "--api-key": true, "--model": true, "--endpoint": true,
		"--vision-model": true, "--sub-model": true, "--sub-backend": true,
		"--sub-endpoint": true, "--sub-agent": true, "--sub-agent-endpoint": true,
		"--orchestrator": true, "--orchestrator-backend": true, "--orchestrator-endpoint": true,
		"--port": true, "--host": true, "--workspace": true,
	}
	for j := 0; j < len(args); j++ {
		if strings.HasPrefix(args[j], "-") {
			if flagsWithValue[args[j]] && j+1 < len(args) {
				j++ // skip the value
			}
			continue
		}
		remaining = append(remaining, args[j])
	}

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
