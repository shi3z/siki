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

type Config struct {
	ModelPath    string `json:"model_path"`
	ModelName    string `json:"model_name"`
	Backend      string `json:"backend"` // vllm, mlx, ollama
	APIEndpoint  string `json:"api_endpoint"`
	Workspace    string `json:"workspace"`
	MaxTurns     int    `json:"max_turns"`
	SystemPrompt string `json:"system_prompt"`
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

Available tools:
- read_file, write_file, list_files: File operations
- execute_command: Run shell commands
- search_files, grep: Search code and files
- web_search: Search the internet
- web_fetch: Get text content from a URL
- web_images: Extract representative images (OGP, main images) from a URL
- diagram: Generate diagrams using Graphviz DOT language

IMPORTANT RULES:
1. Always use tools when needed.
2. For current events/news, use web_search.
3. When mentioning URLs with visual content, use web_images to show representative images. The tool returns markdown image syntax - include it directly in your response.
4. Use the diagram tool to visualize relationships, workflows, architectures, or research results. The tool returns markdown image syntax - include it directly in your response.
5. Respond in the user's language.`,
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
}

// ============================================================================
// Tool Execution
// ============================================================================

type Agent struct {
	config   *Config
	messages []Message
}

func (a *Agent) executeTool(name string, args map[string]interface{}) (string, error) {
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
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
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
	cmd := exec.Command("dot", "-Tsvg", "-o", outputPath, tmpFile.Name())
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("graphviz error: %s - %w", string(output), err)
	}

	// Return markdown image syntax
	diagramURL := fmt.Sprintf("/diagrams/%s", filename)
	return fmt.Sprintf("Diagram generated:\n\n![%s](%s)", title, diagramURL), nil
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
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
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

	req, _ := http.NewRequestWithContext(ctx, "GET", a.config.APIEndpoint+"/models", nil)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("server not available at %s", a.config.APIEndpoint)
	}
	resp.Body.Close()
	return nil
}

func (a *Agent) chat(ctx context.Context) (*Message, error) {
	return a.chatStream(ctx, nil)
}

func (a *Agent) chatStream(ctx context.Context, onContent func(string)) (*Message, error) {
	// Convert tools to OpenAI format
	var toolDefs []map[string]interface{}
	for _, tool := range tools {
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
		Model:       a.config.ModelName,
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
		a.config.APIEndpoint+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

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
  --backend <vllm|mlx|ollama>  Set LLM backend
  --model <name>               Set model name
  --endpoint <url>             Set API endpoint
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
	Message        string `json:"message"`
	ConversationID string `json:"conversation_id"`
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
	Model    string `json:"model"`
	Backend  string `json:"backend"`
	Endpoint string `json:"endpoint"`
	Version  string `json:"version"`
}

type SettingsRequest struct {
	Model    string `json:"model"`
	Backend  string `json:"backend"`
	Endpoint string `json:"endpoint"`
}

func NewWebServer(config *Config) *WebServer {
	return &WebServer{
		config:        config,
		conversations: make(map[string]*Agent),
	}
}

func (ws *WebServer) getOrCreateAgent(convID string) *Agent {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if agent, exists := ws.conversations[convID]; exists {
		return agent
	}

	agent := &Agent{
		config: ws.config,
		messages: []Message{
			{Role: "system", Content: ws.config.SystemPrompt},
		},
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

func (ws *WebServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	resp := StatusResponse{
		Model:    ws.config.ModelName,
		Backend:  ws.config.Backend,
		Endpoint: ws.config.APIEndpoint,
		Version:  Version,
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
	if req.Model != "" {
		ws.config.ModelName = req.Model
	}
	if req.Backend != "" {
		ws.config.Backend = req.Backend
	}
	if req.Endpoint != "" {
		ws.config.APIEndpoint = req.Endpoint
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

	// Add user message
	agent.messages = append(agent.messages, Message{
		Role:    "user",
		Content: req.Message,
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

	// Add user message
	agent.messages = append(agent.messages, Message{
		Role:    "user",
		Content: req.Message,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Agent loop
	for turn := 0; turn < ws.config.MaxTurns; turn++ {
		// Try streaming first
		response, err := agent.chatStream(ctx, func(content string) {
			sendEvent(StreamEvent{Type: "content", Content: content})
		})

		if err != nil {
			sendEvent(StreamEvent{Type: "error", Error: err.Error()})
			return
		}

		agent.messages = append(agent.messages, *response)

		if len(response.ToolCalls) == 0 {
			// If no streaming happened, send the full content
			if response.Content != "" {
				sendEvent(StreamEvent{Type: "done"})
			}
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
	http.HandleFunc("/diagrams/", ws.handleDiagrams)

	// Initialize diagram directory
	if err := initDiagramDir(); err != nil {
		fmt.Printf("Warning: failed to initialize diagram dir: %v\n", err)
	}

	fmt.Printf("siki v%s - 式神 Web GUI\n", Version)
	fmt.Printf("Backend: %s, Model: %s\n", config.Backend, config.ModelName)
	fmt.Printf("API Endpoint: %s\n", config.APIEndpoint)

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
			{Role: "system", Content: config.SystemPrompt},
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
	fmt.Printf("Backend: %s, Model: %s, Endpoint: %s\n\n",
		config.Backend, config.ModelName, config.APIEndpoint)

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
				{Role: "system", Content: config.SystemPrompt},
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
				// Update endpoint based on backend if not explicitly set
				if !endpointOverridden {
					if args[i+1] == "ollama" {
						config.APIEndpoint = "http://localhost:11434/v1"
					} else {
						config.APIEndpoint = "http://localhost:8000/v1"
					}
				}
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
