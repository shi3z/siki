package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// ============================================================================
// Test Helpers
// ============================================================================

// setupTestDirs overrides global directory variables to use temporary directories.
// Returns a cleanup function that restores original values.
// Tests using this MUST NOT use t.Parallel().
func setupTestDirs(t *testing.T) func() {
	t.Helper()

	origThreadDir := threadDir
	origPluginDir := pluginDir
	origPlaygroundDir := playgroundDir
	origDiagramDir := diagramDir
	origDockerWorkspaceDir := dockerWorkspaceDir
	origLoadedPlugins := loadedPlugins

	tmp := t.TempDir()
	threadDir = filepath.Join(tmp, "threads")
	pluginDir = filepath.Join(tmp, "plugins")
	playgroundDir = filepath.Join(tmp, "playground")
	diagramDir = filepath.Join(tmp, "diagrams")
	dockerWorkspaceDir = filepath.Join(tmp, "workspace")

	os.MkdirAll(threadDir, 0755)
	os.MkdirAll(pluginDir, 0755)
	os.MkdirAll(playgroundDir, 0755)
	os.MkdirAll(diagramDir, 0755)
	os.MkdirAll(dockerWorkspaceDir, 0755)

	loadedPlugins = nil

	return func() {
		threadDir = origThreadDir
		pluginDir = origPluginDir
		playgroundDir = origPlaygroundDir
		diagramDir = origDiagramDir
		dockerWorkspaceDir = origDockerWorkspaceDir
		loadedPlugins = origLoadedPlugins
	}
}

// testConfig creates a Config pointing at the given mock server URL.
func testConfig(serverURL string) *Config {
	endpoint := serverURL + "/v1"
	return &Config{
		ModelName:   "test-model",
		Backend:     "ollama",
		APIEndpoint: endpoint,
		Workspace:   os.TempDir(),
		MaxTurns:    5,
		VisionModel: "test-vision",
		Providers: []Provider{{
			Name:     "default",
			Backend:  "ollama",
			Endpoint: endpoint,
			Model:    "test-model",
		}},
		SystemPrompt: "You are a test assistant.",
	}
}

// mockLLMServer starts an httptest.Server with the given handler for chat completions.
func mockLLMServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handler)
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"data":[{"id":"test-model"}]}`)
	})
	mux.HandleFunc("/api/chat", handler) // Ollama native API for vision
	return httptest.NewServer(mux)
}

// staticLLMResponse returns a handler that always returns a non-streaming ChatResponse.
func staticLLMResponse(content string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":    "test-resp",
			"model": "test-model",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]string{"role": "assistant", "content": content},
					"finish_reason": "stop",
				},
			},
		})
	}
}

// streamingLLMResponse returns a handler that sends SSE chunks, then [DONE].
func streamingLLMResponse(chunks []string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req ChatRequest
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &req)

		if !req.Stream {
			staticLLMResponse(strings.Join(chunks, ""))(w, r)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "no flusher", 500)
			return
		}
		for i, chunk := range chunks {
			role := ""
			if i == 0 {
				role = "assistant"
			}
			data := map[string]interface{}{
				"id":    "test-stream",
				"model": "test-model",
				"choices": []map[string]interface{}{
					{
						"index": 0,
						"delta": map[string]string{"role": role, "content": chunk},
					},
				},
			}
			j, _ := json.Marshal(data)
			fmt.Fprintf(w, "data: %s\n\n", j)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}
}

// toolCallLLMResponse returns a handler that sends a tool call response.
func toolCallLLMResponse(toolName, toolArgs, toolID string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":    "test-tc",
			"model": "test-model",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "",
						"tool_calls": []map[string]interface{}{
							{
								"id":   toolID,
								"type": "function",
								"function": map[string]string{
									"name":      toolName,
									"arguments": toolArgs,
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		})
	}
}

func skipIfNoNode(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("node"); err != nil {
		t.Skip("node not available, skipping")
	}
}

func skipIfNoDocker(t *testing.T) {
	t.Helper()
	if !isDockerAvailable() {
		t.Skip("docker not available, skipping")
	}
}

// flushRecorder wraps httptest.ResponseRecorder to satisfy http.Flusher.
type flushRecorder struct {
	*httptest.ResponseRecorder
}

func (f *flushRecorder) Flush() {}

func newFlushRecorder() *flushRecorder {
	return &flushRecorder{httptest.NewRecorder()}
}

// ============================================================================
// 1. Pure Function Tests — Config
// ============================================================================

func TestPrimaryProvider_WithProviders(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Providers: []Provider{
			{Name: "p1", Backend: "ollama", Model: "m1"},
			{Name: "p2", Backend: "openai", Model: "m2"},
		},
	}
	p := cfg.primaryProvider()
	if p.Name != "p1" || p.Model != "m1" {
		t.Errorf("expected p1/m1, got %s/%s", p.Name, p.Model)
	}
}

func TestPrimaryProvider_LegacyFields(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Backend:     "vllm",
		APIEndpoint: "http://localhost:8000/v1",
		ModelName:   "legacy-model",
		APIKey:      "key123",
	}
	p := cfg.primaryProvider()
	if p.Backend != "vllm" || p.Model != "legacy-model" || p.APIKey != "key123" {
		t.Errorf("unexpected provider: %+v", p)
	}
}

func TestFindProvider(t *testing.T) {
	t.Parallel()
	cfg := &Config{
		Providers: []Provider{
			{Name: "Alpha", Backend: "ollama"},
			{Name: "Beta", Backend: "openai"},
		},
	}
	// Case insensitive
	p := cfg.findProvider("alpha")
	if p == nil || p.Name != "Alpha" {
		t.Errorf("expected Alpha, got %v", p)
	}
	p = cfg.findProvider("BETA")
	if p == nil || p.Name != "Beta" {
		t.Errorf("expected Beta, got %v", p)
	}
	p = cfg.findProvider("gamma")
	if p != nil {
		t.Errorf("expected nil, got %v", p)
	}
}

func TestDefaultEndpointForBackend(t *testing.T) {
	t.Parallel()
	tests := []struct {
		backend  string
		expected string
	}{
		{"ollama", "http://localhost:11434/v1"},
		{"vllm", "http://localhost:8000/v1"},
		{"mlx", "http://localhost:8000/v1"},
		{"openai", "https://api.openai.com/v1"},
		{"anthropic", "https://api.anthropic.com/v1"},
		{"gemini", "https://generativelanguage.googleapis.com/v1beta/openai"},
		{"unknown", "http://localhost:11434/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.backend, func(t *testing.T) {
			got := defaultEndpointForBackend(tt.backend)
			if got != tt.expected {
				t.Errorf("defaultEndpointForBackend(%q) = %q, want %q", tt.backend, got, tt.expected)
			}
		})
	}
}

func TestSetProviderHeaders(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		provider Provider
		wantAuth string
		wantKey  string
	}{
		{
			"ollama_no_key",
			Provider{Backend: "ollama"},
			"", "",
		},
		{
			"ollama_with_key",
			Provider{Backend: "ollama", APIKey: "mykey"},
			"Bearer mykey", "",
		},
		{
			"openai",
			Provider{Backend: "openai", APIKey: "sk-xxx"},
			"Bearer sk-xxx", "",
		},
		{
			"anthropic",
			Provider{Backend: "anthropic", APIKey: "ant-key"},
			"", "ant-key",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("POST", "http://example.com", nil)
			setProviderHeaders(req, tt.provider)
			if tt.wantAuth != "" && req.Header.Get("Authorization") != tt.wantAuth {
				t.Errorf("Authorization = %q, want %q", req.Header.Get("Authorization"), tt.wantAuth)
			}
			if tt.wantKey != "" && req.Header.Get("x-api-key") != tt.wantKey {
				t.Errorf("x-api-key = %q, want %q", req.Header.Get("x-api-key"), tt.wantKey)
			}
		})
	}
}

// ============================================================================
// 2. Pure Function Tests — Message JSON
// ============================================================================

func TestMessageMarshalJSON_PlainText(t *testing.T) {
	t.Parallel()
	msg := Message{Role: "user", Content: "hello"}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if m["role"] != "user" || m["content"] != "hello" {
		t.Errorf("unexpected: %s", data)
	}
}

func TestMessageMarshalJSON_WithImages(t *testing.T) {
	t.Parallel()
	msg := Message{Role: "user", Content: "what is this?", Images: []string{"data:image/png;base64,abc123"}}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	content, ok := m["content"].([]interface{})
	if !ok {
		t.Fatalf("expected content to be array, got %T", m["content"])
	}
	if len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(content))
	}
	// First part: text
	textPart := content[0].(map[string]interface{})
	if textPart["type"] != "text" || textPart["text"] != "what is this?" {
		t.Errorf("unexpected text part: %v", textPart)
	}
	// Second part: image_url
	imgPart := content[1].(map[string]interface{})
	if imgPart["type"] != "image_url" {
		t.Errorf("expected image_url type, got %v", imgPart["type"])
	}
}

func TestMessageMarshalJSON_WithToolCalls(t *testing.T) {
	t.Parallel()
	msg := Message{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{ID: "tc1", Type: "function", Function: ToolCallFunc{Name: "web_search", Arguments: `{"query":"test"}`}},
		},
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	tcs, ok := m["tool_calls"].([]interface{})
	if !ok || len(tcs) != 1 {
		t.Fatalf("expected 1 tool_call, got %v", m["tool_calls"])
	}
}

func TestMessageUnmarshalJSON_StringContent(t *testing.T) {
	t.Parallel()
	raw := `{"role":"assistant","content":"hello world"}`
	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Role != "assistant" || msg.Content != "hello world" {
		t.Errorf("unexpected: role=%q content=%q", msg.Role, msg.Content)
	}
}

func TestMessageUnmarshalJSON_ArrayContent(t *testing.T) {
	t.Parallel()
	raw := `{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"data:img"}}]}`
	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Content != "hello" {
		t.Errorf("expected text extracted from array, got %q", msg.Content)
	}
}

func TestMessageUnmarshalJSON_WithToolCalls(t *testing.T) {
	t.Parallel()
	raw := `{"role":"assistant","content":"","tool_calls":[{"id":"tc1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"test.txt\"}"}}]}`
	var msg Message
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Function.Name != "read_file" {
		t.Errorf("unexpected tool_calls: %+v", msg.ToolCalls)
	}
}

func TestMessageRoundTrip(t *testing.T) {
	t.Parallel()
	original := Message{
		Role:       "tool",
		Content:    "result data",
		ToolCallID: "call_123",
	}
	data, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Message
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Role != original.Role || decoded.Content != original.Content || decoded.ToolCallID != original.ToolCallID {
		t.Errorf("round trip failed: got %+v", decoded)
	}
}

// ============================================================================
// 3. Pure Function Tests — HTML Processing
// ============================================================================

func TestCleanHTMLEntities(t *testing.T) {
	t.Parallel()
	tests := []struct {
		input, expected string
	}{
		{"&amp;", "&"},
		{"&lt;b&gt;", "<b>"},
		{"&quot;hi&quot;", `"hi"`},
		{"&#39;test&#39;", "'test'"},
		{"&nbsp;space", "space"},       // &nbsp; → space, then TrimSpace
		{"  trimmed  ", "trimmed"},
		{"a&nbsp;b", "a b"},            // &nbsp; in middle preserved
		{"&amp;&lt;&gt;&quot;&#39;", `&<>"'`},
	}
	for _, tt := range tests {
		got := cleanHTMLEntities(tt.input)
		if got != tt.expected {
			t.Errorf("cleanHTMLEntities(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestExtractTextFromHTML(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		input    string
		contains string
		excludes string
	}{
		{
			"removes_script",
			"<html><body>Hello<script>var x=1;</script> World</body></html>",
			"Hello",
			"var x",
		},
		{
			"removes_style",
			"<html><body>Text<style>.cls{color:red}</style>More</body></html>",
			"More",
			"color:red",
		},
		{
			"basic_tags",
			"<p>paragraph</p><div>div</div>",
			"paragraph",
			"<p>",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractTextFromHTML(tt.input)
			if !strings.Contains(got, tt.contains) {
				t.Errorf("expected to contain %q, got %q", tt.contains, got)
			}
			if tt.excludes != "" && strings.Contains(got, tt.excludes) {
				t.Errorf("expected to exclude %q, got %q", tt.excludes, got)
			}
		})
	}
}

func TestExtractMetaContent(t *testing.T) {
	t.Parallel()
	html := `<html><head>
		<meta property="og:image" content="https://example.com/og.jpg">
		<meta name="twitter:image" content="https://example.com/tw.jpg">
	</head></html>`
	og := extractMetaContent(html, "og:image")
	if og != "https://example.com/og.jpg" {
		t.Errorf("og:image = %q", og)
	}
	tw := extractMetaContent(html, "twitter:image")
	if tw != "https://example.com/tw.jpg" {
		t.Errorf("twitter:image = %q", tw)
	}
	missing := extractMetaContent(html, "og:title")
	if missing != "" {
		t.Errorf("expected empty, got %q", missing)
	}
}

func TestExtractImgSrcURLs(t *testing.T) {
	t.Parallel()
	html := `<img src="https://example.com/big.jpg"><img src="/icon16.png"><img src="data:image/gif;base64,xxx"><img src="https://example.com/photo.jpg">`
	urls := extractImgSrcURLs(html, "https://example.com")
	// Should include big.jpg and photo.jpg, skip icon16 (too short path) and data URI
	found := map[string]bool{}
	for _, u := range urls {
		found[u] = true
	}
	if !found["https://example.com/big.jpg"] {
		t.Error("expected big.jpg in results")
	}
	if !found["https://example.com/photo.jpg"] {
		t.Error("expected photo.jpg in results")
	}
	for _, u := range urls {
		if strings.HasPrefix(u, "data:") {
			t.Error("data URI should be excluded")
		}
	}
}

func TestIsArticleURL(t *testing.T) {
	t.Parallel()
	tests := []struct {
		url     string
		baseURL string
		expect  bool
	}{
		{"https://example.com/blog/article1", "https://example.com", true},
		{"https://other.com/post", "https://example.com", false},                   // different host
		{"https://example.com/tag/news", "https://example.com", false},              // tag prefix
		{"https://example.com/category/tech", "https://example.com", false},         // category prefix
		{"https://example.com/page/2", "https://example.com", false},                // page prefix
		{"https://example.com/about", "https://example.com", false},                 // about
		{"https://example.com/", "https://example.com", false},                      // root
		{"https://note.com/user/n/nabc123", "https://note.com/user", true},          // note.com
	}
	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			got := isArticleURL(tt.url, tt.baseURL)
			if got != tt.expect {
				t.Errorf("isArticleURL(%q, %q) = %v, want %v", tt.url, tt.baseURL, got, tt.expect)
			}
		})
	}
}

func TestExtractKeywords(t *testing.T) {
	t.Parallel()
	keywords := extractKeywords("Go言語 テスト Go言語 プログラミング")
	if len(keywords) == 0 {
		t.Error("expected keywords, got none")
	}
	// Should not have duplicates
	seen := map[string]bool{}
	for _, kw := range keywords {
		if seen[kw] {
			t.Errorf("duplicate keyword: %q", kw)
		}
		seen[kw] = true
	}
}

// ============================================================================
// 4. Pure Function Tests — fixIncompleteToolCalls
// ============================================================================

func TestFixIncompleteToolCalls_NoToolCalls(t *testing.T) {
	t.Parallel()
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
	}
	result := fixIncompleteToolCalls(msgs)
	if len(result) != 3 {
		t.Errorf("expected 3 messages, got %d", len(result))
	}
}

func TestFixIncompleteToolCalls_CompleteSequence(t *testing.T) {
	t.Parallel()
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "search"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "tc1", Function: ToolCallFunc{Name: "web_search"}}}},
		{Role: "tool", Content: "results", ToolCallID: "tc1"},
		{Role: "assistant", Content: "here are results"},
	}
	result := fixIncompleteToolCalls(msgs)
	if len(result) != 5 {
		t.Errorf("expected 5 messages (no placeholders), got %d", len(result))
	}
}

func TestFixIncompleteToolCalls_MissingResult(t *testing.T) {
	t.Parallel()
	msgs := []Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "search"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "tc1", Function: ToolCallFunc{Name: "web_search"}}}},
		// Missing tool result for tc1
		{Role: "user", Content: "try again"},
	}
	result := fixIncompleteToolCalls(msgs)
	// Should insert placeholder between assistant(tc) and user
	if len(result) != 5 {
		t.Errorf("expected 5 messages (1 placeholder), got %d", len(result))
	}
	// The placeholder should be at index 3
	if result[3].Role != "tool" || result[3].ToolCallID != "tc1" {
		t.Errorf("expected placeholder at index 3, got role=%q tcid=%q", result[3].Role, result[3].ToolCallID)
	}
	if !strings.Contains(result[3].Content, "web_search") {
		t.Error("placeholder should mention tool name")
	}
}

func TestFixIncompleteToolCalls_MultipleBlocks(t *testing.T) {
	t.Parallel()
	msgs := []Message{
		{Role: "system", Content: "sys"},
		// First complete tool call
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "tc1", Function: ToolCallFunc{Name: "read_file"}}}},
		{Role: "tool", Content: "file content", ToolCallID: "tc1"},
		// Second incomplete tool call
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "tc2", Function: ToolCallFunc{Name: "web_fetch"}}}},
		// Missing result for tc2
		{Role: "user", Content: "ok"},
	}
	result := fixIncompleteToolCalls(msgs)
	// tc1: complete (no placeholder), tc2: missing (placeholder added)
	hasPlaceholder := false
	for _, m := range result {
		if m.Role == "tool" && m.ToolCallID == "tc2" {
			hasPlaceholder = true
			if !strings.Contains(m.Content, "web_fetch") {
				t.Error("placeholder should mention web_fetch")
			}
		}
	}
	if !hasPlaceholder {
		t.Error("expected placeholder for tc2")
	}
}

func TestFixIncompleteToolCalls_Empty(t *testing.T) {
	t.Parallel()
	result := fixIncompleteToolCalls(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

// ============================================================================
// 5. Pure Function Tests — Plugin
// ============================================================================

func TestPluginIsEnabled(t *testing.T) {
	t.Parallel()
	// nil Enabled means enabled (default)
	p1 := Plugin{Name: "test"}
	if !p1.IsEnabled() {
		t.Error("nil Enabled should mean enabled")
	}
	// Explicit true
	tr := true
	p2 := Plugin{Name: "test", Enabled: &tr}
	if !p2.IsEnabled() {
		t.Error("Enabled=true should be enabled")
	}
	// Explicit false
	fa := false
	p3 := Plugin{Name: "test", Enabled: &fa}
	if p3.IsEnabled() {
		t.Error("Enabled=false should be disabled")
	}
}

// ============================================================================
// 6. Thread Persistence Tests
// ============================================================================

func TestSaveAndLoadThreadMeta(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	thread := &Thread{
		ID:           "test-1",
		Title:        "Test Thread",
		CreatedAt:    time.Now().Truncate(time.Millisecond),
		UpdatedAt:    time.Now().Truncate(time.Millisecond),
		MessageCount: 5,
	}
	if err := saveThreadMeta(thread); err != nil {
		t.Fatal(err)
	}

	loaded, err := loadThreadMeta("test-1")
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ID != "test-1" || loaded.Title != "Test Thread" || loaded.MessageCount != 5 {
		t.Errorf("unexpected loaded thread: %+v", loaded)
	}
}

func TestLoadThreadMeta_NotFound(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	_, err := loadThreadMeta("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent thread")
	}
}

func TestAppendToLog_And_LoadMessages(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := "log-test"
	msg1 := ThreadMessage{Role: "user", Content: "hello", Timestamp: time.Now().Unix()}
	msg2 := ThreadMessage{Role: "assistant", Content: "hi there", Timestamp: time.Now().Unix()}

	if err := appendToLog(threadID, msg1); err != nil {
		t.Fatal(err)
	}
	if err := appendToLog(threadID, msg2); err != nil {
		t.Fatal(err)
	}

	msgs, err := loadThreadMessages(threadID)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[0].Content != "hello" {
		t.Errorf("msg[0] unexpected: %+v", msgs[0])
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "hi there" {
		t.Errorf("msg[1] unexpected: %+v", msgs[1])
	}
}

func TestLoadThreadMessages_Empty(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	msgs, err := loadThreadMessages("nonexistent")
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected 0 messages, got %d", len(msgs))
	}
}

func TestLoadThreadMessages_MalformedLine(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Write a file with one valid and one invalid line
	f := filepath.Join(threadDir, "bad.jsonl")
	content := `{"role":"user","content":"valid","timestamp":123}
not valid json
{"role":"assistant","content":"also valid","timestamp":124}
`
	os.WriteFile(f, []byte(content), 0644)

	msgs, err := loadThreadMessages("bad")
	if err != nil {
		t.Fatal(err)
	}
	// Should skip malformed line
	if len(msgs) != 2 {
		t.Errorf("expected 2 valid messages, got %d", len(msgs))
	}
}

func TestAppendMessageToThread(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := "append-test"
	msg := Message{Role: "user", Content: "test message"}
	appendMessageToThread(threadID, msg, "")

	// Should create both meta and log
	meta, err := loadThreadMeta(threadID)
	if err != nil {
		t.Fatal(err)
	}
	if meta.MessageCount != 1 {
		t.Errorf("expected count 1, got %d", meta.MessageCount)
	}

	msgs, _ := loadThreadMessages(threadID)
	if len(msgs) != 1 || msgs[0].Content != "test message" {
		t.Errorf("unexpected messages: %+v", msgs)
	}

	// Append another
	msg2 := Message{Role: "assistant", Content: "response"}
	appendMessageToThread(threadID, msg2, "")

	meta2, _ := loadThreadMeta(threadID)
	if meta2.MessageCount != 2 {
		t.Errorf("expected count 2, got %d", meta2.MessageCount)
	}
}

func TestAppendMessageToThread_SetsTitle(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := "title-test"
	msg := Message{Role: "user", Content: "このマシンのディレクトリ構成教えて"}
	appendMessageToThread(threadID, msg, "")

	meta, _ := loadThreadMeta(threadID)
	if meta.Title == "" || meta.Title == "New thread" {
		t.Errorf("expected title from user message, got %q", meta.Title)
	}
}

func TestListThreads_Empty(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threads, _ := listThreads()
	if len(threads) != 0 {
		t.Errorf("expected 0 threads, got %d", len(threads))
	}
}

func TestListThreads_Multiple(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Create two threads
	saveThreadMeta(&Thread{ID: "t1", Title: "Thread 1", CreatedAt: time.Now(), UpdatedAt: time.Now()})
	saveThreadMeta(&Thread{ID: "t2", Title: "Thread 2", CreatedAt: time.Now(), UpdatedAt: time.Now()})

	threads, _ := listThreads()
	if len(threads) != 2 {
		t.Errorf("expected 2 threads, got %d", len(threads))
	}
}

func TestDeleteThread(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := "del-test"
	saveThreadMeta(&Thread{ID: threadID, Title: "Delete Me"})
	appendToLog(threadID, ThreadMessage{Role: "user", Content: "hello", Timestamp: 1})

	if err := deleteThread(threadID); err != nil {
		t.Fatal(err)
	}

	// Both files should be gone
	if _, err := os.Stat(filepath.Join(threadDir, threadID+".json")); !os.IsNotExist(err) {
		t.Error("json file should be deleted")
	}
	if _, err := os.Stat(filepath.Join(threadDir, threadID+".jsonl")); !os.IsNotExist(err) {
		t.Error("jsonl file should be deleted")
	}
}

func TestLoadThread_BackwardCompatMigration(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Write old-format .json with embedded messages
	oldThread := Thread{
		ID:        "old-1",
		Title:     "New thread",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Messages: []ThreadMessage{
			{Role: "user", Content: "hello world", Timestamp: time.Now().Unix()},
			{Role: "assistant", Content: "hi there", Timestamp: time.Now().Unix()},
		},
	}
	data, _ := json.MarshalIndent(oldThread, "", "  ")
	os.WriteFile(filepath.Join(threadDir, "old-1.json"), data, 0644)

	// Load should trigger migration
	thread, err := loadThread("old-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(thread.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(thread.Messages))
	}

	// .jsonl file should now exist
	if _, err := os.Stat(filepath.Join(threadDir, "old-1.jsonl")); os.IsNotExist(err) {
		t.Error("migration should create .jsonl file")
	}

	// Title should be fixed (was "New thread")
	meta, _ := loadThreadMeta("old-1")
	if meta.Title == "New thread" {
		t.Error("migration should fix 'New thread' title")
	}
	if meta.MessageCount != 2 {
		t.Errorf("expected count 2, got %d", meta.MessageCount)
	}
}

// ============================================================================
// 7. Plugin System Tests
// ============================================================================

func TestSaveAndLoadPlugins(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	plugin := Plugin{
		Name:        "test-plugin",
		Description: "A test plugin",
		Version:     "1.0",
		Tested:      true,
		Tool: &PluginTool{
			Description: "Does things",
			Parameters:  map[string]interface{}{"type": "object"},
			Code:        "console.log('hello');",
		},
	}
	if err := savePlugin(plugin); err != nil {
		t.Fatal(err)
	}

	if err := loadPlugins(); err != nil {
		t.Fatal(err)
	}
	if len(loadedPlugins) != 1 {
		t.Fatalf("expected 1 plugin, got %d", len(loadedPlugins))
	}
	if loadedPlugins[0].Name != "test-plugin" || loadedPlugins[0].Tool == nil {
		t.Errorf("unexpected plugin: %+v", loadedPlugins[0])
	}
}

func TestDeletePluginFile(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	savePlugin(Plugin{Name: "doomed", Description: "will die", Version: "1.0"})
	if err := deletePluginFile("doomed"); err != nil {
		t.Fatal(err)
	}
	loadPlugins()
	if len(loadedPlugins) != 0 {
		t.Errorf("expected 0 plugins after delete, got %d", len(loadedPlugins))
	}
}

func TestGetAllTools_WithPlugin(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Load a plugin with a tool
	loadedPlugins = []Plugin{
		{
			Name:    "myplugin",
			Version: "1.0",
			Tested:  true,
			Tool: &PluginTool{
				Description: "My tool",
				Parameters:  map[string]interface{}{"type": "object"},
				Code:        "console.log('test');",
			},
		},
	}

	allTools := getAllTools()
	found := false
	for _, tool := range allTools {
		if tool.Name == "plugin_myplugin" {
			found = true
			break
		}
	}
	if !found {
		t.Error("expected plugin_myplugin in all tools")
	}
}

func TestGetAllTools_SkipsDisabledPlugin(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	disabled := false
	loadedPlugins = []Plugin{
		{
			Name:    "disabled-plugin",
			Version: "1.0",
			Enabled: &disabled,
			Tested:  true,
			Tool: &PluginTool{
				Description: "Disabled tool",
				Parameters:  map[string]interface{}{"type": "object"},
				Code:        "console.log('nope');",
			},
		},
	}

	allTools := getAllTools()
	for _, tool := range allTools {
		if tool.Name == "plugin_disabled-plugin" {
			t.Error("disabled plugin should not appear in tools")
		}
	}
}

// ============================================================================
// 8. HTTP Handler Tests
// ============================================================================

func TestHandleStatus(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	req := httptest.NewRequest("GET", "/api/status", nil)
	w := httptest.NewRecorder()
	ws.handleStatus(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var resp StatusResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Model != "test-model" || resp.Backend != "ollama" || resp.Version != Version {
		t.Errorf("unexpected status: %+v", resp)
	}
	if resp.VisionModel != "test-vision" {
		t.Errorf("expected vision model, got %q", resp.VisionModel)
	}
}

func TestHandleThreads_CRUD(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	// Create
	body := `{"id":"th-1","title":"Test"}`
	req := httptest.NewRequest("POST", "/api/threads", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	ws.handleThreads(w, req)
	if w.Code != 200 {
		t.Errorf("create: expected 200, got %d", w.Code)
	}

	// List
	req = httptest.NewRequest("GET", "/api/threads", nil)
	w = httptest.NewRecorder()
	ws.handleThreads(w, req)
	var threads []ThreadListItem
	json.NewDecoder(w.Body).Decode(&threads)
	if len(threads) != 1 || threads[0].ID != "th-1" {
		t.Errorf("list unexpected: %+v", threads)
	}

	// Get single
	req = httptest.NewRequest("GET", "/api/threads/th-1", nil)
	w = httptest.NewRecorder()
	ws.handleThreads(w, req)
	var thread Thread
	json.NewDecoder(w.Body).Decode(&thread)
	if thread.ID != "th-1" || thread.Title != "Test" {
		t.Errorf("get unexpected: %+v", thread)
	}

	// Rename (POST not PUT)
	req = httptest.NewRequest("POST", "/api/threads/th-1/rename", strings.NewReader(`{"title":"Renamed"}`))
	req.Header.Set("Content-Type", "application/json")
	w = httptest.NewRecorder()
	ws.handleThreads(w, req)
	if w.Code != 200 {
		t.Errorf("rename: expected 200, got %d: %s", w.Code, w.Body.String())
	}
	meta, _ := loadThreadMeta("th-1")
	if meta.Title != "Renamed" {
		t.Errorf("expected 'Renamed', got %q", meta.Title)
	}

	// Delete
	req = httptest.NewRequest("DELETE", "/api/threads/th-1", nil)
	w = httptest.NewRecorder()
	ws.handleThreads(w, req)
	if w.Code != 200 {
		t.Errorf("delete: expected 200, got %d", w.Code)
	}
	threads2, _ := listThreads()
	if len(threads2) != 0 {
		t.Error("thread should be deleted")
	}
}

func TestHandleThreads_GetNotFound(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	ws := NewWebServer(testConfig(server.URL))

	req := httptest.NewRequest("GET", "/api/threads/nonexistent", nil)
	w := httptest.NewRecorder()
	ws.handleThreads(w, req)
	if w.Code != 404 {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestHandlePlugins_List(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	loadedPlugins = []Plugin{
		{Name: "p1", Description: "Plugin 1", Version: "1.0"},
	}

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	ws := NewWebServer(testConfig(server.URL))

	req := httptest.NewRequest("GET", "/api/plugins", nil)
	w := httptest.NewRecorder()
	ws.handlePlugins(w, req)

	var plugins []Plugin
	json.NewDecoder(w.Body).Decode(&plugins)
	if len(plugins) != 1 || plugins[0].Name != "p1" {
		t.Errorf("unexpected plugins: %+v", plugins)
	}
}

func TestHandleUpload_SingleFile(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	ws := NewWebServer(testConfig(server.URL))

	// Create multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, _ := writer.CreateFormFile("files", "test.txt")
	part.Write([]byte("hello world"))
	writer.Close()

	req := httptest.NewRequest("POST", "/api/upload", &buf)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	w := httptest.NewRecorder()
	ws.handleUpload(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Check file exists in workspace
	content, err := os.ReadFile(filepath.Join(dockerWorkspaceDir, "test.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "hello world" {
		t.Errorf("file content = %q", content)
	}
}

func TestHandleSettings_SetProviders(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	body := `{"action":"set_providers","providers":[{"name":"new","backend":"openai","endpoint":"https://api.openai.com/v1","model":"gpt-4","api_key":"sk-test"}]}`
	req := httptest.NewRequest("POST", "/api/settings", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	ws.handleSettings(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if len(cfg.Providers) != 1 || cfg.Providers[0].Name != "new" {
		t.Errorf("providers not updated: %+v", cfg.Providers)
	}
}

func TestHandleDockerStatus(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	ws := NewWebServer(testConfig(server.URL))

	req := httptest.NewRequest("GET", "/api/docker/status", nil)
	w := httptest.NewRecorder()
	ws.handleDockerStatus(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	var resp map[string]interface{}
	json.NewDecoder(w.Body).Decode(&resp)
	if _, ok := resp["docker_available"]; !ok {
		t.Error("expected 'docker_available' field in response")
	}
}

// ============================================================================
// 9. Agent / LLM Tests
// ============================================================================

func TestAgentChat_SimpleResponse(t *testing.T) {
	server := mockLLMServer(t, staticLLMResponse("Hello from mock!"))
	defer server.Close()

	cfg := testConfig(server.URL)
	agent := &Agent{
		config:   cfg,
		messages: []Message{{Role: "system", Content: "test"}},
	}

	agent.messages = append(agent.messages, Message{Role: "user", Content: "hi"})
	resp, err := agent.chat(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Hello from mock!" {
		t.Errorf("expected 'Hello from mock!', got %q", resp.Content)
	}
}

func TestAgentChat_ServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", 500)
	}))
	defer server.Close()

	cfg := testConfig(server.URL)
	agent := &Agent{
		config:   cfg,
		messages: []Message{{Role: "system", Content: "test"}, {Role: "user", Content: "hi"}},
	}

	_, err := agent.chat(context.Background())
	if err == nil {
		t.Error("expected error for 500 response")
	}
}

func TestAgentChatStream_BasicContent(t *testing.T) {
	chunks := []string{"Hello", " ", "World", "!"}
	server := mockLLMServer(t, streamingLLMResponse(chunks))
	defer server.Close()

	cfg := testConfig(server.URL)
	agent := &Agent{
		config:   cfg,
		messages: []Message{{Role: "system", Content: "test"}, {Role: "user", Content: "hi"}},
	}

	var collected []string
	resp, err := agent.chatStream(context.Background(), StreamCallbacks{
		OnContent: func(s string) { collected = append(collected, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Content != "Hello World!" {
		t.Errorf("expected 'Hello World!', got %q", resp.Content)
	}
	if len(collected) == 0 {
		t.Error("OnContent should have been called")
	}
}

func TestAgentChatStream_ThinkTags(t *testing.T) {
	chunks := []string{"<think>reasoning here</think>The answer is 42."}
	server := mockLLMServer(t, streamingLLMResponse(chunks))
	defer server.Close()

	cfg := testConfig(server.URL)
	agent := &Agent{
		config:   cfg,
		messages: []Message{{Role: "system", Content: "test"}, {Role: "user", Content: "meaning of life?"}},
	}

	var thinkParts []string
	var contentParts []string
	resp, err := agent.chatStream(context.Background(), StreamCallbacks{
		OnContent:  func(s string) { contentParts = append(contentParts, s) },
		OnThinking: func(s string) { thinkParts = append(thinkParts, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Thinking != "reasoning here" {
		t.Errorf("expected thinking 'reasoning here', got %q", resp.Thinking)
	}
	if resp.Content != "The answer is 42." {
		t.Errorf("expected content 'The answer is 42.', got %q", resp.Content)
	}
}

func TestHandleStreamingResponse_ThinkTagAcrossChunks(t *testing.T) {
	// Simulate <think> tag content split across SSE chunks
	// Split: "<think>reas" | "oning</think>answer"
	chunks := []string{"<think>reas", "oning</think>answer"}
	server := mockLLMServer(t, streamingLLMResponse(chunks))
	defer server.Close()

	cfg := testConfig(server.URL)
	agent := &Agent{
		config:   cfg,
		messages: []Message{{Role: "system", Content: "test"}, {Role: "user", Content: "test"}},
	}

	var contentParts []string
	var thinkParts []string
	resp, err := agent.chatStream(context.Background(), StreamCallbacks{
		OnContent:  func(s string) { contentParts = append(contentParts, s) },
		OnThinking: func(s string) { thinkParts = append(thinkParts, s) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Thinking != "reasoning" {
		t.Errorf("expected thinking 'reasoning', got %q", resp.Thinking)
	}
	if resp.Content != "answer" {
		t.Errorf("expected content 'answer', got %q", resp.Content)
	}
}

// ============================================================================
// 10. Docker Integration Tests (skip if unavailable)
// ============================================================================

func TestIsDockerAvailable(t *testing.T) {
	// This always runs — just reports status
	available := isDockerAvailable()
	t.Logf("Docker available: %v", available)
}

func TestDockerExec_SimpleEcho(t *testing.T) {
	skipIfNoDocker(t)
	cleanup := setupTestDirs(t)
	defer cleanup()

	if err := ensureDockerContainer(); err != nil {
		t.Skipf("cannot start container: %v", err)
	}
	defer stopDockerContainer()

	output, err := dockerExec("echo hello", 30)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(output, "hello") {
		t.Errorf("expected 'hello' in output, got %q", output)
	}
}

// ============================================================================
// 11. Vision Tests
// ============================================================================

func TestDescribeImages_Empty(t *testing.T) {
	t.Parallel()
	result := describeImages(nil, "moondream", "http://localhost:11434/v1")
	if result != "" {
		t.Errorf("expected empty, got %q", result)
	}
}

func TestDescribeImages_NoVisionModel(t *testing.T) {
	t.Parallel()
	result := describeImages([]string{"data:image/png;base64,abc"}, "", "http://localhost:11434/v1")
	if result != "" {
		t.Errorf("expected empty, got %q", result)
	}
}

func TestDescribeImages_WithMockOllama(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"message": map[string]string{
					"content": "A photo of a Star Wars X-wing model",
				},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer mockServer.Close()

	result := describeImages(
		[]string{"data:image/jpeg;base64,/9j/test"},
		"moondream",
		mockServer.URL+"/v1", // describeImages strips /v1 to get base URL
	)
	if !strings.Contains(result, "X-wing") {
		t.Errorf("expected X-wing description, got %q", result)
	}
}

func TestDescribeImages_StripDataURI(t *testing.T) {
	t.Parallel()
	// Verify the base64 extraction works with and without data URI prefix
	var receivedBody map[string]interface{}
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/chat" {
			json.NewDecoder(r.Body).Decode(&receivedBody)
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"message": map[string]string{"content": "description"},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer mockServer.Close()

	describeImages(
		[]string{"data:image/png;base64,AAAA"},
		"test-vision",
		mockServer.URL+"/v1",
	)

	// Check that images array in request has the stripped base64 (AAAA not the full data URI)
	if receivedBody != nil {
		msgs, ok := receivedBody["messages"].([]interface{})
		if ok && len(msgs) > 0 {
			msg := msgs[0].(map[string]interface{})
			images := msg["images"].([]interface{})
			if len(images) > 0 && images[0].(string) != "AAAA" {
				t.Errorf("expected stripped base64 'AAAA', got %q", images[0])
			}
		}
	}
}

// ============================================================================
// 12. WebServer Integration Tests
// ============================================================================

func TestBuildSystemPrompt_IncludesDate(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	cfg := &Config{SystemPrompt: "You are a test assistant."}
	prompt := buildSystemPrompt(cfg)
	now := time.Now()
	dateStr := fmt.Sprintf("%d年%d月%d日", now.Year(), int(now.Month()), now.Day())
	if !strings.Contains(prompt, dateStr) {
		t.Errorf("prompt should contain date %q, got %q", dateStr, prompt[:100])
	}
}

func TestBuildSystemPrompt_IncludesPlugins(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	loadedPlugins = []Plugin{
		{Name: "weather", Description: "Get weather info", Version: "1.0", Tested: true},
	}

	cfg := &Config{SystemPrompt: "Base prompt."}
	prompt := buildSystemPrompt(cfg)
	if !strings.Contains(prompt, "weather") {
		t.Error("prompt should include plugin info")
	}
	if !strings.Contains(prompt, "TESTED") {
		t.Error("prompt should show TESTED status")
	}
}

func TestBuildSystemPrompt_IncludesProviders(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	cfg := &Config{
		SystemPrompt: "Base prompt.",
		Providers: []Provider{
			{Name: "primary", Backend: "ollama", Model: "model1"},
			{Name: "secondary", Backend: "openai", Model: "gpt-4"},
		},
	}
	prompt := buildSystemPrompt(cfg)
	if !strings.Contains(prompt, "secondary") || !strings.Contains(prompt, "gpt-4") {
		t.Error("prompt should include provider info when multiple providers exist")
	}
}

func TestWebServer_HandleChatSimple(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("Test response"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	body := `{"message":"hello","conversation_id":"conv-1"}`
	req := httptest.NewRequest("POST", "/api/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	ws.handleChat(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var resp ChatAPIResponse
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Response != "Test response" {
		t.Errorf("expected 'Test response', got %q", resp.Response)
	}
	if resp.Error != "" {
		t.Errorf("unexpected error: %s", resp.Error)
	}
}

func TestWebServer_HandleChatStream_SSE(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, streamingLLMResponse([]string{"Hello", " stream"}))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	body := `{"message":"hi","conversation_id":"conv-sse"}`
	req := httptest.NewRequest("POST", "/api/chat/stream", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := newFlushRecorder()
	ws.handleChatStream(w, req)

	// Parse SSE events from response body
	result := w.Body.String()
	if !strings.Contains(result, `"type":"content"`) {
		t.Error("expected content events in SSE stream")
	}
	if !strings.Contains(result, `"type":"done"`) {
		t.Error("expected done event in SSE stream")
	}

	// Verify thread was saved
	msgs, _ := loadThreadMessages("conv-sse")
	if len(msgs) < 2 { // at least user + assistant
		t.Errorf("expected at least 2 messages saved, got %d", len(msgs))
	}
}

func TestWebServer_GetOrCreateAgent_FixesIncomplete(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	// Create a thread with an incomplete tool call sequence
	threadID := "fix-test"
	saveThreadMeta(&Thread{ID: threadID, Title: "Fix Test", MessageCount: 3})
	appendToLog(threadID, ThreadMessage{Role: "user", Content: "search", Timestamp: 1})
	appendToLog(threadID, ThreadMessage{
		Role:      "assistant",
		ToolCalls: []ToolCall{{ID: "tc1", Type: "function", Function: ToolCallFunc{Name: "web_search", Arguments: `{"query":"test"}`}}},
		Timestamp: 2,
	})
	// Missing tool result — then user message
	appendToLog(threadID, ThreadMessage{Role: "user", Content: "try again", Timestamp: 3})

	agent := ws.getOrCreateAgent(threadID)

	// Agent should have a placeholder tool result inserted
	hasPlaceholder := false
	for _, m := range agent.messages {
		if m.Role == "tool" && m.ToolCallID == "tc1" {
			hasPlaceholder = true
		}
	}
	if !hasPlaceholder {
		t.Error("getOrCreateAgent should insert placeholder for incomplete tool calls")
	}
}

// ============================================================================
// 13. Conversation Search Tests
// ============================================================================

func TestSearchConversation_InMemory(t *testing.T) {
	agent := &Agent{
		config: &Config{},
		messages: []Message{
			{Role: "system", Content: "system prompt"},
			{Role: "user", Content: "東京の天気は？"},
			{Role: "assistant", Content: "東京は晴れです。気温は25度です。"},
			{Role: "user", Content: "大阪は？"},
			{Role: "assistant", Content: "大阪は曇りです。"},
		},
	}
	result := agent.searchConversationInMemory("東京")
	if result == "" {
		t.Error("expected matches for '東京'")
	}
	if !strings.Contains(result, "晴れ") {
		t.Error("expected to find '晴れ' in results")
	}
}

// ============================================================================
// 14. SSE Event Parsing Helper (for verifying handleChatStream output)
// ============================================================================

func parseSSEEvents(body string) []StreamEvent {
	var events []StreamEvent
	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				continue
			}
			var event StreamEvent
			if json.Unmarshal([]byte(data), &event) == nil {
				events = append(events, event)
			}
		}
	}
	return events
}

func TestParseSSEEvents(t *testing.T) {
	t.Parallel()
	body := `data: {"type":"content","content":"hello"}

data: {"type":"content","content":" world"}

data: {"type":"done"}

data: [DONE]

`
	events := parseSSEEvents(body)
	if len(events) != 3 {
		t.Errorf("expected 3 events, got %d", len(events))
	}
	if events[0].Content != "hello" {
		t.Errorf("first event content = %q", events[0].Content)
	}
}
