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

// ============================================================================
// 15. Flux Klein 4B Image Server Tests
// ============================================================================

func TestNeedsImageGeneration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    string
		expected bool
	}{
		// Japanese keywords
		{"画像生成して", true},
		{"画像を生成してください", true},
		{"画像を作って", true},
		{"インフォグラフィックを作って", true},
		{"インフォグラフィクスを作って", true},
		{"写真を生成して", true},
		{"写真を作って", true},
		{"絵を描いて", true},
		{"絵を生成して", true},
		{"コンセプトアートを描いて", true},
		// English keywords
		{"generate an image of a cat", true},
		{"image generation please", true},
		{"generate image of sunset", true},
		{"concept art of a dragon", true},
		// Should NOT match
		{"今日のニュースを教えて", false},
		{"コードを書いて", false},
		{"検索して", false},
		{"hello world", false},
		{"ダイアグラムを描いて", false},
		{"", false},
	}

	for _, tt := range tests {
		got := needsImageGeneration(tt.input)
		if got != tt.expected {
			t.Errorf("needsImageGeneration(%q) = %v, want %v", tt.input, got, tt.expected)
		}
	}
}

func TestInitImageServerDir(t *testing.T) {
	// Save and restore imageServerDir
	origDir := imageServerDir
	defer func() { imageServerDir = origDir }()

	tmp := t.TempDir()
	// Override HOME to isolate
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmp)
	defer os.Setenv("HOME", origHome)

	err := initImageServerDir()
	if err != nil {
		t.Fatalf("initImageServerDir() failed: %v", err)
	}

	// Verify directory was created
	expectedDir := filepath.Join(tmp, ".siki", "image_server")
	if imageServerDir != expectedDir {
		t.Errorf("imageServerDir = %q, want %q", imageServerDir, expectedDir)
	}

	// Verify server.py was written
	scriptPath := filepath.Join(expectedDir, "server.py")
	data, err := os.ReadFile(scriptPath)
	if err != nil {
		t.Fatalf("server.py not found: %v", err)
	}
	if !strings.Contains(string(data), "FastAPI") {
		t.Error("server.py should contain FastAPI import")
	}
	if !strings.Contains(string(data), "Flux Klein 4B") {
		t.Error("server.py should contain Flux Klein 4B header")
	}

	// Verify output directory
	outputDir := filepath.Join(expectedDir, "output")
	if info, err := os.Stat(outputDir); err != nil || !info.IsDir() {
		t.Error("output directory should exist")
	}
}

func TestGenerateImage_MockServer(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Create a mock image server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status":       "ready",
				"model_loaded": true,
			})
		case "/generate":
			var req struct {
				Prompt string `json:"prompt"`
				Width  int    `json:"width"`
				Height int    `json:"height"`
			}
			json.NewDecoder(r.Body).Decode(&req)

			if req.Prompt == "" {
				w.WriteHeader(400)
				json.NewEncoder(w).Encode(map[string]string{"error": "prompt is required"})
				return
			}

			// Return a tiny 1x1 PNG as base64
			pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"image_base64": pngBase64,
				"width":        req.Width,
				"height":       req.Height,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer mockServer.Close()

	// Mark server as ready (skip actual startup)
	imageServerReady = true
	defer func() { imageServerReady = false }()

	config := &Config{
		ImageEndpoint: mockServer.URL,
		ImageModel:    "black-forest-labs/FLUX.2-klein-4B",
		ImageEnabled:  true,
	}

	urlPath, err := generateImage("a beautiful sunset", 512, 512, config)
	if err != nil {
		t.Fatalf("generateImage failed: %v", err)
	}

	if !strings.HasPrefix(urlPath, "/playground/image_") {
		t.Errorf("expected /playground/image_*.png path, got %q", urlPath)
	}
	if !strings.HasSuffix(urlPath, ".png") {
		t.Errorf("expected .png suffix, got %q", urlPath)
	}

	// Verify the file was actually written to playground
	filename := strings.TrimPrefix(urlPath, "/playground/")
	filePath := filepath.Join(playgroundDir, filename)
	if _, err := os.Stat(filePath); err != nil {
		t.Errorf("generated image file not found at %s: %v", filePath, err)
	}
}

func TestGenerateImage_ServerError(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ready", "model_loaded": true})
		case "/generate":
			w.WriteHeader(500)
			json.NewEncoder(w).Encode(map[string]string{"error": "GPU out of memory"})
		}
	}))
	defer mockServer.Close()

	imageServerReady = true
	defer func() { imageServerReady = false }()

	config := &Config{
		ImageEndpoint: mockServer.URL,
		ImageEnabled:  true,
	}

	_, err := generateImage("test prompt", 512, 512, config)
	if err == nil {
		t.Fatal("expected error from server 500, got nil")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error should contain status 500, got: %v", err)
	}
}

func TestGenerateImage_CustomDimensions(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	var receivedWidth, receivedHeight int
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ready", "model_loaded": true})
		case "/generate":
			var req struct {
				Width  int `json:"width"`
				Height int `json:"height"`
			}
			json.NewDecoder(r.Body).Decode(&req)
			receivedWidth = req.Width
			receivedHeight = req.Height

			pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
			json.NewEncoder(w).Encode(map[string]interface{}{
				"image_base64": pngBase64,
				"width":        req.Width,
				"height":       req.Height,
			})
		}
	}))
	defer mockServer.Close()

	imageServerReady = true
	defer func() { imageServerReady = false }()

	config := &Config{ImageEndpoint: mockServer.URL, ImageEnabled: true}
	_, err := generateImage("test", 768, 1024, config)
	if err != nil {
		t.Fatalf("generateImage failed: %v", err)
	}
	if receivedWidth != 768 {
		t.Errorf("width = %d, want 768", receivedWidth)
	}
	if receivedHeight != 1024 {
		t.Errorf("height = %d, want 1024", receivedHeight)
	}
}

func TestEnsureImageServer_AlreadyRunning(t *testing.T) {
	// If server is already marked as ready, ensureImageServer should return immediately
	imageServerReady = true
	defer func() { imageServerReady = false }()

	config := &Config{ImageEndpoint: "http://localhost:9999", ImageEnabled: true}
	err := ensureImageServer(config)
	if err != nil {
		t.Errorf("ensureImageServer should succeed when already ready, got: %v", err)
	}
}

func TestEnsureImageServer_ExternalServer(t *testing.T) {
	// Reset state
	imageServerReady = false
	defer func() { imageServerReady = false }()

	// Create mock external server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ready", "model_loaded": true})
		}
	}))
	defer mockServer.Close()

	config := &Config{ImageEndpoint: mockServer.URL, ImageEnabled: true}
	err := ensureImageServer(config)
	if err != nil {
		t.Fatalf("ensureImageServer failed with external server: %v", err)
	}
	if !imageServerReady {
		t.Error("imageServerReady should be true after detecting external server")
	}
}

func TestEnsureImageServer_DefaultEndpoint(t *testing.T) {
	imageServerReady = false
	defer func() { imageServerReady = false }()

	// Empty endpoint should default to localhost:8100
	config := &Config{ImageEndpoint: "", ImageEnabled: true}

	// This will fail because no server is running on 8100 and we can't start one in tests
	// but it exercises the default endpoint code path
	err := ensureImageServer(config)
	// Expect error since no server and no python/gpu in test env
	if err == nil {
		// Only passes if something is actually running on 8100
		imageServerReady = false
	}
}

func TestGenerateImage_ExecuteTool(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	// Create mock image server
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "ready", "model_loaded": true})
		case "/generate":
			pngBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
			json.NewEncoder(w).Encode(map[string]interface{}{
				"image_base64": pngBase64,
				"width":        512,
				"height":       512,
			})
		}
	}))
	defer mockServer.Close()

	imageServerReady = true
	defer func() { imageServerReady = false }()

	config := &Config{
		ImageEndpoint: mockServer.URL,
		ImageModel:    "black-forest-labs/FLUX.2-klein-4B",
		ImageEnabled:  true,
	}

	agent := &Agent{config: config}
	result, err := agent.executeTool("generate_image", map[string]interface{}{
		"prompt": "a cute cat sitting on a rainbow",
		"width":  float64(512),
		"height": float64(512),
	})
	if err != nil {
		t.Fatalf("executeTool generate_image failed: %v", err)
	}
	if !strings.Contains(result, "Image generated successfully") {
		t.Errorf("expected success message, got: %s", result)
	}
	if !strings.Contains(result, "/playground/image_") {
		t.Errorf("expected image URL in result, got: %s", result)
	}
	if !strings.Contains(result, "![Generated Image]") {
		t.Errorf("expected markdown image, got: %s", result)
	}
}

func TestGenerateImage_ExecuteTool_NoPrompt(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	config := &Config{ImageEnabled: true}
	agent := &Agent{config: config}

	_, err := agent.executeTool("generate_image", map[string]interface{}{})
	if err == nil {
		t.Fatal("expected error for missing prompt")
	}
	if !strings.Contains(err.Error(), "prompt is required") {
		t.Errorf("expected 'prompt is required' error, got: %v", err)
	}
}

func TestImageServerScript_Content(t *testing.T) {
	t.Parallel()

	// Verify the embedded Python script has all required endpoints
	if !strings.Contains(imageServerScript, "@app.get(\"/health\")") {
		t.Error("script missing /health endpoint")
	}
	if !strings.Contains(imageServerScript, "@app.post(\"/generate\")") {
		t.Error("script missing /generate endpoint")
	}
	if !strings.Contains(imageServerScript, "@app.post(\"/load\")") {
		t.Error("script missing /load endpoint")
	}
	if !strings.Contains(imageServerScript, "DiffusionPipeline") {
		t.Error("script should use DiffusionPipeline for auto-detection")
	}
	if !strings.Contains(imageServerScript, "FLUX_MODEL") {
		t.Error("script should read FLUX_MODEL environment variable")
	}
	if !strings.Contains(imageServerScript, "FLUX_FP8") {
		t.Error("script should support FLUX_FP8 quantization flag")
	}
	if !strings.Contains(imageServerScript, "image_base64") {
		t.Error("script should return image_base64 in response")
	}
}

func TestImageConfigDefaults(t *testing.T) {
	t.Parallel()

	cfg := defaultConfig()
	if cfg.ImageModel != "black-forest-labs/FLUX.2-klein-4B" {
		t.Errorf("default ImageModel = %q, want FLUX.2-klein-4B", cfg.ImageModel)
	}
	if cfg.ImageEndpoint != "http://localhost:8100" {
		t.Errorf("default ImageEndpoint = %q, want http://localhost:8100", cfg.ImageEndpoint)
	}
	if !cfg.ImageEnabled {
		t.Error("ImageEnabled should default to true")
	}
}

func TestGenerateImageTool_Definition(t *testing.T) {
	t.Parallel()

	var found bool
	for _, tool := range tools {
		if tool.Name == "generate_image" {
			found = true
			if !strings.Contains(tool.Description, "Flux Klein 4B") {
				t.Error("tool description should mention Flux Klein 4B")
			}
			params := tool.Parameters
			props, ok := params["properties"].(map[string]interface{})
			if !ok {
				t.Fatal("properties should be a map")
			}
			if _, ok := props["prompt"]; !ok {
				t.Error("should have prompt parameter")
			}
			if _, ok := props["width"]; !ok {
				t.Error("should have width parameter")
			}
			if _, ok := props["height"]; !ok {
				t.Error("should have height parameter")
			}
			required, ok := params["required"].([]string)
			if !ok {
				t.Fatal("required should be a string slice")
			}
			if len(required) != 1 || required[0] != "prompt" {
				t.Errorf("required = %v, want [prompt]", required)
			}
			break
		}
	}
	if !found {
		t.Error("generate_image tool not found in builtinTools")
	}
}

// ============================================================================
// Feature: isDissatisfied keyword detection
// ============================================================================

func TestIsDissatisfied(t *testing.T) {
	positives := []string{
		"だめ、もっと良くして",
		"ダメだ",
		"駄目です",
		"やり直して",
		"違う、そうじゃない",
		"もっと良くできないの",
		"いまいちだな",
		"再生成して",
		"改善してください",
		"修正してくれ",
		"直して",
		"redo this",
		"try again please",
		"not good enough",
		"that's wrong",
		"fix it",
		"please improve this",
		"ひどいな",
	}
	for _, msg := range positives {
		if !isDissatisfied(msg) {
			t.Errorf("isDissatisfied(%q) = false, want true", msg)
		}
	}

	negatives := []string{
		"ありがとう",
		"次はどうする？",
		"猫の絵を描いて",
		"天気を教えて",
		"hello",
		"great work",
		"perfect",
	}
	for _, msg := range negatives {
		if isDissatisfied(msg) {
			t.Errorf("isDissatisfied(%q) = true, want false", msg)
		}
	}
}

// ============================================================================
// Feature: modelThinkingEvent
// ============================================================================

func TestModelThinkingEvent(t *testing.T) {
	tests := []struct {
		name        string
		config      *Config
		useSubAgent bool
		wantModel   string
	}{
		{
			name:        "sub-agent used when available",
			config:      &Config{SubModel: "gpt-oss:20b", SubAgent: "qwen3.5:27b"},
			useSubAgent: true,
			wantModel:   "qwen3.5:27b",
		},
		{
			name:        "sub-model used when sub-agent not requested",
			config:      &Config{SubModel: "gpt-oss:20b", SubAgent: "qwen3.5:27b"},
			useSubAgent: false,
			wantModel:   "gpt-oss:20b",
		},
		{
			name:        "sub-model used when no sub-agent configured",
			config:      &Config{SubModel: "gpt-oss:20b"},
			useSubAgent: true,
			wantModel:   "gpt-oss:20b",
		},
		{
			name:        "empty model when nothing configured",
			config:      &Config{},
			useSubAgent: false,
			wantModel:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			event := modelThinkingEvent("test content", tt.config, tt.useSubAgent)
			if event.Type != "thinking" {
				t.Errorf("Type = %q, want %q", event.Type, "thinking")
			}
			if event.Content != "test content" {
				t.Errorf("Content = %q, want %q", event.Content, "test content")
			}
			if event.Model != tt.wantModel {
				t.Errorf("Model = %q, want %q", event.Model, tt.wantModel)
			}
		})
	}
}

// ============================================================================
// Feature: isComicRequest detection
// ============================================================================

func TestIsComicRequest(t *testing.T) {
	positives := []string{
		"openclawをテーマに４コマ漫画をかいて",
		"4コマ漫画を描いて",
		"猫のマンガを作って",
		"AIについてのコミックを作って",
		"make a comic about Go",
	}
	for _, msg := range positives {
		if !isComicRequest(msg) {
			t.Errorf("isComicRequest(%q) = false, want true", msg)
		}
	}

	negatives := []string{
		"猫の絵を描いて",
		"インフォグラフィック",
		"ニュースを調べて",
		"天気を教えて",
	}
	for _, msg := range negatives {
		if isComicRequest(msg) {
			t.Errorf("isComicRequest(%q) = true, want false", msg)
		}
	}
}

// ============================================================================
// Autonomous Idle Thinking Tests
// ============================================================================

func TestBroadcastIdleEvent(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("ok"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	// Create two client channels
	ch1 := make(chan StreamEvent, 16)
	ch2 := make(chan StreamEvent, 16)
	ws.idleClientMu.Lock()
	ws.idleClients[ch1] = true
	ws.idleClients[ch2] = true
	ws.idleClientMu.Unlock()

	// Broadcast an event
	ws.broadcastIdleEvent(StreamEvent{Type: "idle_start", Content: "test task", Model: "test-model"})

	// Both channels should receive the event
	select {
	case ev := <-ch1:
		if ev.Type != "idle_start" || ev.Content != "test task" {
			t.Errorf("ch1 got unexpected event: %+v", ev)
		}
	case <-time.After(time.Second):
		t.Error("ch1 did not receive event")
	}
	select {
	case ev := <-ch2:
		if ev.Type != "idle_start" || ev.Content != "test task" {
			t.Errorf("ch2 got unexpected event: %+v", ev)
		}
	case <-time.After(time.Second):
		t.Error("ch2 did not receive event")
	}

	// Clean up
	ws.idleClientMu.Lock()
	delete(ws.idleClients, ch1)
	delete(ws.idleClients, ch2)
	ws.idleClientMu.Unlock()
}

func TestIdleCancelOnActivity(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := mockLLMServer(t, staticLLMResponse("Test response"))
	defer server.Close()
	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	// Simulate an active idle cancel
	ctx, cancel := context.WithCancel(context.Background())
	ws.mu.Lock()
	ws.idleCancel = cancel
	ws.mu.Unlock()

	// Verify context is not cancelled yet
	select {
	case <-ctx.Done():
		t.Fatal("context should not be cancelled yet")
	default:
		// ok
	}

	// Send a chat request (triggers lastActivity update and idle cancel)
	body := `{"message":"hello","conversation_id":"cancel-test"}`
	req := httptest.NewRequest("POST", "/api/chat/stream", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := newFlushRecorder()
	ws.handleChatStream(w, req)

	// Context should now be cancelled
	select {
	case <-ctx.Done():
		// ok - cancel was called
	default:
		t.Error("idle context should have been cancelled after chat activity")
	}

	// idleCancel should be nil
	ws.mu.RLock()
	if ws.idleCancel != nil {
		t.Error("idleCancel should be nil after cancel")
	}
	ws.mu.RUnlock()
}

func TestAutonomousTaskSelection(t *testing.T) {
	t.Parallel()

	// Verify task pool has entries
	if len(autonomousTasks) == 0 {
		t.Fatal("autonomousTasks should not be empty")
	}

	// Verify no duplicate task names
	seen := make(map[string]bool)
	for _, task := range autonomousTasks {
		if task.Name == "" {
			t.Error("task name should not be empty")
		}
		if seen[task.Name] {
			t.Errorf("duplicate task name: %s", task.Name)
		}
		seen[task.Name] = true

		// Verify prompt function works
		prompt := task.PromptFunc("test summary")
		if prompt == "" {
			t.Errorf("task %q generated empty prompt", task.Name)
		}
		if !strings.Contains(prompt, "test summary") {
			t.Errorf("task %q prompt should contain the summary", task.Name)
		}
	}

	// Verify lastIdleTask prevents consecutive duplicate selection
	cleanup := setupTestDirs(t)
	defer cleanup()

	cfg := &Config{}
	ws := NewWebServer(cfg)
	ws.lastIdleTask = autonomousTasks[0].Name

	// Run multiple selections and check that we don't always get the same task
	differentFound := false
	for i := 0; i < 20; i++ {
		// Simulate the selection logic from runAutonomousThinking
		var selected string
		for attempts := 0; attempts < 10; attempts++ {
			idx := attempts % len(autonomousTasks) // deterministic for test
			candidate := autonomousTasks[idx]
			if candidate.Name != ws.lastIdleTask || len(autonomousTasks) == 1 {
				selected = candidate.Name
				break
			}
		}
		if selected != ws.lastIdleTask {
			differentFound = true
			break
		}
	}
	if len(autonomousTasks) > 1 && !differentFound {
		t.Error("task selection should pick a different task from lastIdleTask")
	}
}

// ============================================================================
// Event Persistence Tests (Thread Replay)
// ============================================================================

func TestEventPersistenceToJSONL(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := "event-test-" + fmt.Sprintf("%d", time.Now().UnixMilli())

	// Create thread metadata
	meta := &Thread{
		ID:        threadID,
		Title:     "Event Test",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	saveThreadMeta(meta)

	// Simulate saving display-only events
	appendToLog(threadID, ThreadMessage{
		EventType: "thinking",
		Role:      "assistant",
		Content:   "サブモデルで処理中...",
		Model:     "test-model",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "tool_start",
		Role:      "assistant",
		ToolName:  "web_search",
		Content:   "web_search",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "tool",
		Content:   "search results here",
		ToolName:  "web_search",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "plan_progress",
		Role:      "assistant",
		Content:   "タスク 1/2 完了 ✅",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "suggestions",
		Role:      "assistant",
		Content:   `["深掘りして","別の観点から"]`,
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "assistant",
		Content:   "Final response",
		Timestamp: time.Now().Unix(),
	})

	// Load thread and verify all messages are present
	thread, err := loadThread(threadID)
	if err != nil {
		t.Fatalf("loadThread failed: %v", err)
	}
	if len(thread.Messages) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(thread.Messages))
	}

	// Verify event types
	if thread.Messages[0].EventType != "thinking" {
		t.Errorf("msg[0] should be thinking event, got %q", thread.Messages[0].EventType)
	}
	if thread.Messages[0].Model != "test-model" {
		t.Errorf("msg[0] model should be test-model, got %q", thread.Messages[0].Model)
	}
	if thread.Messages[1].EventType != "tool_start" {
		t.Errorf("msg[1] should be tool_start event, got %q", thread.Messages[1].EventType)
	}
	if thread.Messages[1].ToolName != "web_search" {
		t.Errorf("msg[1] tool_name should be web_search, got %q", thread.Messages[1].ToolName)
	}
	if thread.Messages[2].EventType != "" {
		t.Errorf("msg[2] should be normal message, got event_type %q", thread.Messages[2].EventType)
	}
	if thread.Messages[3].EventType != "plan_progress" {
		t.Errorf("msg[3] should be plan_progress event, got %q", thread.Messages[3].EventType)
	}
	if thread.Messages[4].EventType != "suggestions" {
		t.Errorf("msg[4] should be suggestions event, got %q", thread.Messages[4].EventType)
	}
	if thread.Messages[5].Content != "Final response" {
		t.Errorf("msg[5] should be final response, got %q", thread.Messages[5].Content)
	}
}

func TestEventSkippedInLLMContext(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{"response": "test"})
	}))
	defer server.Close()

	threadID := "ctx-test-" + fmt.Sprintf("%d", time.Now().UnixMilli())

	// Create thread metadata
	meta := &Thread{
		ID:        threadID,
		Title:     "Context Test",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	saveThreadMeta(meta)

	// Save a mix of regular messages and display-only events
	appendToLog(threadID, ThreadMessage{
		Role:      "user",
		Content:   "Hello",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "thinking",
		Role:      "assistant",
		Content:   "Processing...",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "tool_start",
		Role:      "assistant",
		ToolName:  "web_search",
		Content:   "web_search",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "tool",
		Content:   "tool result",
		ToolName:  "web_search",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "plan_progress",
		Role:      "assistant",
		Content:   "Task 1 done",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		EventType: "suggestions",
		Role:      "assistant",
		Content:   `["suggestion1"]`,
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "assistant",
		Content:   "Response",
		Timestamp: time.Now().Unix(),
	})

	cfg := testConfig(server.URL)
	ws := NewWebServer(cfg)

	// getOrCreateAgent should load only non-event messages
	agent := ws.getOrCreateAgent(threadID)

	// Should have: system + user + tool + assistant = 4 messages (no events)
	expectedCount := 4 // system, user, tool, assistant
	if len(agent.messages) != expectedCount {
		t.Errorf("expected %d messages in LLM context, got %d", expectedCount, len(agent.messages))
		for i, m := range agent.messages {
			t.Logf("  msg[%d]: role=%s content=%q", i, m.Role, truncateString(m.Content, 50))
		}
	}

	// Verify no event messages leaked into LLM context
	for i, m := range agent.messages {
		if m.Content == "Processing..." || m.Content == "web_search" || m.Content == "Task 1 done" || m.Content == `["suggestion1"]` {
			t.Errorf("event message leaked into LLM context at index %d: %q", i, m.Content)
		}
	}
}

func TestProactiveThreadCreation(t *testing.T) {
	cleanup := setupTestDirs(t)
	defer cleanup()

	threadID := proactiveThreadIDPrefix + fmt.Sprintf("%d", time.Now().UnixMilli())

	// Create a proactive thread with unread flag
	meta := &Thread{
		ID:        threadID,
		Title:     "💡 予測されたタスク",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Unread:    true,
		Proactive: true,
	}
	if err := saveThreadMeta(meta); err != nil {
		t.Fatalf("saveThreadMeta failed: %v", err)
	}

	// Save messages
	appendToLog(threadID, ThreadMessage{
		Role:      "user",
		Content:   "テスト質問",
		Timestamp: time.Now().Unix(),
	})
	appendToLog(threadID, ThreadMessage{
		Role:      "assistant",
		Content:   "テスト回答",
		Timestamp: time.Now().Unix(),
	})

	// Verify unread and proactive flags in thread list
	items, err := listThreads()
	if err != nil {
		t.Fatalf("listThreads failed: %v", err)
	}
	found := false
	for _, item := range items {
		if item.ID == threadID {
			found = true
			if !item.Unread {
				t.Error("proactive thread should be unread")
			}
			if !item.Proactive {
				t.Error("proactive thread should have proactive flag")
			}
			break
		}
	}
	if !found {
		t.Fatal("proactive thread not found in listThreads")
	}

	// Verify that loading the thread marks it as read
	loaded, err := loadThread(threadID)
	if err != nil {
		t.Fatalf("loadThread failed: %v", err)
	}
	if len(loaded.Messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(loaded.Messages))
	}
	if loaded.Unread != true {
		t.Error("loadThread should not modify unread status (that's handleThreads' job)")
	}

	// Simulate marking as read (like the handler does)
	loaded.Unread = false
	saveThreadMeta(loaded)

	// Re-check thread list
	items, _ = listThreads()
	for _, item := range items {
		if item.ID == threadID {
			if item.Unread {
				t.Error("thread should be marked as read after viewing")
			}
			break
		}
	}
}

func TestComicPlanCharacterDesign(t *testing.T) {
	// Test that createComicPlan generates the correct task structure
	// with character design pipeline: scenario → char image → describe → 4 panels

	// We can't call createComicPlan directly (needs LLM), but we can verify
	// the plan structure by simulating what it produces
	plan := &Plan{
		Goal:      "Test comic",
		CreatedAt: time.Now().Format(time.RFC3339),
		Status:    "planning",
	}

	// Task 1: Scenario display (summarize)
	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          1,
		Description: "## 4コマ漫画シナリオ\n**テーマ:** Test\n",
		Status:      "pending",
		Tool:        "summarize",
	})

	// Task 2: Character reference image
	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          2,
		Description: "キャラクターデザイン参照画像を生成",
		Status:      "pending",
		Tool:        "generate_image",
		ImagePrompt: "character reference sheet, full body, front view, white background, anime style",
	})

	// Task 3: Describe image (vision model)
	plan.Tasks = append(plan.Tasks, PlanTask{
		ID:          3,
		Description: "参照画像からキャラクター外見を詳細解析",
		Status:      "pending",
		Tool:        "describe_image",
	})

	// Tasks 4-7: Comic panels
	for i := 0; i < 4; i++ {
		plan.Tasks = append(plan.Tasks, PlanTask{
			ID:          i + 4,
			Description: fmt.Sprintf("%dコマ目「%s」を画像生成", i+1, []string{"起", "承", "転", "結"}[i]),
			Status:      "pending",
			Tool:        "generate_image",
			ImagePrompt: fmt.Sprintf("comic panel %d, manga style", i+1),
		})
	}

	// Verify plan structure
	if len(plan.Tasks) != 7 {
		t.Fatalf("expected 7 tasks, got %d", len(plan.Tasks))
	}

	// Verify task order: summarize, generate_image, describe_image, 4x generate_image
	expectedTools := []string{"summarize", "generate_image", "describe_image", "generate_image", "generate_image", "generate_image", "generate_image"}
	for i, task := range plan.Tasks {
		if task.Tool != expectedTools[i] {
			t.Errorf("task %d: expected tool %q, got %q", task.ID, expectedTools[i], task.Tool)
		}
	}

	// Verify task IDs are sequential
	for i, task := range plan.Tasks {
		if task.ID != i+1 {
			t.Errorf("task %d: expected ID %d, got %d", i, i+1, task.ID)
		}
	}

	// Verify character reference image task has ImagePrompt
	if plan.Tasks[1].ImagePrompt == "" {
		t.Error("character reference image task should have ImagePrompt")
	}

	// Verify describe_image task has no ImagePrompt (it reads from previous result)
	if plan.Tasks[2].ImagePrompt != "" {
		t.Error("describe_image task should not have ImagePrompt")
	}

	// Verify panel tasks (4-7) have ImagePrompt
	for i := 3; i < 7; i++ {
		if plan.Tasks[i].ImagePrompt == "" {
			t.Errorf("panel task %d should have ImagePrompt", plan.Tasks[i].ID)
		}
	}

	// Test that isComicRequest detects comic requests
	comicTests := []struct {
		msg    string
		expect bool
	}{
		{"4コマ漫画を描いて", true},
		{"４コマ漫画をかいて", true},
		{"漫画を作って", true},
		{"マンガを描いて", true},
		{"comic about cats", true},
		{"天気を教えて", false},
		{"画像を生成して", false},
	}
	for _, tc := range comicTests {
		if got := isComicRequest(tc.msg); got != tc.expect {
			t.Errorf("isComicRequest(%q) = %v, want %v", tc.msg, got, tc.expect)
		}
	}
}

func TestDescribeImageForCharacter(t *testing.T) {
	// Test with empty inputs
	result := describeImageForCharacter("", "test prompt", "moondream", &Config{})
	if result != "" {
		t.Error("expected empty result for empty base64 image")
	}

	result = describeImageForCharacter("abc123", "test prompt", "", &Config{})
	if result != "" {
		t.Error("expected empty result for empty vision model")
	}
}

func TestVerifyGoalFulfillment(t *testing.T) {
	t.Parallel()

	// Comic request with no images generated → fail
	plan := &Plan{
		Tasks: []PlanTask{
			{ID: 1, Tool: "summarize", Status: "completed"},
			{ID: 2, Tool: "generate_image", Status: "failed"},
			{ID: 3, Tool: "describe_image", Status: "completed"},
			{ID: 4, Tool: "generate_image", Status: "failed"},
			{ID: 5, Tool: "generate_image", Status: "failed"},
			{ID: 6, Tool: "generate_image", Status: "failed"},
			{ID: 7, Tool: "generate_image", Status: "failed"},
		},
	}
	if verifyGoalFulfillment(plan, "4コマ漫画を書いて") {
		t.Error("should fail: comic request with 0 images")
	}

	// Comic request with some images → pass
	plan.Tasks[3].Status = "completed"
	plan.Tasks[4].Status = "completed"
	if !verifyGoalFulfillment(plan, "4コマ漫画を書いて") {
		t.Error("should pass: comic request with some images generated")
	}

	// All images completed → pass
	for i := range plan.Tasks {
		plan.Tasks[i].Status = "completed"
	}
	if !verifyGoalFulfillment(plan, "4コマ漫画を書いて") {
		t.Error("should pass: all tasks completed")
	}

	// More than half failed → fail
	failPlan := &Plan{
		Tasks: []PlanTask{
			{ID: 1, Tool: "web_search", Status: "failed"},
			{ID: 2, Tool: "web_fetch", Status: "failed"},
			{ID: 3, Tool: "summarize", Status: "completed"},
		},
	}
	if verifyGoalFulfillment(failPlan, "調べて") {
		t.Error("should fail: more than half tasks failed")
	}

	// nil plan → fail
	if verifyGoalFulfillment(nil, "test") {
		t.Error("should fail: nil plan")
	}
}

// ============================================================================
// Refusal Detection Tests
// ============================================================================

func TestRefusalDetection(t *testing.T) {
	tests := []struct {
		name     string
		userMsg  string
		response string
		want     bool
	}{
		{"refusal できません", "最新ニュースを教えて", "それについてはできません。", true},
		{"refusal わかりません", "調べてください", "わかりません。", true},
		{"refusal 情報がありません", "AIの最新動向は？", "その情報がありません。", true},
		{"refusal 申し訳ありません", "検索して", "申し訳ありませんが対応できません。", true},
		{"refusal 提供することができません", "予測して", "情報を提供することができません。", true},
		{"short analysis response", "過去の会話を分析して", "OK", true},
		{"short prediction response", "予測してください", "無理です", true},
		{"normal response", "こんにちは", "こんにちは！何かお手伝いできますか？", false},
		{"realtime no URL", "最新ニュース", "今日のニュースは色々あります。AIが進化しています。人工知能の最新動向として、様々な企業が新しいモデルを発表しています。これらの動向は注目に値します。", true},
		{"realtime with URL", "最新ニュース", "https://example.com のニュース", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := needsToolButDidnt(tt.userMsg, tt.response)
			if got != tt.want {
				t.Errorf("needsToolButDidnt(%q, %q) = %v, want %v", tt.userMsg, tt.response, got, tt.want)
			}
		})
	}
}

func TestConversationKeywordRouting(t *testing.T) {
	tests := []struct {
		msg  string
		want bool
	}{
		{"過去の会話を調べろ", true},
		{"履歴を見せて", true},
		{"前に話したことを教えて", true},
		{"さっきの話の続き", true},
		{"予測してくれ", true},
		{"予想して", true},
		{"次に何を言うか当てて", true},
		{"やり取りを調べて", true},
		{"天気を教えて", false},
		{"こんにちは", false},
	}

	for _, tt := range tests {
		t.Run(tt.msg, func(t *testing.T) {
			got := containsConversationKeywords(tt.msg)
			if got != tt.want {
				t.Errorf("containsConversationKeywords(%q) = %v, want %v", tt.msg, got, tt.want)
			}
		})
	}
}

// ============================================================================
// User Profile Tests
// ============================================================================

func TestUserProfileSaveLoad(t *testing.T) {
	// Use temp dir for profile
	tmp := t.TempDir()
	origHome := os.Getenv("HOME")
	os.Setenv("HOME", tmp)
	defer os.Setenv("HOME", origHome)

	os.MkdirAll(filepath.Join(tmp, ".siki"), 0755)

	// Initially no profile
	p := loadUserProfile()
	if p != nil {
		t.Error("expected nil profile initially")
	}

	// Save a profile
	profile := &UserProfile{
		Interests:  []string{"AI", "Go"},
		Occupation: "engineer",
		TechLevel:  "advanced",
	}
	if err := saveUserProfile(profile); err != nil {
		t.Fatalf("saveUserProfile failed: %v", err)
	}

	// Load it back
	loaded := loadUserProfile()
	if loaded == nil {
		t.Fatal("expected non-nil profile after save")
	}
	if len(loaded.Interests) != 2 || loaded.Interests[0] != "AI" {
		t.Errorf("unexpected interests: %v", loaded.Interests)
	}
	if loaded.Occupation != "engineer" {
		t.Errorf("unexpected occupation: %s", loaded.Occupation)
	}

	// Test merge
	incoming := &UserProfile{
		Interests:  []string{"Go", "Rust"}, // Go already exists
		Occupation: "researcher",
		Preferences: []string{"dark mode"},
	}
	merged := mergeProfile(loaded, incoming)
	if len(merged.Interests) != 3 { // AI, Go, Rust
		t.Errorf("expected 3 interests after merge, got %d: %v", len(merged.Interests), merged.Interests)
	}
	if merged.Occupation != "researcher" {
		t.Errorf("expected occupation updated to researcher, got %s", merged.Occupation)
	}
	if len(merged.Preferences) != 1 || merged.Preferences[0] != "dark mode" {
		t.Errorf("unexpected preferences: %v", merged.Preferences)
	}

	// Merge with nil
	result := mergeProfile(nil, incoming)
	if result != incoming {
		t.Error("mergeProfile(nil, x) should return x")
	}
	result = mergeProfile(loaded, nil)
	if result != loaded {
		t.Error("mergeProfile(x, nil) should return x")
	}
}

// ============================================================================
// Digest Email Format Tests
// ============================================================================

func TestDigestEmailFormat(t *testing.T) {
	body := formatDigestEmailBody("siki ダイジェスト — 2026/03/04 09:00", "<h2>AI最新動向</h2><p>テスト内容</p>")

	if !strings.Contains(body, "<!DOCTYPE html>") {
		t.Error("expected HTML doctype")
	}
	if !strings.Contains(body, "siki ダイジェスト") {
		t.Error("expected subject in body")
	}
	if !strings.Contains(body, "AI最新動向") {
		t.Error("expected content in body")
	}
	if !strings.Contains(body, "Generated by siki") {
		t.Error("expected footer")
	}
}
