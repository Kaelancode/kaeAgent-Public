// llm/claude.go
package llm

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/Kaelancode/kaeAgent-Public/internal/llmhttp"
	"github.com/Kaelancode/kaeAgent-Public/internal/sse"
)

const (
	claudeDefaultBase = "https://api.anthropic.com/v1"
	claudeAPIVersion  = "2023-06-01"
)

// ClaudeProvider implements Provider for the Anthropic Claude API.
type ClaudeProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

var _ Provider = (*ClaudeProvider)(nil)

// NewClaudeProvider creates a provider reading ANTHROPIC_API_KEY from the environment.
func NewClaudeProvider() (*ClaudeProvider, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("claude: ANTHROPIC_API_KEY environment variable is not set")
	}
	base := os.Getenv("ANTHROPIC_BASE_URL")
	if base == "" {
		base = claudeDefaultBase
	}
	return &ClaudeProvider{
		apiKey:  key,
		baseURL: base,
		client:  llmhttp.NewClient(),
	}, nil
}

func (c *ClaudeProvider) Name() string { return "anthropic" }

func (c *ClaudeProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	body := c.buildRequestBody(req, false)
	raw, err := llmhttp.DoJSONWithRetry(ctx, "claude", c.client, c.baseURL+"/messages", body, c.setRequestHeaders)
	if err != nil {
		return nil, fmt.Errorf("claude: complete: %w", err)
	}
	return c.parseResponse(raw)
}

func (c *ClaudeProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	body := c.buildRequestBody(req, true)
	bodyReader, err := llmhttp.OpenSSEStream(ctx, "claude", c.client, c.baseURL+"/messages", body, c.setRequestHeaders)
	if err != nil {
		return nil, err
	}

	ch := make(chan Event, 64)
	go c.readSSE(ctx, bodyReader, ch)
	return ch, nil
}

func (c *ClaudeProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return []ModelInfo{
		{ID: "claude-sonnet-4-20250514", Name: "Claude Sonnet 4", ContextWindow: 200000, Provider: "anthropic"},
		{ID: "claude-3-5-sonnet-20241022", Name: "Claude 3.5 Sonnet", ContextWindow: 200000, Provider: "anthropic"},
		{ID: "claude-3-5-haiku-20241022", Name: "Claude 3.5 Haiku", ContextWindow: 200000, Provider: "anthropic"},
		{ID: "claude-3-opus-20240229", Name: "Claude 3 Opus", ContextWindow: 200000, Provider: "anthropic"},
	}, nil
}

func (c *ClaudeProvider) buildRequestBody(req *Request, stream bool) map[string]any {
	var systemMsg string
	msgs := make([]map[string]any, 0, len(req.Messages))

	for _, m := range req.Messages {
		if m.Role == "system" {
			systemMsg = m.Content
			continue
		}
		if m.Role == "tool" {
			msgs = append(msgs, map[string]any{
				"role": "user",
				"content": []map[string]any{
					{
						"type":        "tool_result",
						"tool_use_id": m.ToolCallID,
						"content":     m.Content,
					},
				},
			})
			continue
		}

		msg := map[string]any{"role": m.Role}
		if len(m.ToolCalls) > 0 {
			content := make([]map[string]any, 0)
			if m.Content != "" {
				content = append(content, map[string]any{"type": "text", "text": m.Content})
			}
			for _, tc := range m.ToolCalls {
				content = append(content, map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": tc.Input,
				})
			}
			msg["content"] = content
		} else {
			msg["content"] = m.Content
		}
		msgs = append(msgs, msg)
	}

	body := map[string]any{
		"model":    req.Model,
		"messages": msgs,
	}
	if systemMsg != "" {
		body["system"] = systemMsg
	}
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	body["max_tokens"] = maxTokens

	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if stream {
		body["stream"] = true
	}
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			tools[i] = map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.Parameters,
			}
		}
		body["tools"] = tools
	}
	applyProviderOptions(body, req.Options, reservedOptions("model", "messages", "stream", "tools", "system"))
	return body
}

func (c *ClaudeProvider) parseResponse(data []byte) (*Response, error) {
	var raw struct {
		Content []struct {
			Type  string         `json:"type"`
			Text  string         `json:"text"`
			ID    string         `json:"id"`
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
		Usage      struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("claude: parse response: %w", err)
	}

	resp := &Response{
		FinishReason: raw.StopReason,
		Usage: Usage{
			InputTokens:  raw.Usage.InputTokens,
			OutputTokens: raw.Usage.OutputTokens,
			TotalTokens:  raw.Usage.InputTokens + raw.Usage.OutputTokens,
		},
		Raw: json.RawMessage(data),
	}

	for _, block := range raw.Content {
		switch block.Type {
		case "text":
			resp.Content = append(resp.Content, ContentBlock{Type: "text", Text: block.Text})
		case "tool_use":
			resp.Content = append(resp.Content, ContentBlock{
				Type: "tool_call",
				ToolCall: &ToolCall{
					ID:    block.ID,
					Name:  block.Name,
					Input: block.Input,
				},
			})
		}
	}
	return resp, nil
}

func (c *ClaudeProvider) setRequestHeaders(req *http.Request) {
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", claudeAPIVersion)
}

func (c *ClaudeProvider) readSSE(ctx context.Context, body io.ReadCloser, ch chan<- Event) {
	defer body.Close()
	defer close(ch)

	reader := bufio.NewReader(body)
	var eventType string
	var sawStop bool

	for {
		line, err := sse.ReadLine(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("claude: sse read: %w", err)})
			return
		}

		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		switch eventType {
		case "content_block_delta":
			var delta struct {
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
				Index int `json:"index"`
			}
			if err := json.Unmarshal([]byte(payload), &delta); err != nil {
				if !sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("claude: sse delta parse: %w", err)}) {
					return
				}
				continue
			}
			switch delta.Delta.Type {
			case "text_delta":
				if !sendEvent(ctx, ch, Event{Kind: EventText, Text: &TextDelta{Content: delta.Delta.Text}}) {
					return
				}
			case "input_json_delta":
				if !sendEvent(ctx, ch, Event{Kind: EventToolCall, Tool: &ToolCallDelta{
					Index: delta.Index,
					Input: delta.Delta.PartialJSON,
				}}) {
					return
				}
			}

		case "content_block_start":
			var start struct {
				Index        int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(payload), &start); err != nil {
				if !sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("claude: sse start parse: %w", err)}) {
					return
				}
				continue
			}
			if start.ContentBlock.Type == "tool_use" {
				if !sendEvent(ctx, ch, Event{Kind: EventToolCall, Tool: &ToolCallDelta{
					Index: start.Index,
					ID:    start.ContentBlock.ID,
					Name:  start.ContentBlock.Name,
				}}) {
					return
				}
			}

		case "message_delta":
			var md struct {
				Usage struct {
					OutputTokens int `json:"output_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal([]byte(payload), &md); err == nil && md.Usage.OutputTokens > 0 {
				if !sendEvent(ctx, ch, Event{Kind: EventUsage, Usage: &UsageDelta{OutputTokens: md.Usage.OutputTokens}}) {
					return
				}
			}

		case "message_start":
			var ms struct {
				Message struct {
					Usage struct {
						InputTokens int `json:"input_tokens"`
					} `json:"usage"`
				} `json:"message"`
			}
			if err := json.Unmarshal([]byte(payload), &ms); err == nil && ms.Message.Usage.InputTokens > 0 {
				if !sendEvent(ctx, ch, Event{Kind: EventUsage, Usage: &UsageDelta{InputTokens: ms.Message.Usage.InputTokens}}) {
					return
				}
			}

		case "message_stop":
			sawStop = true
			if !sendEvent(ctx, ch, Event{Kind: EventDone}) {
				return
			}
			return

		case "error":
			_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("claude: stream error: %s", payload)})
			return
		}
	}

	if !sawStop {
		_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("claude: stream ended without message_stop")})
	}
}
