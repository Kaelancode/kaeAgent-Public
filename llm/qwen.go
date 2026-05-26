// llm/qwen.go
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	qwenDefaultBase    = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	qwenMaxRetries     = 3
	qwenInitialBackoff = 500 * time.Millisecond
)

// QwenProvider implements Provider for Alibaba's Qwen API (DashScope OpenAI-compatible mode).
type QwenProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// NewQwenProvider creates a provider reading DASHSCOPE_API_KEY from the environment.
func NewQwenProvider() (*QwenProvider, error) {
	key := os.Getenv("DASHSCOPE_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("qwen: DASHSCOPE_API_KEY environment variable is not set")
	}
	base := os.Getenv("DASHSCOPE_BASE_URL")
	if base == "" {
		base = qwenDefaultBase
	}
	return &QwenProvider{
		apiKey:  key,
		baseURL: base,
		client:  &http.Client{Timeout: 120 * time.Second},
	}, nil
}

func (q *QwenProvider) Name() string { return "qwen" }

func (q *QwenProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	body := q.buildRequestBody(req, false)
	raw, err := q.doWithRetry(ctx, "/chat/completions", body)
	if err != nil {
		return nil, fmt.Errorf("qwen: complete: %w", err)
	}
	return q.parseResponse(raw)
}

func (q *QwenProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	body := q.buildRequestBody(req, true)
	httpReq, err := q.newRequest(ctx, "/chat/completions", body)
	if err != nil {
		return nil, fmt.Errorf("qwen: stream request: %w", err)
	}

	resp, err := q.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("qwen: stream do: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("qwen: stream status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan Event, 64)
	go q.readSSE(ctx, resp.Body, ch)
	return ch, nil
}

func (q *QwenProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return []ModelInfo{
		{ID: "qwen-turbo", Name: "Qwen Turbo", ContextWindow: 131072, Provider: "qwen"},
		{ID: "qwen-plus", Name: "Qwen Plus", ContextWindow: 131072, Provider: "qwen"},
		{ID: "qwen-max", Name: "Qwen Max", ContextWindow: 32768, Provider: "qwen"},
		{ID: "qwen-long", Name: "Qwen Long", ContextWindow: 10000000, Provider: "qwen"},
	}, nil
}

func (q *QwenProvider) buildRequestBody(req *Request, stream bool) map[string]any {
	msgs := make([]map[string]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		msg := map[string]any{"role": m.Role, "content": m.Content}
		if m.Name != "" {
			msg["name"] = m.Name
		}
		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}
		if len(m.ToolCalls) > 0 {
			tcs := make([]map[string]any, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				inputJSON, _ := json.Marshal(tc.Input)
				tcs[j] = map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": string(inputJSON),
					},
				}
			}
			msg["tool_calls"] = tcs
		}
		msgs = append(msgs, msg)
	}

	body := map[string]any{
		"model":    req.Model,
		"messages": msgs,
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
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
				"type": "function",
				"function": map[string]any{
					"name":        t.Name,
					"description": t.Description,
					"parameters":  t.Parameters,
				},
			}
		}
		body["tools"] = tools
	}
	for k, v := range req.Options {
		body[k] = v
	}
	return body
}

func (q *QwenProvider) newRequest(ctx context.Context, path string, body map[string]any) (*http.Request, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("qwen: marshal: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, q.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("qwen: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+q.apiKey)
	return httpReq, nil
}

func (q *QwenProvider) doWithRetry(ctx context.Context, path string, body map[string]any) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < qwenMaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(float64(qwenInitialBackoff) * math.Pow(2, float64(attempt-1)))
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("qwen: retry cancelled: %w", ctx.Err())
			case <-time.After(backoff):
			}
		}
		httpReq, err := q.newRequest(ctx, path, body)
		if err != nil {
			return nil, err
		}
		resp, err := q.client.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("qwen: request: %w", err)
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("qwen: status %d: %s", resp.StatusCode, string(data))
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("qwen: status %d: %s", resp.StatusCode, string(data))
		}
		return data, nil
	}
	return nil, fmt.Errorf("qwen: retries exhausted: %w", lastErr)
}

func (q *QwenProvider) parseResponse(data []byte) (*Response, error) {
	var raw struct {
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("qwen: parse response: %w", err)
	}

	resp := &Response{
		Usage: Usage{
			InputTokens:  raw.Usage.PromptTokens,
			OutputTokens: raw.Usage.CompletionTokens,
			TotalTokens:  raw.Usage.TotalTokens,
		},
		Raw: json.RawMessage(data),
	}

	if len(raw.Choices) > 0 {
		choice := raw.Choices[0]
		resp.FinishReason = choice.FinishReason
		if choice.Message.Content != "" {
			resp.Content = append(resp.Content, ContentBlock{Type: "text", Text: choice.Message.Content})
		}
		for _, tc := range choice.Message.ToolCalls {
			var input map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				return nil, fmt.Errorf("qwen: parse tool call arguments for %q: %w", tc.Function.Name, err)
			}
			resp.Content = append(resp.Content, ContentBlock{
				Type: "tool_call",
				ToolCall: &ToolCall{
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				},
			})
		}
	}
	return resp, nil
}

func (q *QwenProvider) readSSE(ctx context.Context, body io.ReadCloser, ch chan<- Event) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	var sawDone bool
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			if !sendEvent(ctx, ch, Event{Kind: EventDone}) {
				return
			}
			return
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if !sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("qwen: sse parse: %w", err)}) {
				return
			}
			continue
		}

		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta
			if delta.Content != "" {
				if !sendEvent(ctx, ch, Event{Kind: EventText, Text: &TextDelta{Content: delta.Content}}) {
					return
				}
			}
			for _, tc := range delta.ToolCalls {
				if !sendEvent(ctx, ch, Event{Kind: EventToolCall, Tool: &ToolCallDelta{
					Index: tc.Index,
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: tc.Function.Arguments,
				}}) {
					return
				}
			}
		}

		if chunk.Usage != nil {
			if !sendEvent(ctx, ch, Event{Kind: EventUsage, Usage: &UsageDelta{
				InputTokens:  chunk.Usage.PromptTokens,
				OutputTokens: chunk.Usage.CompletionTokens,
				TotalTokens:  chunk.Usage.TotalTokens,
			}}) {
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("qwen: sse scan: %w", err)})
		return
	}
	if !sawDone {
		_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("qwen: stream ended without [DONE]")})
	}
}
