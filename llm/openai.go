// llm/openai.go
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
	openaiDefaultBase = "https://api.openai.com/v1"
)

// OpenAIProvider implements Provider for the OpenAI API.
type OpenAIProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

var _ Provider = (*OpenAIProvider)(nil)

// NewOpenAIProvider creates a provider reading OPENAI_API_KEY from the environment.
func NewOpenAIProvider() (*OpenAIProvider, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("openai: OPENAI_API_KEY environment variable is not set")
	}
	base := os.Getenv("OPENAI_BASE_URL")
	if base == "" {
		base = openaiDefaultBase
	}
	return &OpenAIProvider{
		apiKey:  key,
		baseURL: base,
		client:  llmhttp.NewClient(),
	}, nil
}

func (o *OpenAIProvider) Name() string { return "openai" }

func (o *OpenAIProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	body := o.buildRequestBody(req, false)
	raw, err := llmhttp.DoJSONWithRetry(ctx, "openai", o.client, o.baseURL+"/chat/completions", body, o.setRequestHeaders)
	if err != nil {
		return nil, fmt.Errorf("openai: complete: %w", err)
	}
	return o.parseResponse(raw)
}

func (o *OpenAIProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	body := o.buildRequestBody(req, true)
	bodyReader, err := llmhttp.OpenSSEStream(ctx, "openai", o.client, o.baseURL+"/chat/completions", body, o.setRequestHeaders)
	if err != nil {
		return nil, err
	}

	ch := make(chan Event, 64)
	go o.readSSE(ctx, bodyReader, ch)
	return ch, nil
}

func (o *OpenAIProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	raw, err := o.doGet(ctx, "/models")
	if err != nil {
		return nil, fmt.Errorf("openai: models: %w", err)
	}
	var resp struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("openai: models parse: %w", err)
	}
	models := make([]ModelInfo, len(resp.Data))
	for i, m := range resp.Data {
		models[i] = ModelInfo{ID: m.ID, Name: m.ID, Provider: "openai"}
	}
	return models, nil
}

func (o *OpenAIProvider) buildRequestBody(req *Request, stream bool) map[string]any {
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
	applyProviderOptions(body, req.Options, reservedOptions("model", "messages", "stream", "tools"))
	return body
}

func (o *OpenAIProvider) doGet(ctx context.Context, path string) ([]byte, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+path, nil)
	if err != nil {
		return nil, fmt.Errorf("openai: get request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)
	resp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai: get: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: get status %d: %s", resp.StatusCode, string(data))
	}
	return data, nil
}

func (o *OpenAIProvider) setRequestHeaders(req *http.Request) {
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
}

func (o *OpenAIProvider) parseResponse(data []byte) (*Response, error) {
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
		return nil, fmt.Errorf("openai: parse response: %w", err)
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
			resp.Content = append(resp.Content, ContentBlock{
				Type: "text",
				Text: choice.Message.Content,
			})
		}
		for _, tc := range choice.Message.ToolCalls {
			var input map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &input); err != nil {
				return nil, fmt.Errorf("openai: parse tool call arguments for %q: %w", tc.Function.Name, err)
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

func (o *OpenAIProvider) readSSE(ctx context.Context, body io.ReadCloser, ch chan<- Event) {
	defer body.Close()
	defer close(ch)

	reader := bufio.NewReader(body)
	var sawDone bool
	for {
		line, err := sse.ReadLine(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("openai: sse read: %w", err)})
			return
		}
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
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if !sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("openai: sse parse: %w", err)}) {
				return
			}
			return
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

	if !sawDone {
		_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("openai: stream ended without [DONE]")})
	}
}
