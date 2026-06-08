// llm/gemini.go
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

	"github.com/yourorg/agent-sdk/internal/llmhttp"
	"github.com/yourorg/agent-sdk/internal/sse"
)

const (
	geminiDefaultBase = "https://generativelanguage.googleapis.com/v1beta"
)

// GeminiProvider implements Provider for Google's Gemini API.
type GeminiProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

var _ Provider = (*GeminiProvider)(nil)

// NewGeminiProvider creates a provider reading GEMINI_API_KEY from the environment.
func NewGeminiProvider() (*GeminiProvider, error) {
	key := os.Getenv("GEMINI_API_KEY")
	if key == "" {
		return nil, fmt.Errorf("gemini: GEMINI_API_KEY environment variable is not set")
	}
	base := os.Getenv("GEMINI_BASE_URL")
	if base == "" {
		base = geminiDefaultBase
	}
	return &GeminiProvider{
		apiKey:  key,
		baseURL: base,
		client:  llmhttp.NewClient(),
	}, nil
}

func (g *GeminiProvider) Name() string { return "gcp.gemini" }

func (g *GeminiProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	body := g.buildRequestBody(req)
	url := fmt.Sprintf("%s/models/%s:generateContent", g.baseURL, req.Model)
	raw, err := llmhttp.DoJSONWithRetry(ctx, "gemini", g.client, url, body, g.setRequestHeaders)
	if err != nil {
		return nil, fmt.Errorf("gemini: complete: %w", err)
	}
	return g.parseResponse(raw)
}

func (g *GeminiProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	body := g.buildRequestBody(req)
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", g.baseURL, req.Model)
	bodyReader, err := llmhttp.OpenSSEStream(ctx, "gemini", g.client, url, body, g.setRequestHeaders)
	if err != nil {
		return nil, err
	}

	ch := make(chan Event, 64)
	go g.readSSE(ctx, bodyReader, ch)
	return ch, nil
}

func (g *GeminiProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	url := fmt.Sprintf("%s/models", g.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini: models request: %w", err)
	}
	httpReq.Header.Set("x-goog-api-key", g.apiKey)
	resp, err := g.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: models do: %w", err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini: models status %d: %s", resp.StatusCode, string(data))
	}

	var result struct {
		Models []struct {
			Name                       string   `json:"name"`
			DisplayName                string   `json:"displayName"`
			InputTokenLimit            int      `json:"inputTokenLimit"`
			OutputTokenLimit           int      `json:"outputTokenLimit"`
			SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
		} `json:"models"`
	}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("gemini: models parse: %w", err)
	}

	models := make([]ModelInfo, 0, len(result.Models))
	for _, m := range result.Models {
		models = append(models, ModelInfo{
			ID:            m.Name,
			Name:          m.DisplayName,
			ContextWindow: m.InputTokenLimit,
			Provider:      "gcp.gemini",
		})
	}
	return models, nil
}

func (g *GeminiProvider) buildRequestBody(req *Request) map[string]any {
	contents := make([]map[string]any, 0, len(req.Messages))
	var systemInstruction map[string]any

	for _, m := range req.Messages {
		if m.Role == "system" {
			systemInstruction = map[string]any{
				"parts": []map[string]any{{"text": m.Content}},
			}
			continue
		}

		role := m.Role
		if role == "assistant" {
			role = "model"
		}
		if role == "tool" {
			contents = append(contents, map[string]any{
				"role": "function",
				"parts": []map[string]any{
					{
						"functionResponse": map[string]any{
							"name":     m.Name,
							"response": map[string]any{"result": m.Content},
						},
					},
				},
			})
			continue
		}

		parts := make([]map[string]any, 0)
		if m.Content != "" {
			parts = append(parts, map[string]any{"text": m.Content})
		}
		for _, tc := range m.ToolCalls {
			parts = append(parts, map[string]any{
				"functionCall": map[string]any{
					"name": tc.Name,
					"args": tc.Input,
				},
			})
		}
		if len(parts) > 0 {
			contents = append(contents, map[string]any{"role": role, "parts": parts})
		}
	}

	body := map[string]any{
		"contents": contents,
	}
	if systemInstruction != nil {
		body["systemInstruction"] = systemInstruction
	}

	genConfig := map[string]any{}
	if req.MaxTokens > 0 {
		genConfig["maxOutputTokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}

	if len(req.Tools) > 0 {
		funcDecls := make([]map[string]any, len(req.Tools))
		for i, t := range req.Tools {
			funcDecls[i] = map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.Parameters,
			}
		}
		body["tools"] = []map[string]any{
			{"functionDeclarations": funcDecls},
		}
	}

	for k, v := range req.Options {
		switch k {
		case "temperature":
			genConfig["temperature"] = v
		case "maxOutputTokens", "max_output_tokens":
			genConfig["maxOutputTokens"] = v
		case "topP", "top_p":
			genConfig["topP"] = v
		case "topK", "top_k":
			genConfig["topK"] = v
		case "candidateCount", "candidate_count":
			genConfig["candidateCount"] = v
		case "stopSequences", "stop_sequences":
			genConfig["stopSequences"] = v
		case "responseMimeType", "response_mime_type":
			genConfig["responseMimeType"] = v
		case "responseSchema", "response_schema":
			genConfig["responseSchema"] = v
		case "presencePenalty", "presence_penalty":
			genConfig["presencePenalty"] = v
		case "frequencyPenalty", "frequency_penalty":
			genConfig["frequencyPenalty"] = v
		case "seed":
			genConfig["seed"] = v
		case "generationConfig":
			if overrides, ok := v.(map[string]any); ok {
				for overrideKey, overrideValue := range overrides {
					genConfig[overrideKey] = overrideValue
				}
				continue
			}
			body[k] = v
		default:
			if _, ok := reservedOptions("contents", "tools", "systemInstruction")[k]; ok {
				continue
			}
			body[k] = v
		}
	}
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	return body
}

func (g *GeminiProvider) parseResponse(data []byte) (*Response, error) {
	var raw struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text         string `json:"text"`
					FunctionCall *struct {
						Name string         `json:"name"`
						Args map[string]any `json:"args"`
					} `json:"functionCall"`
				} `json:"parts"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("gemini: parse response: %w", err)
	}

	resp := &Response{
		Usage: Usage{
			InputTokens:  raw.UsageMetadata.PromptTokenCount,
			OutputTokens: raw.UsageMetadata.CandidatesTokenCount,
			TotalTokens:  raw.UsageMetadata.TotalTokenCount,
		},
		Raw: json.RawMessage(data),
	}

	if len(raw.Candidates) > 0 {
		cand := raw.Candidates[0]
		resp.FinishReason = cand.FinishReason
		for i, part := range cand.Content.Parts {
			if part.Text != "" {
				resp.Content = append(resp.Content, ContentBlock{Type: "text", Text: part.Text})
			}
			if part.FunctionCall != nil {
				resp.Content = append(resp.Content, ContentBlock{
					Type: "tool_call",
					ToolCall: &ToolCall{
						ID:    fmt.Sprintf("call_%d", i),
						Name:  part.FunctionCall.Name,
						Input: part.FunctionCall.Args,
					},
				})
			}
		}
	}
	return resp, nil
}

func (g *GeminiProvider) setRequestHeaders(req *http.Request) {
	req.Header.Set("x-goog-api-key", g.apiKey)
}

func (g *GeminiProvider) readSSE(ctx context.Context, body io.ReadCloser, ch chan<- Event) {
	defer body.Close()
	defer close(ch)

	reader := bufio.NewReader(body)
	for {
		line, err := sse.ReadLine(reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("gemini: sse read: %w", err)})
			return
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		var chunk struct {
			Candidates []struct {
				Content struct {
					Parts []struct {
						Text         string `json:"text"`
						FunctionCall *struct {
							Name string         `json:"name"`
							Args map[string]any `json:"args"`
						} `json:"functionCall"`
					} `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
			UsageMetadata *struct {
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			} `json:"usageMetadata"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if !sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("gemini: sse parse: %w", err)}) {
				return
			}
			return
		}

		if len(chunk.Candidates) > 0 {
			for idx, part := range chunk.Candidates[0].Content.Parts {
				if part.Text != "" {
					if !sendEvent(ctx, ch, Event{Kind: EventText, Text: &TextDelta{Content: part.Text}}) {
						return
					}
				}
				if part.FunctionCall != nil {
					argsJSON, _ := json.Marshal(part.FunctionCall.Args)
					if !sendEvent(ctx, ch, Event{Kind: EventToolCall, Tool: &ToolCallDelta{
						Index: idx,
						Name:  part.FunctionCall.Name,
						Input: string(argsJSON),
					}}) {
						return
					}
				}
			}
		}

		if chunk.UsageMetadata != nil {
			if !sendEvent(ctx, ch, Event{Kind: EventUsage, Usage: &UsageDelta{
				InputTokens:  chunk.UsageMetadata.PromptTokenCount,
				OutputTokens: chunk.UsageMetadata.CandidatesTokenCount,
				TotalTokens:  chunk.UsageMetadata.TotalTokenCount,
			}}) {
				return
			}
		}
	}

	_ = sendEvent(ctx, ch, Event{Kind: EventDone})
}
