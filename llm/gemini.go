// llm/gemini.go
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
	geminiDefaultBase    = "https://generativelanguage.googleapis.com/v1beta"
	geminiMaxRetries     = 3
	geminiInitialBackoff = 500 * time.Millisecond
)

// GeminiProvider implements Provider for Google's Gemini API.
type GeminiProvider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

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
		client:  &http.Client{Timeout: 120 * time.Second},
	}, nil
}

func (g *GeminiProvider) Name() string { return "gcp.gemini" }

func (g *GeminiProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	body := g.buildRequestBody(req)
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", g.baseURL, req.Model, g.apiKey)
	raw, err := g.doWithRetry(ctx, url, body)
	if err != nil {
		return nil, fmt.Errorf("gemini: complete: %w", err)
	}
	return g.parseResponse(raw)
}

func (g *GeminiProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	body := g.buildRequestBody(req)
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", g.baseURL, req.Model, g.apiKey)

	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("gemini: stream marshal: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("gemini: stream request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := g.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("gemini: stream do: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("gemini: stream status %d: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan Event, 64)
	go g.readSSE(ctx, resp.Body, ch)
	return ch, nil
}

func (g *GeminiProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	url := fmt.Sprintf("%s/models?key=%s", g.baseURL, g.apiKey)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("gemini: models request: %w", err)
	}
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
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
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
			body[k] = v
		}
	}
	if len(genConfig) > 0 {
		body["generationConfig"] = genConfig
	}

	return body
}

func (g *GeminiProvider) doWithRetry(ctx context.Context, url string, body map[string]any) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < geminiMaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(float64(geminiInitialBackoff) * math.Pow(2, float64(attempt-1)))
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("gemini: retry cancelled: %w", ctx.Err())
			case <-time.After(backoff):
			}
		}

		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("gemini: marshal: %w", err)
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("gemini: new request: %w", err)
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := g.client.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("gemini: request: %w", err)
			continue
		}
		respData, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("gemini: status %d: %s", resp.StatusCode, string(respData))
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("gemini: status %d: %s", resp.StatusCode, string(respData))
		}
		return respData, nil
	}
	return nil, fmt.Errorf("gemini: retries exhausted: %w", lastErr)
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

func (g *GeminiProvider) readSSE(ctx context.Context, body io.ReadCloser, ch chan<- Event) {
	defer body.Close()
	defer close(ch)

	scanner := bufio.NewScanner(body)
	for scanner.Scan() {
		line := scanner.Text()
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
			continue
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

	if err := scanner.Err(); err != nil {
		_ = sendEvent(ctx, ch, Event{Kind: EventError, Err: fmt.Errorf("gemini: sse scan: %w", err)})
		return
	}
	_ = sendEvent(ctx, ch, Event{Kind: EventDone})
}
