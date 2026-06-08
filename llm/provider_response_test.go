package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAIParseResponse_InvalidToolArguments(t *testing.T) {
	provider := &OpenAIProvider{}

	_, err := provider.parseResponse([]byte(`{
		"choices": [{
			"message": {
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {
						"name": "greet",
						"arguments": "{\"name\":"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 1,
			"completion_tokens": 1,
			"total_tokens": 2
		}
	}`))
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), `openai: parse tool call arguments for "greet"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestQwenParseResponse_InvalidToolArguments(t *testing.T) {
	provider := &QwenProvider{}

	_, err := provider.parseResponse([]byte(`{
		"choices": [{
			"message": {
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {
						"name": "greet",
						"arguments": "{\"name\":"
					}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 1,
			"completion_tokens": 1,
			"total_tokens": 2
		}
	}`))
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
	if !strings.Contains(err.Error(), `qwen: parse tool call arguments for "greet"`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestOpenAIStreamErrorIncludesResponseBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad upstream request", http.StatusBadRequest)
	}))
	defer server.Close()

	provider := &OpenAIProvider{
		apiKey:  "test-key",
		baseURL: server.URL,
		client:  server.Client(),
	}

	_, err := provider.Stream(context.Background(), &Request{
		Model:    "gpt-test",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected stream error, got nil")
	}
	if !strings.Contains(err.Error(), "openai: stream status 400") {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(err.Error(), "bad upstream request") {
		t.Fatalf("expected response body in error, got %v", err)
	}
}

func TestGeminiRequestsUseAPIKeyHeader(t *testing.T) {
	var gotPath string
	var gotQuery string
	var gotKey string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[]}`))
	}))
	defer server.Close()

	provider := &GeminiProvider{
		apiKey:  "test-key",
		baseURL: server.URL,
		client:  server.Client(),
	}

	_, err := provider.Complete(context.Background(), &Request{
		Model:    "gemini-test",
		Messages: []Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete failed: %v", err)
	}
	if gotPath != "/models/gemini-test:generateContent" {
		t.Fatalf("unexpected path: %q", gotPath)
	}
	if strings.Contains(gotQuery, "key=") {
		t.Fatalf("expected API key not to be in query, got %q", gotQuery)
	}
	if gotKey != "test-key" {
		t.Fatalf("expected x-goog-api-key header, got %q", gotKey)
	}
}
