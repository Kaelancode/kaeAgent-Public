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
