package llmhttp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewJSONRequestSetsHeadersAndBody(t *testing.T) {
	req, err := NewJSONRequest(context.Background(), "test", "https://example.test", map[string]any{
		"model": "demo",
	}, func(req *http.Request) {
		req.Header.Set("Authorization", "Bearer token")
	})
	if err != nil {
		t.Fatalf("NewJSONRequest: %v", err)
	}
	if got := req.Header.Get("Content-Type"); got != "application/json" {
		t.Fatalf("expected JSON content type, got %q", got)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer token" {
		t.Fatalf("expected authorization header, got %q", got)
	}
	var body map[string]any
	if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if body["model"] != "demo" {
		t.Fatalf("expected model demo, got %#v", body)
	}
}

func TestDoJSONWithRetryRetriesTransientStatus(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "try again", http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	data, err := DoJSONWithRetry(context.Background(), "test", server.Client(), server.URL, map[string]any{"x": 1}, nil)
	if err != nil {
		t.Fatalf("DoJSONWithRetry: %v", err)
	}
	if string(data) != `{"ok":true}` {
		t.Fatalf("unexpected response: %s", string(data))
	}
	if attempts != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts)
	}
}

func TestOpenSSEStreamReturnsBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("data: ok\n\n"))
	}))
	defer server.Close()

	body, err := OpenSSEStream(context.Background(), "test", server.Client(), server.URL, map[string]any{"stream": true}, nil)
	if err != nil {
		t.Fatalf("OpenSSEStream: %v", err)
	}
	defer body.Close()
}
