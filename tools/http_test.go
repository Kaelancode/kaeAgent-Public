package tools

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPTool_BlocksLoopbackHost(t *testing.T) {
	_, err := httpHandler(context.Background(), map[string]any{
		"url": "http://127.0.0.1/",
	})
	if err == nil {
		t.Fatal("expected loopback host to be blocked")
	}
	if !strings.Contains(err.Error(), "not allowed") && !strings.Contains(err.Error(), "disallowed") {
		t.Fatalf("expected blocked-host error, got %v", err)
	}
}

func TestHTTPTool_BlocksUnsupportedScheme(t *testing.T) {
	_, err := httpHandler(context.Background(), map[string]any{
		"url": "ftp://example.com/file.txt",
	})
	if err == nil {
		t.Fatal("expected unsupported scheme to be blocked")
	}
	if !strings.Contains(err.Error(), "unsupported url scheme") {
		t.Fatalf("expected unsupported scheme error, got %v", err)
	}
}

func TestHTTPTool_BlocksRedirectToLoopback(t *testing.T) {
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://127.0.0.1/secret", http.StatusFound)
	}))
	defer redirector.Close()

	_, err := httpHandler(context.Background(), map[string]any{
		"url": redirector.URL,
	})
	if err == nil {
		t.Fatal("expected redirect to loopback host to be blocked")
	}
	if !strings.Contains(err.Error(), "not allowed") && !strings.Contains(err.Error(), "disallowed") {
		t.Fatalf("expected blocked redirect error, got %v", err)
	}
}
