// tools/http.go
package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

// NewHTTPTool creates a built-in tool that performs HTTP requests.
func NewHTTPTool() ToolDef {
	minURLLen := 1
	return ToolDef{
		Name:        "http_request",
		Description: "Make an HTTP request to the specified URL. Supports GET, POST, PUT, PATCH, DELETE methods. Returns the response status, headers, and body.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"url": {
					Type:        "string",
					Description: "The URL to send the request to",
					MinLength:   &minURLLen,
				},
				"method": {
					Type:        "string",
					Description: "HTTP method",
					Enum:        []any{"GET", "POST", "PUT", "PATCH", "DELETE"},
					Default:     "GET",
				},
				"headers": {
					Type:        "object",
					Description: "HTTP headers as key-value pairs",
				},
				"body": {
					Type:        "string",
					Description: "Request body (for POST, PUT, PATCH)",
				},
				"timeout_seconds": {
					Type:        "integer",
					Description: "Request timeout in seconds (default: 30)",
					Default:     float64(30),
				},
			},
			Required: []string{"url"},
		},
		Tags:    []string{"web", "http"},
		Handler: httpHandler,
	}
}

func httpHandler(ctx context.Context, input map[string]any) (any, error) {
	rawURL, _ := input["url"].(string)
	if rawURL == "" {
		return nil, fmt.Errorf("http: url is required")
	}
	parsedURL, err := validateHTTPRequestURL(ctx, rawURL)
	if err != nil {
		return nil, err
	}

	method := "GET"
	if m, ok := input["method"].(string); ok && m != "" {
		method = strings.ToUpper(m)
	}

	timeout := 30 * time.Second
	if t, ok := input["timeout_seconds"].(float64); ok && t > 0 {
		timeout = time.Duration(t) * time.Second
	}

	var bodyReader io.Reader
	if b, ok := input["body"].(string); ok && b != "" {
		bodyReader = bytes.NewBufferString(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, parsedURL.String(), bodyReader)
	if err != nil {
		return nil, fmt.Errorf("http: create request: %w", err)
	}

	if headers, ok := input["headers"].(map[string]any); ok {
		for k, v := range headers {
			if vs, ok := v.(string); ok {
				req.Header.Set(k, vs)
			}
		}
	}

	client := &http.Client{
		Timeout: timeout,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("http: stopped after %d redirects", len(via))
			}
			if _, err := validateHTTPRequestURL(req.Context(), req.URL.String()); err != nil {
				return err
			}
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http: request failed: %w", err)
	}
	defer resp.Body.Close()

	const maxBody = 1 << 20 // 1MB
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return nil, fmt.Errorf("http: read response: %w", err)
	}

	respHeaders := make(map[string]string)
	for k := range resp.Header {
		respHeaders[k] = resp.Header.Get(k)
	}

	result := map[string]any{
		"status":      resp.StatusCode,
		"status_text": resp.Status,
		"headers":     respHeaders,
	}

	var jsonBody any
	if err := json.Unmarshal(body, &jsonBody); err == nil {
		result["body"] = jsonBody
	} else {
		result["body"] = string(body)
	}

	return result, nil
}

func validateHTTPRequestURL(ctx context.Context, rawURL string) (*url.URL, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("http: invalid url: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("http: unsupported url scheme %q", parsed.Scheme)
	}
	if parsed.Hostname() == "" {
		return nil, fmt.Errorf("http: url host is required")
	}
	if err := validateHTTPRequestHost(ctx, parsed.Hostname()); err != nil {
		return nil, err
	}
	return parsed, nil
}

func validateHTTPRequestHost(ctx context.Context, host string) error {
	if isBlockedHTTPHostname(host) {
		return fmt.Errorf("http: host %q is not allowed", host)
	}

	addrs, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return fmt.Errorf("http: resolve host %q: %w", host, err)
	}
	if len(addrs) == 0 {
		return fmt.Errorf("http: resolve host %q: no addresses found", host)
	}
	for _, addr := range addrs {
		ip, ok := netip.AddrFromSlice(addr.IP)
		if !ok {
			return fmt.Errorf("http: invalid resolved address for host %q", host)
		}
		if isBlockedHTTPIP(ip.Unmap()) {
			return fmt.Errorf("http: host %q resolves to a disallowed address", host)
		}
	}
	return nil
}

func isBlockedHTTPHostname(host string) bool {
	host = strings.TrimSpace(strings.ToLower(strings.TrimSuffix(host, ".")))
	switch host {
	case "", "localhost", "metadata.google.internal":
		return true
	}
	return strings.HasSuffix(host, ".internal")
}

func isBlockedHTTPIP(ip netip.Addr) bool {
	return ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified()
}
