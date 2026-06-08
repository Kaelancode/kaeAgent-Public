package llmhttp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

const (
	DefaultTimeout = 120 * time.Second
	MaxRetries     = 3
	InitialBackoff = 500 * time.Millisecond
)

type HeaderFunc func(*http.Request)

func NewClient() *http.Client {
	return &http.Client{Timeout: DefaultTimeout}
}

func NewJSONRequest(ctx context.Context, provider, url string, body map[string]any, setHeaders HeaderFunc) (*http.Request, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("%s: marshal: %w", provider, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("%s: new request: %w", provider, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if setHeaders != nil {
		setHeaders(req)
	}
	return req, nil
}

func DoJSONWithRetry(ctx context.Context, provider string, client *http.Client, url string, body map[string]any, setHeaders HeaderFunc) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < MaxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(float64(InitialBackoff) * math.Pow(2, float64(attempt-1)))
			select {
			case <-ctx.Done():
				return nil, fmt.Errorf("%s: retry cancelled: %w", provider, ctx.Err())
			case <-time.After(backoff):
			}
		}

		req, err := NewJSONRequest(ctx, provider, url, body, setHeaders)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("%s: request: %w", provider, err)
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("%s: status %d: %s", provider, resp.StatusCode, string(data))
			continue
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("%s: status %d: %s", provider, resp.StatusCode, string(data))
		}
		return data, nil
	}
	return nil, fmt.Errorf("%s: retries exhausted: %w", provider, lastErr)
}

func OpenSSEStream(ctx context.Context, provider string, client *http.Client, url string, body map[string]any, setHeaders HeaderFunc) (io.ReadCloser, error) {
	req, err := NewJSONRequest(ctx, provider, url, body, setHeaders)
	if err != nil {
		return nil, fmt.Errorf("%s: stream request: %w", provider, err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s: stream do: %w", provider, err)
	}
	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		return nil, fmt.Errorf("%s: stream status %d: %s", provider, resp.StatusCode, string(data))
	}
	return resp.Body, nil
}
