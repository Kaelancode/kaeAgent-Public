package tools

import (
	"context"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

func TestDispatcher_Dispatch(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name:        "echo",
		Description: "Echo back the input",
		Schema:      &schema.Schema{Type: "object"},
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			return input["msg"], nil
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	result := d.Dispatch(context.Background(), ToolCall{
		ID:    "call_1",
		Name:  "echo",
		Input: map[string]any{"msg": "hello"},
	})

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	if result.Content != "hello" {
		t.Errorf("expected hello, got %v", result.Content)
	}
	if result.CallID != "call_1" {
		t.Errorf("expected call_1, got %s", result.CallID)
	}
}

func TestDispatcher_DispatchNotFound(t *testing.T) {
	r := NewRegistry()
	d := NewDispatcher(r)

	result := d.Dispatch(context.Background(), ToolCall{Name: "unknown"})
	if result.Err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestDispatcher_DispatchValidationError(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "strict",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"url"},
			Properties: map[string]*schema.Schema{
				"url": {Type: "string"},
			},
		},
		Handler: func(_ context.Context, _ map[string]any) (any, error) { return nil, nil },
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	result := d.Dispatch(context.Background(), ToolCall{
		Name:  "strict",
		Input: map[string]any{},
	})
	if result.Err == nil {
		t.Fatal("expected validation error")
	}
}

func TestDispatcher_DispatchPassesCoercedInputToHandler(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "coerce",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"count", "enabled", "items"},
			Properties: map[string]*schema.Schema{
				"count":   {Type: "integer"},
				"enabled": {Type: "boolean"},
				"items": {
					Type:  "array",
					Items: &schema.Schema{Type: "integer"},
				},
			},
		},
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			count, ok := input["count"].(float64)
			if !ok {
				return nil, fmt.Errorf("count was not coerced: %T", input["count"])
			}
			enabled, ok := input["enabled"].(bool)
			if !ok {
				return nil, fmt.Errorf("enabled was not coerced: %T", input["enabled"])
			}
			items, ok := input["items"].([]any)
			if !ok {
				return nil, fmt.Errorf("items was not preserved as []any: %T", input["items"])
			}
			first, ok := items[0].(float64)
			if !ok {
				return nil, fmt.Errorf("items[0] was not coerced: %T", items[0])
			}
			return map[string]any{
				"count":   count,
				"enabled": enabled,
				"first":   first,
			}, nil
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	original := map[string]any{
		"count":   "5",
		"enabled": "true",
		"items":   []any{"7"},
	}
	result := d.Dispatch(context.Background(), ToolCall{
		Name:  "coerce",
		Input: original,
	})

	if result.Err != nil {
		t.Fatalf("unexpected error: %v", result.Err)
	}
	content := result.Content.(map[string]any)
	if content["count"] != float64(5) || content["enabled"] != true || content["first"] != float64(7) {
		t.Fatalf("handler did not receive coerced values: %#v", content)
	}
	if original["count"] != "5" {
		t.Fatalf("dispatcher mutated original input: %#v", original)
	}
}

func TestDispatcher_DispatchHandlerError(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "failing",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			return nil, fmt.Errorf("boom")
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	result := d.Dispatch(context.Background(), ToolCall{Name: "failing"})
	if result.Err == nil {
		t.Fatal("expected handler error")
	}
}

func TestDispatcher_DispatchChecksNilHandlerBeforeValidation(t *testing.T) {
	r := NewRegistry()
	r.tools["nil_handler"] = ToolDef{
		Name: "nil_handler",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"required"},
			Properties: map[string]*schema.Schema{
				"required": {Type: "string"},
			},
		},
	}

	d := NewDispatcher(r)
	result := d.Dispatch(context.Background(), ToolCall{
		Name:  "nil_handler",
		Input: map[string]any{},
	})
	if result.Err == nil {
		t.Fatal("expected nil handler error")
	}
	if !strings.Contains(result.Err.Error(), "has no handler") {
		t.Fatalf("expected nil handler error before validation, got %v", result.Err)
	}
}

func TestDispatcher_DispatchAll(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "add",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			a, _ := input["a"].(float64)
			b, _ := input["b"].(float64)
			return a + b, nil
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	calls := []ToolCall{
		{ID: "c1", Name: "add", Input: map[string]any{"a": float64(1), "b": float64(2)}},
		{ID: "c2", Name: "add", Input: map[string]any{"a": float64(3), "b": float64(4)}},
	}

	results := d.DispatchAll(context.Background(), calls, 0)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Content != float64(3) {
		t.Errorf("expected 3, got %v", results[0].Content)
	}
	if results[1].Content != float64(7) {
		t.Errorf("expected 7, got %v", results[1].Content)
	}
}

func TestDispatcher_DispatchAllMaxConcurrent(t *testing.T) {
	var (
		current int32
		maxSeen int32
		release = make(chan struct{})
	)

	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "block",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			n := atomic.AddInt32(&current, 1)
			for {
				prev := atomic.LoadInt32(&maxSeen)
				if n <= prev || atomic.CompareAndSwapInt32(&maxSeen, prev, n) {
					break
				}
			}
			<-release
			atomic.AddInt32(&current, -1)
			return input["id"], nil
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	calls := []ToolCall{
		{ID: "c1", Name: "block", Input: map[string]any{"id": "one"}},
		{ID: "c2", Name: "block", Input: map[string]any{"id": "two"}},
		{ID: "c3", Name: "block", Input: map[string]any{"id": "three"}},
	}

	done := make(chan []ToolResult, 1)
	go func() {
		done <- d.DispatchAll(context.Background(), calls, 2)
	}()

	deadline := time.Now().Add(500 * time.Millisecond)
	for atomic.LoadInt32(&maxSeen) < 2 {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for concurrent tool execution")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if atomic.LoadInt32(&maxSeen) > 2 {
		t.Fatalf("expected max concurrency 2, got %d", atomic.LoadInt32(&maxSeen))
	}

	close(release)

	results := <-done
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0].Content != "one" || results[1].Content != "two" || results[2].Content != "three" {
		t.Fatalf("expected original order preserved, got %#v", results)
	}
}

func TestDispatcher_DispatchAllContextCancelledBeforeQueuedStart(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})

	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name: "block",
		Schema: &schema.Schema{
			Type: "object",
		},
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			started <- struct{}{}
			<-release
			return input["id"], nil
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	ctx, cancel := context.WithCancel(context.Background())
	calls := []ToolCall{
		{ID: "c1", Name: "block", Input: map[string]any{"id": "one"}},
		{ID: "c2", Name: "block", Input: map[string]any{"id": "two"}},
		{ID: "c3", Name: "block", Input: map[string]any{"id": "three"}},
	}

	done := make(chan []ToolResult, 1)
	go func() {
		done <- d.DispatchAll(ctx, calls, 1)
	}()

	<-started
	cancel()
	close(release)

	results := <-done
	if results[0].Err != nil {
		t.Fatalf("expected first started call to complete, got error %v", results[0].Err)
	}
	for i := 1; i < len(results); i++ {
		if results[i].Err == nil {
			t.Fatalf("expected queued call %d to be cancelled before start", i)
		}
	}
}

func TestDispatcher_DispatchAllRecoversPanic(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name:   "panic",
		Schema: &schema.Schema{Type: "object"},
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			panic("boom")
		},
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	d := NewDispatcher(r)
	results := d.DispatchAll(context.Background(), []ToolCall{
		{ID: "c1", Name: "panic", Input: map[string]any{}},
	}, 2)
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	if results[0].Err == nil {
		t.Fatal("expected panic error")
	}
	if !strings.Contains(results[0].Err.Error(), "panic: boom") {
		t.Fatalf("unexpected error: %v", results[0].Err)
	}
}

func TestResultToString(t *testing.T) {
	tests := []struct {
		name     string
		result   ToolResult
		expected string
	}{
		{
			name:     "string content",
			result:   ToolResult{Content: "hello"},
			expected: "hello",
		},
		{
			name:     "error result",
			result:   ToolResult{Err: fmt.Errorf("failed")},
			expected: "Error: failed",
		},
		{
			name:     "map content",
			result:   ToolResult{Content: map[string]any{"key": "value"}},
			expected: `{"key":"value"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ResultToString(tt.result)
			if got != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, got)
			}
		})
	}
}
