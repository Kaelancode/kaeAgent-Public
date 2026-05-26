// observability/tracer.go
package observability

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"sort"
	"sync"
	"time"
)

// Span is an opaque handle to a trace span. Concrete implementations (e.g.
// OpenTelemetry) can type-assert this to their own span type.
type Span interface{}

// Tracer defines the interface for distributed tracing. Implementations can
// bridge to OpenTelemetry, Datadog, or any other tracing backend without
// importing third-party libraries here.
type Tracer interface {
	// StartSpan begins a new span as a child of any span in the context.
	// Returns a new context carrying the span and the span handle.
	StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, Span)

	// EndSpan finishes the span. If err is non-nil the span is marked as failed.
	EndSpan(ctx context.Context, span Span, err error)

	// AddEvent records a timestamped event on the span.
	AddEvent(ctx context.Context, span Span, name string, attrs map[string]string)

	// SetSpanAttributes sets attributes on an existing span. Values can be
	// string, int, int64, float64, or bool. This is used for setting
	// semantic-convention attributes (e.g. gen_ai.*) after a span is created.
	SetSpanAttributes(ctx context.Context, span Span, attrs map[string]any)
}

// --- NoopTracer ---

type noopSpan struct{}

// NoopTracer is a tracer that does nothing. Use as a default when no tracing
// backend is configured.
type NoopTracer struct{}

var _ Tracer = (*NoopTracer)(nil)

func (NoopTracer) StartSpan(ctx context.Context, _ string, _ map[string]string) (context.Context, Span) {
	return ctx, noopSpan{}
}

func (NoopTracer) EndSpan(_ context.Context, _ Span, _ error) {}

func (NoopTracer) AddEvent(_ context.Context, _ Span, _ string, _ map[string]string) {}

func (NoopTracer) SetSpanAttributes(_ context.Context, _ Span, _ map[string]any) {}

// --- StdoutTracer ---

type stdoutSpan struct {
	id       string
	name     string
	attrs    map[string]string
	start    time.Time
	events   []spanEvent
	parentID string
}

type spanEvent struct {
	time  time.Time
	name  string
	attrs map[string]string
}

type ctxKey struct{}

// StdoutTracer prints trace spans and events to the provided io.Writer in a
// human-readable format. Use os.Stderr for terminal output or any io.Writer
// for structured logging.
type StdoutTracer struct {
	mu     sync.Mutex
	Writer io.Writer
}

var _ Tracer = (*StdoutTracer)(nil)

// NewStdoutTracer creates a tracer that writes to the given writer.
func NewStdoutTracer(w io.Writer) *StdoutTracer {
	return &StdoutTracer{Writer: w}
}

func (t *StdoutTracer) StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, Span) {
	span := &stdoutSpan{
		id:    generateSpanID(),
		name:  name,
		attrs: attrs,
		start: time.Now(),
	}

	if parent, ok := ctx.Value(ctxKey{}).(*stdoutSpan); ok {
		span.parentID = parent.id
	}

	t.mu.Lock()
	t.write("[TRACE] span_start  %-20s  id=%s", span.name, span.id)
	if span.parentID != "" {
		t.write("  parent=%s", span.parentID)
	}
	t.writeAttrs(attrs)
	t.writeln("")
	t.mu.Unlock()

	return context.WithValue(ctx, ctxKey{}, span), span
}

func (t *StdoutTracer) EndSpan(_ context.Context, s Span, err error) {
	span, ok := s.(*stdoutSpan)
	if !ok {
		return
	}

	elapsed := time.Since(span.start)

	t.mu.Lock()
	defer t.mu.Unlock()

	status := "ok"
	if err != nil {
		status = fmt.Sprintf("error: %v", err)
	}

	t.write("[TRACE] span_end    %-20s  id=%s  duration=%s  status=%s", span.name, span.id, elapsed.Round(time.Millisecond), status)

	if len(span.events) > 0 {
		t.writeln("")
		for _, ev := range span.events {
			offset := ev.time.Sub(span.start).Round(time.Millisecond)
			t.write("[TRACE]   event     %-20s  +%s", ev.name, offset)
			t.writeAttrs(ev.attrs)
			t.writeln("")
		}
	} else {
		t.writeln("")
	}
}

func (t *StdoutTracer) AddEvent(_ context.Context, s Span, name string, attrs map[string]string) {
	span, ok := s.(*stdoutSpan)
	if !ok {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	span.events = append(span.events, spanEvent{
		time:  time.Now(),
		name:  name,
		attrs: attrs,
	})
}

func (t *StdoutTracer) SetSpanAttributes(_ context.Context, s Span, attrs map[string]any) {
	span, ok := s.(*stdoutSpan)
	if !ok {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for k, v := range attrs {
		span.attrs[k] = fmt.Sprintf("%v", v)
	}
}

func (t *StdoutTracer) write(format string, args ...any) {
	fmt.Fprintf(t.Writer, format, args...)
}

func (t *StdoutTracer) writeln(s string) {
	fmt.Fprintln(t.Writer, s)
}

func (t *StdoutTracer) writeAttrs(attrs map[string]string) {
	if len(attrs) == 0 {
		return
	}
	keys := make([]string, 0, len(attrs))
	for k := range attrs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		t.write("  %s=%s", k, attrs[k])
	}
}

func generateSpanID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
