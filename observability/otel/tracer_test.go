package otel

import (
	"context"
	"fmt"
	"testing"

	"github.com/yourorg/agent-sdk/observability"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func newTestTracer() (observability.Tracer, *tracetest.InMemoryExporter) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exp),
	)
	return NewTracer(tp, "test-agent"), exp
}

func TestOtelTracer_ImplementsInterface(t *testing.T) {
	var _ observability.Tracer = (*otelTracer)(nil)
}

func TestOtelTracer_StartAndEndSpan(t *testing.T) {
	tracer, exp := newTestTracer()

	ctx, span := tracer.StartSpan(context.Background(), "agent.step", map[string]string{
		"session_id": "sess_123",
	})
	tracer.EndSpan(ctx, span, nil)

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if s.Name != "agent.step" {
		t.Errorf("expected span name 'agent.step', got %q", s.Name)
	}

	var foundSession bool
	for _, a := range s.Attributes {
		if string(a.Key) == "session_id" && a.Value.AsString() == "sess_123" {
			foundSession = true
		}
	}
	if !foundSession {
		t.Error("expected session_id attribute on span")
	}
}

func TestOtelTracer_SpanWithError(t *testing.T) {
	tracer, exp := newTestTracer()

	ctx, span := tracer.StartSpan(context.Background(), "failing.span", nil)
	tracer.EndSpan(ctx, span, fmt.Errorf("something broke"))

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if s.Status.Code != 1 {
		t.Errorf("expected Error status code, got %d", s.Status.Code)
	}
	if s.Status.Description != "something broke" {
		t.Errorf("expected error description 'something broke', got %q", s.Status.Description)
	}
}

func TestOtelTracer_AddEvent(t *testing.T) {
	tracer, exp := newTestTracer()

	ctx, span := tracer.StartSpan(context.Background(), "agent.step", nil)
	tracer.AddEvent(ctx, span, "step_start", map[string]string{
		"message_count": "5",
	})
	tracer.EndSpan(ctx, span, nil)

	spans := exp.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 span, got %d", len(spans))
	}
	s := spans[0]
	if len(s.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(s.Events))
	}
	ev := s.Events[0]
	if ev.Name != "step_start" {
		t.Errorf("expected event name 'step_start', got %q", ev.Name)
	}
	var foundCount bool
	for _, a := range ev.Attributes {
		if string(a.Key) == "message_count" && a.Value.AsString() == "5" {
			foundCount = true
		}
	}
	if !foundCount {
		t.Error("expected message_count attribute on event")
	}
}

func TestOtelTracer_NestedSpans(t *testing.T) {
	tracer, exp := newTestTracer()

	ctx, parentSpan := tracer.StartSpan(context.Background(), "parent", nil)
	childCtx, childSpan := tracer.StartSpan(ctx, "child", nil)

	tracer.EndSpan(childCtx, childSpan, nil)
	tracer.EndSpan(ctx, parentSpan, nil)

	spans := exp.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(spans))
	}

	var parent, child *tracetest.SpanStub
	for i := range spans {
		s := &spans[i]
		switch s.Name {
		case "parent":
			parent = s
		case "child":
			child = s
		}
	}
	if parent == nil || child == nil {
		t.Fatal("expected both parent and child spans")
	}

	if child.SpanContext.TraceID() != parent.SpanContext.TraceID() {
		t.Error("expected parent and child to share the same trace ID")
	}
	if child.Parent.SpanID() != parent.SpanContext.SpanID() {
		t.Errorf("expected child's parent span ID %s, got %s",
			parent.SpanContext.SpanID(), child.Parent.SpanID())
	}
}

func TestOtelTracer_EndSpanWithNilSpan(t *testing.T) {
	tracer, _ := newTestTracer()
	tracer.EndSpan(context.Background(), nil, nil)
}

func TestOtelTracer_AddEventWithNilSpan(t *testing.T) {
	tracer, _ := newTestTracer()
	tracer.AddEvent(context.Background(), nil, "event", nil)
}

func TestNewTracer(t *testing.T) {
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	tr := NewTracer(tp, "my-agent")
	if tr == nil {
		t.Fatal("expected non-nil tracer")
	}
}

func TestProviderConfig_GRPCWithHeaders(t *testing.T) {
	cfg := ProviderConfig{
		Endpoint:     "localhost:4317",
		ServiceName:  "test-agent",
		Insecure:     true,
		Headers:      map[string]string{"x-mlflow-experiment-id": "123"},
		ExporterType: "grpc",
	}
	client, err := newOLTPClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestProviderConfig_HTTPWithHeaders(t *testing.T) {
	cfg := ProviderConfig{
		Endpoint:     "localhost:5000",
		ServiceName:  "test-agent",
		Insecure:     true,
		Headers:      map[string]string{"x-mlflow-experiment-id": "123"},
		ExporterType: "http",
	}
	client, err := newOLTPClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestProviderConfig_DefaultIsGRPC(t *testing.T) {
	cfg := ProviderConfig{
		Endpoint:    "localhost:4317",
		ServiceName: "test-agent",
		Insecure:    true,
	}
	client, err := newOLTPClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}
