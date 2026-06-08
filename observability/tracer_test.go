package observability

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestNoopTracer(t *testing.T) {
	tracer := NoopTracer{}

	ctx, span := tracer.StartSpan(context.Background(), "test", map[string]string{"key": "value"})
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}
	if span == nil {
		t.Fatal("expected non-nil span")
	}

	tracer.AddEvent(ctx, span, "event", map[string]string{"a": "b"})
	tracer.EndSpan(ctx, span, nil)
	tracer.EndSpan(ctx, span, fmt.Errorf("test error"))
}

func TestNoopTracer_ImplementsInterface(t *testing.T) {
	var _ Tracer = NoopTracer{}
	var _ Tracer = &NoopTracer{}
}

func TestStdoutTracer_SpanLifecycle(t *testing.T) {
	var buf bytes.Buffer
	tracer := NewStdoutTracer(&buf)

	ctx, span := tracer.StartSpan(context.Background(), "agent.step", map[string]string{
		"session_id": "sess_123",
	})
	if ctx == nil {
		t.Fatal("expected non-nil context")
	}

	tracer.AddEvent(ctx, span, "step_start", map[string]string{
		"message_count": "5",
		"tool_count":    "2",
	})

	tracer.AddEvent(ctx, span, "step_complete", map[string]string{
		"input_tokens": "100",
	})

	tracer.EndSpan(ctx, span, nil)

	output := buf.String()

	if !strings.Contains(output, "span_start") {
		t.Error("expected span_start in output")
	}
	if !strings.Contains(output, "agent.step") {
		t.Error("expected span name in output")
	}
	if !strings.Contains(output, "span_end") {
		t.Error("expected span_end in output")
	}
	if !strings.Contains(output, "status=ok") {
		t.Error("expected status=ok in output")
	}
	if !strings.Contains(output, "step_start") {
		t.Error("expected step_start event in output")
	}
	if !strings.Contains(output, "step_complete") {
		t.Error("expected step_complete event in output")
	}
	if !strings.Contains(output, "session_id=sess_123") {
		t.Error("expected session_id attribute in output")
	}
}

func TestStdoutTracer_SpanWithError(t *testing.T) {
	var buf bytes.Buffer
	tracer := NewStdoutTracer(&buf)

	ctx, span := tracer.StartSpan(context.Background(), "failing.span", nil)
	tracer.EndSpan(ctx, span, fmt.Errorf("something broke"))

	output := buf.String()
	if !strings.Contains(output, "error: something broke") {
		t.Errorf("expected error in output, got: %s", output)
	}
}

func TestStdoutTracer_NestedSpans(t *testing.T) {
	var buf bytes.Buffer
	tracer := NewStdoutTracer(&buf)

	ctx, parentSpan := tracer.StartSpan(context.Background(), "parent", nil)
	_, childSpan := tracer.StartSpan(ctx, "child", nil)

	tracer.EndSpan(ctx, childSpan, nil)
	tracer.EndSpan(ctx, parentSpan, nil)

	output := buf.String()
	if !strings.Contains(output, "parent") {
		t.Error("expected parent span in output")
	}
	if !strings.Contains(output, "child") {
		t.Error("expected child span in output")
	}
	if !strings.Contains(output, "parent=") {
		t.Error("expected parent ID reference in child span")
	}
}

func TestStdoutTracer_StartSpanCopiesAttrs(t *testing.T) {
	var buf bytes.Buffer
	tracer := NewStdoutTracer(&buf)
	attrs := map[string]string{"session_id": "original"}

	_, span := tracer.StartSpan(context.Background(), "agent.step", attrs)
	attrs["session_id"] = "mutated"
	attrs["new_key"] = "unexpected"

	got := span.(*stdoutSpan)
	if got.attrs["session_id"] != "original" {
		t.Fatalf("expected copied session_id to remain original, got %q", got.attrs["session_id"])
	}
	if _, exists := got.attrs["new_key"]; exists {
		t.Fatalf("expected copied attrs to exclude later mutation, got %#v", got.attrs)
	}
}

func TestStdoutTracer_ImplementsInterface(t *testing.T) {
	var _ Tracer = &StdoutTracer{}
}
