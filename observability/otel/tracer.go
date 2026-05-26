package otel

import (
	"context"
	"fmt"

	"github.com/yourorg/agent-sdk/observability"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

type otelTracer struct {
	tracer trace.Tracer
}

var _ observability.Tracer = (*otelTracer)(nil)

func NewTracer(tp trace.TracerProvider, serviceName string) observability.Tracer {
	return &otelTracer{
		tracer: tp.Tracer(serviceName),
	}
}

func (t *otelTracer) StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, observability.Span) {
	opts := []trace.SpanStartOption{
		trace.WithAttributes(mapToAttrs(attrs)...),
	}
	ctx, span := t.tracer.Start(ctx, name, opts...)
	return ctx, span
}

func (t *otelTracer) EndSpan(_ context.Context, s observability.Span, err error) {
	span, ok := s.(trace.Span)
	if !ok {
		return
	}
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
	}
	span.End()
}

func (t *otelTracer) AddEvent(_ context.Context, s observability.Span, name string, attrs map[string]string) {
	span, ok := s.(trace.Span)
	if !ok {
		return
	}
	span.AddEvent(name, trace.WithAttributes(mapToAttrs(attrs)...))
}

func (t *otelTracer) SetSpanAttributes(_ context.Context, s observability.Span, attrs map[string]any) {
	span, ok := s.(trace.Span)
	if !ok {
		return
	}
	span.SetAttributes(typedAttrsToOTel(attrs)...)
}

func typedAttrsToOTel(m map[string]any) []attribute.KeyValue {
	if len(m) == 0 {
		return nil
	}
	attrs := make([]attribute.KeyValue, 0, len(m))
	for k, v := range m {
		switch val := v.(type) {
		case string:
			attrs = append(attrs, attribute.String(k, val))
		case int:
			attrs = append(attrs, attribute.Int64(k, int64(val)))
		case int64:
			attrs = append(attrs, attribute.Int64(k, val))
		case float64:
			attrs = append(attrs, attribute.Float64(k, val))
		case bool:
			attrs = append(attrs, attribute.Bool(k, val))
		default:
			attrs = append(attrs, attribute.String(k, fmt.Sprintf("%v", val)))
		}
	}
	return attrs
}

func mapToAttrs(m map[string]string) []attribute.KeyValue {
	if len(m) == 0 {
		return nil
	}
	attrs := make([]attribute.KeyValue, 0, len(m))
	for k, v := range m {
		attrs = append(attrs, attribute.String(k, v))
	}
	return attrs
}
