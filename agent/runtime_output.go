package agent

import (
	"context"

	"github.com/Kaelancode/kaeAgent-Public/streaming"
)

type runOutputAdapter interface {
	EmitToolResult(ctx context.Context, callID, name, content string) error
	EmitFinalText(ctx context.Context, text string) error
	EmitDone(ctx context.Context) error
	EmitError(ctx context.Context, err error) error
}

type noopRunOutputAdapter struct{}

func (noopRunOutputAdapter) EmitToolResult(context.Context, string, string, string) error { return nil }
func (noopRunOutputAdapter) EmitFinalText(context.Context, string) error                  { return nil }
func (noopRunOutputAdapter) EmitDone(context.Context) error                               { return nil }
func (noopRunOutputAdapter) EmitError(context.Context, error) error                       { return nil }

type streamingRunOutputAdapter struct {
	rt  *Runtime
	out chan<- streaming.Event
}

func (a streamingRunOutputAdapter) EmitToolResult(ctx context.Context, callID, name, content string) error {
	return a.rt.sendStreamingEvent(ctx, a.out, streaming.Event{
		Kind: streaming.EventToolResult,
		Result: &streaming.ToolResultDelta{
			CallID:  callID,
			Name:    name,
			Content: content,
		},
	})
}

func (a streamingRunOutputAdapter) EmitFinalText(ctx context.Context, text string) error {
	return a.rt.sendStreamingEvent(ctx, a.out, streaming.Event{
		Kind:  streaming.EventFinalText,
		Final: &streaming.FinalTextDelta{Content: text},
	})
}

func (a streamingRunOutputAdapter) EmitDone(ctx context.Context) error {
	return a.rt.sendStreamingEvent(ctx, a.out, streaming.Event{Kind: streaming.EventDone})
}

func (a streamingRunOutputAdapter) EmitError(ctx context.Context, err error) error {
	return a.rt.sendStreamingEvent(ctx, a.out, streaming.Event{Kind: streaming.EventError, Err: err})
}
