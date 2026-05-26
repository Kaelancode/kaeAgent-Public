package streaming

import (
	"context"
	"fmt"
	"testing"
)

func TestRunner_OnAndRun(t *testing.T) {
	ch := make(chan Event, 10)
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "hello "}}
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "world"}}
	ch <- Event{Kind: EventDone}
	close(ch)

	var collected []string
	runner := NewRunner()
	runner.On("collector", func(e Event) error {
		if e.Text != nil {
			collected = append(collected, e.Text.Content)
		}
		return nil
	})

	err := runner.Run(context.Background(), ch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(collected) != 2 {
		t.Fatalf("expected 2 text events, got %d", len(collected))
	}
	if collected[0] != "hello " || collected[1] != "world" {
		t.Errorf("unexpected content: %v", collected)
	}
}

func TestRunner_OnKind(t *testing.T) {
	ch := make(chan Event, 10)
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "text"}}
	ch <- Event{Kind: EventUsage, Usage: &UsageDelta{InputTokens: 10}}
	ch <- Event{Kind: EventDone}
	close(ch)

	usageCount := 0
	runner := NewRunner()
	runner.OnKind(EventUsage, "usage_counter", func(e Event) error {
		usageCount++
		return nil
	})

	err := runner.Run(context.Background(), ch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if usageCount != 1 {
		t.Errorf("expected 1 usage event, got %d", usageCount)
	}
}

func TestRunner_HandlerError(t *testing.T) {
	ch := make(chan Event, 5)
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "fail"}}
	close(ch)

	runner := NewRunner()
	runner.On("failing", func(e Event) error {
		return fmt.Errorf("handler failed")
	})

	err := runner.Run(context.Background(), ch)
	if err == nil {
		t.Fatal("expected error from failing handler")
	}
}

func TestCollect(t *testing.T) {
	ch := make(chan Event, 5)
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "a"}}
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "b"}}
	ch <- Event{Kind: EventDone}
	close(ch)

	events, err := Collect(context.Background(), ch)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(events) != 3 {
		t.Errorf("expected 3 events, got %d", len(events))
	}
}

func TestCollectReturnsEventError(t *testing.T) {
	ch := make(chan Event, 5)
	streamErr := fmt.Errorf("boom")
	ch <- Event{Kind: EventText, Text: &TextDelta{Content: "a"}}
	ch <- Event{Kind: EventError, Err: streamErr}
	close(ch)

	events, err := Collect(context.Background(), ch)
	if err == nil {
		t.Fatal("expected stream error")
	}
	if err.Error() != streamErr.Error() {
		t.Fatalf("expected %q, got %q", streamErr.Error(), err.Error())
	}
	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}
	if events[1].Kind != EventError {
		t.Fatalf("expected EventError, got %v", events[1].Kind)
	}
}

func TestRunner_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan Event)
	cancel()

	runner := NewRunner()
	err := runner.Run(ctx, ch)
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}
