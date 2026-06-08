package streaming

import (
	"context"
	"errors"
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

func TestRunner_ReturnsEventErrorAfterDispatch(t *testing.T) {
	ch := make(chan Event, 3)
	streamErr := errors.New("stream failed")
	ch <- Event{Kind: EventError, Err: streamErr}
	ch <- Event{Kind: EventDone}
	close(ch)

	var dispatched []EventKind
	runner := NewRunner()
	runner.On("collector", func(event Event) error {
		dispatched = append(dispatched, event.Kind)
		return nil
	})

	err := runner.Run(context.Background(), ch)
	if !errors.Is(err, streamErr) {
		t.Fatalf("expected stream error, got %v", err)
	}
	if len(dispatched) != 1 || dispatched[0] != EventError {
		t.Fatalf("expected only EventError to be dispatched, got %v", dispatched)
	}
}

func TestRunner_ReturnsErrorForEventErrorWithoutDetails(t *testing.T) {
	ch := make(chan Event, 1)
	ch <- Event{Kind: EventError}
	close(ch)

	runner := NewRunner()
	err := runner.Run(context.Background(), ch)
	if err == nil || err.Error() != "runner: stream error event with nil error" {
		t.Fatalf("expected nil-detail stream error, got %v", err)
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

func TestRunner_PrefersReadyEventOverCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ch := make(chan Event, 1)
	ch <- Event{Kind: EventDone}

	dispatched := false
	runner := NewRunner()
	runner.On("collector", func(e Event) error {
		dispatched = true
		return nil
	})

	if err := runner.Run(ctx, ch); err != nil {
		t.Fatalf("expected ready EventDone to be delivered before cancellation, got %v", err)
	}
	if !dispatched {
		t.Fatal("expected ready event to be dispatched")
	}
}

func TestCollectPrefersReadyEventOverCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	ch := make(chan Event, 1)
	ch <- Event{Kind: EventDone}

	events, err := Collect(ctx, ch)
	if err != nil {
		t.Fatalf("expected ready EventDone to be collected before cancellation, got %v", err)
	}
	if len(events) != 1 || events[0].Kind != EventDone {
		t.Fatalf("expected collected EventDone, got %#v", events)
	}
}
