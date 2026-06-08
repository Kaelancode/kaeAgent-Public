// streaming/runner.go
package streaming

import (
	"context"
	"fmt"
	"sync"
)

// Handler processes a single streaming event. Return an error to signal the
// runner to stop dispatching further events.
type Handler func(event Event) error

// Runner fans out streaming events from a source channel to registered handlers.
type Runner struct {
	mu       sync.RWMutex
	handlers []namedHandler
}

type namedHandler struct {
	name    string
	handler Handler
}

// NewRunner creates a runner with no handlers.
func NewRunner() *Runner {
	return &Runner{}
}

// On registers a handler with a name. Handlers are called in registration order.
func (r *Runner) On(name string, h Handler) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.handlers = append(r.handlers, namedHandler{name: name, handler: h})
}

// OnKind registers a handler that only fires for the given event kind.
func (r *Runner) OnKind(kind EventKind, name string, h Handler) {
	r.On(name, func(event Event) error {
		if event.Kind == kind {
			return h(event)
		}
		return nil
	})
}

// Run reads events from the source channel and fans them out to all registered
// handlers. It blocks until the channel is closed or the context is cancelled.
// Returns the first handler error encountered.
func (r *Runner) Run(ctx context.Context, source <-chan Event) error {
	for {
		event, ok, err := nextEvent(ctx, source)
		if err != nil {
			return fmt.Errorf("runner: context cancelled: %w", ctx.Err())
		}
		if !ok {
			return nil
		}
		if err := r.dispatch(event); err != nil {
			return fmt.Errorf("runner: handler error: %w", err)
		}
		if event.Kind == EventError {
			if event.Err != nil {
				return event.Err
			}
			return fmt.Errorf("runner: stream error event with nil error")
		}
		if event.Kind == EventDone {
			return nil
		}
	}
}

func (r *Runner) dispatch(event Event) error {
	r.mu.RLock()
	handlers := make([]namedHandler, len(r.handlers))
	copy(handlers, r.handlers)
	r.mu.RUnlock()

	for _, nh := range handlers {
		if err := nh.handler(event); err != nil {
			return fmt.Errorf("handler %q: %w", nh.name, err)
		}
	}
	return nil
}

// Collect is a convenience function that reads all events from a channel into a slice.
func Collect(ctx context.Context, source <-chan Event) ([]Event, error) {
	var events []Event
	for {
		event, ok, err := nextEvent(ctx, source)
		if err != nil {
			return events, fmt.Errorf("runner: collect cancelled: %w", ctx.Err())
		}
		if !ok {
			return events, nil
		}
		events = append(events, event)
		if event.Kind == EventError {
			if event.Err != nil {
				return events, event.Err
			}
			return events, fmt.Errorf("runner: stream error event with nil error")
		}
		if event.Kind == EventDone {
			return events, nil
		}
	}
}

func nextEvent(ctx context.Context, source <-chan Event) (Event, bool, error) {
	select {
	case event, ok := <-source:
		return event, ok, nil
	default:
	}

	select {
	case <-ctx.Done():
		return Event{}, false, ctx.Err()
	case event, ok := <-source:
		return event, ok, nil
	}
}
