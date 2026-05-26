// tools/dispatcher.go
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// Dispatcher resolves tool calls by name and executes them against a registry.
type Dispatcher struct {
	registry *Registry
}

// NewDispatcher creates a dispatcher backed by the given registry.
func NewDispatcher(r *Registry) *Dispatcher {
	return &Dispatcher{registry: r}
}

// Dispatch executes a single tool call. It validates input against the schema,
// runs the handler, and returns the result.
func (d *Dispatcher) Dispatch(ctx context.Context, call ToolCall) ToolResult {
	tool, ok := d.registry.Get(call.Name)
	if !ok {
		return ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("dispatcher: tool %q not found", call.Name),
		}
	}

	handlerInput := call.Input
	if tool.Schema != nil {
		coerced, errs := tool.Schema.CoerceAndValidate(call.Input)
		if len(errs) > 0 {
			return ToolResult{
				CallID: call.ID,
				Name:   call.Name,
				Err:    fmt.Errorf("dispatcher: validation failed for %q: %v", call.Name, errs),
			}
		}
		handlerInput = coerced
	}

	if tool.Handler == nil {
		return ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("dispatcher: tool %q has no handler", call.Name),
		}
	}

	result, err := tool.Handler(ctx, handlerInput)
	if err != nil {
		return ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("dispatcher: tool %q execution: %w", call.Name, err),
		}
	}

	return ToolResult{
		CallID:  call.ID,
		Name:    call.Name,
		Content: result,
	}
}

// DispatchAll executes multiple tool calls and returns results in the same
// order as the input calls. maxConcurrent is a per-batch limit; values <= 1
// execute sequentially.
func (d *Dispatcher) DispatchAll(ctx context.Context, calls []ToolCall, maxConcurrent int) []ToolResult {
	return d.DispatchAllWith(ctx, calls, maxConcurrent, nil)
}

// DispatchAllWith executes multiple tool calls and returns results in the same
// order as the input calls. maxConcurrent is a per-batch limit; values <= 1
// execute sequentially. If dispatch is nil, Dispatcher.Dispatch is used.
func (d *Dispatcher) DispatchAllWith(ctx context.Context, calls []ToolCall, maxConcurrent int, dispatch func(context.Context, ToolCall) ToolResult) []ToolResult {
	results := make([]ToolResult, len(calls))

	if dispatch == nil {
		dispatch = d.Dispatch
	}

	if maxConcurrent <= 1 {
		for i, call := range calls {
			if err := ctx.Err(); err != nil {
				for j := i; j < len(calls); j++ {
					results[j] = ToolResult{
						CallID: calls[j].ID,
						Name:   calls[j].Name,
						Err:    fmt.Errorf("dispatcher: context cancelled before tool %q started: %w", calls[j].Name, err),
					}
				}
				break
			}
			results[i] = dispatch(ctx, call)
		}
		return results
	}

	var (
		wg  sync.WaitGroup
		sem = make(chan struct{}, maxConcurrent)
	)

	launchStopped := false
	for i, call := range calls {
		select {
		case <-ctx.Done():
			err := ctx.Err()
			for j := i; j < len(calls); j++ {
				results[j] = ToolResult{
					CallID: calls[j].ID,
					Name:   calls[j].Name,
					Err:    fmt.Errorf("dispatcher: context cancelled before tool %q started: %w", calls[j].Name, err),
				}
			}
			launchStopped = true
		case sem <- struct{}{}:
			wg.Add(1)
			go func(idx int, c ToolCall) {
				defer wg.Done()
				defer func() { <-sem }()
				results[idx] = dispatch(ctx, c)
			}(i, call)
		}
		if launchStopped {
			break
		}
	}

	wg.Wait()
	return results
}

// ResultToString converts a ToolResult into a string suitable for sending back
// to the LLM as a tool response message.
func ResultToString(r ToolResult) string {
	if r.Err != nil {
		return fmt.Sprintf("Error: %s", r.Err.Error())
	}
	switch v := r.Content.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	}
}
