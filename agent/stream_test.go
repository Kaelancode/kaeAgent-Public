package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

type streamResponseStep struct {
	textChunks   []string
	rawEvents    []llm.Event
	toolCalls    []llm.ToolCall
	usage        llm.Usage
	finishReason string
	omitDone     bool
}

type fakeStreamProvider struct {
	steps    []streamResponseStep
	stepIdx  int
	requests []*llm.Request
}

func (f *fakeStreamProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	f.requests = append(f.requests, cloneRequest(req))
	if f.stepIdx >= len(f.steps) {
		return nil, errors.New("fake stream provider: complete called after all steps were consumed")
	}
	step := f.steps[f.stepIdx]
	f.stepIdx++

	resp := &llm.Response{
		Usage:        step.usage,
		FinishReason: step.finishReason,
	}

	if len(step.textChunks) > 0 {
		resp.Content = append(resp.Content, llm.ContentBlock{
			Type: "text",
			Text: strings.Join(step.textChunks, ""),
		})
	}
	for _, tc := range step.toolCalls {
		resp.Content = append(resp.Content, llm.ContentBlock{
			Type:     "tool_call",
			ToolCall: &tc,
		})
	}

	return resp, nil
}

func (f *fakeStreamProvider) Stream(_ context.Context, req *llm.Request) (<-chan llm.Event, error) {
	f.requests = append(f.requests, cloneRequest(req))
	if f.stepIdx >= len(f.steps) {
		ch := make(chan llm.Event, 1)
		ch <- llm.Event{Kind: llm.EventDone}
		close(ch)
		return ch, nil
	}

	step := f.steps[f.stepIdx]
	f.stepIdx++

	ch := make(chan llm.Event, 64)
	go func() {
		defer close(ch)

		if len(step.rawEvents) > 0 {
			for _, event := range step.rawEvents {
				ch <- event
			}
			if !step.omitDone {
				ch <- llm.Event{Kind: llm.EventDone}
			}
			return
		}

		for _, chunk := range step.textChunks {
			ch <- llm.Event{Kind: llm.EventText, Text: &llm.TextDelta{Content: chunk}}
		}

		for i, tc := range step.toolCalls {
			inputJSON, _ := json.Marshal(tc.Input)
			ch <- llm.Event{Kind: llm.EventToolCall, Tool: &llm.ToolCallDelta{
				Index: i,
				ID:    tc.ID,
				Name:  tc.Name,
				Input: string(inputJSON),
			}}
		}

		if step.usage.InputTokens > 0 || step.usage.OutputTokens > 0 {
			ch <- llm.Event{Kind: llm.EventUsage, Usage: &llm.UsageDelta{
				InputTokens:  step.usage.InputTokens,
				OutputTokens: step.usage.OutputTokens,
				TotalTokens:  step.usage.TotalTokens,
			}}
		}

		if !step.omitDone {
			ch <- llm.Event{Kind: llm.EventDone}
		}
	}()

	return ch, nil
}

func (f *fakeStreamProvider) Models(_ context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (f *fakeStreamProvider) Name() string                                      { return "fake-stream" }

func TestRuntime_StreamBudgetFailureDoesNotAppendUserMessage(t *testing.T) {
	session := NewSession(SessionConfig{
		Model:        "test-model",
		BudgetConfig: &streaming.BudgetConfig{MaxTokens: 1},
	})
	session.Budget.Add(2, 0)

	provider := &fakeStreamProvider{}
	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
		Session:  session,
	})

	ch, err := rt.Stream(context.Background(), "this should be rejected")
	if err != nil {
		t.Fatalf("stream setup failed: %v", err)
	}
	_, streamErr := collectEvents(ch)
	if streamErr == nil {
		t.Fatal("expected budget stream error")
	}
	if !strings.Contains(streamErr.Error(), "budget: token limit exceeded") {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}
	if len(provider.requests) != 0 {
		t.Fatalf("expected provider not to be called, got %d requests", len(provider.requests))
	}
	if msgs := rt.ConversationMessages(); len(msgs) != 0 {
		t.Fatalf("expected rejected stream not to append conversation messages, got %+v", msgs)
	}
}

func collectEvents(ch <-chan streaming.Event) ([]streaming.Event, error) {
	var events []streaming.Event
	var err error
	for e := range ch {
		events = append(events, e)
		if e.Kind == streaming.EventError && err == nil {
			err = e.Err
		}
	}
	return events, err
}

func TestStream_SimpleTextResponse(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks:   []string{"Hello", " there", "!"},
				usage:        llm.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
				finishReason: "stop",
			},
		},
	}

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	registry := tools.NewRegistry()
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	ch, err := rt.Stream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	var textEvents []string
	var finalText string
	var gotDone bool
	var gotUsage bool

	for _, e := range events {
		switch e.Kind {
		case streaming.EventText:
			textEvents = append(textEvents, e.Text.Content)
		case streaming.EventFinalText:
			finalText = e.Final.Content
		case streaming.EventDone:
			gotDone = true
		case streaming.EventUsage:
			gotUsage = true
		}
	}

	if len(textEvents) != 3 {
		t.Errorf("expected 3 text events, got %d", len(textEvents))
	}
	if finalText != "Hello there!" {
		t.Errorf("expected final text 'Hello there!', got %q", finalText)
	}
	if !gotDone {
		t.Error("expected EventDone")
	}
	if !gotUsage {
		t.Error("expected EventUsage")
	}

	msgs := rt.ConversationMessages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "Hello there!" {
		t.Errorf("expected assistant message 'Hello there!', got role=%s content=%q", msgs[1].Role, msgs[1].Content)
	}
}

func TestStream_ClosedProviderChannelWithoutTerminalEventFails(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks: []string{"partial"},
				usage:      llm.Usage{InputTokens: 3, OutputTokens: 1, TotalTokens: 4},
				omitDone:   true,
			},
		},
	}
	tracer := &recordingTracer{}
	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
		Session:  NewSession(SessionConfig{Model: "fake-model"}),
		Tracer:   tracer,
	})

	ch, err := rt.Stream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "stream ended without terminal event") {
		t.Fatalf("expected missing terminal event error, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) {
		t.Fatalf("expected EventError, got %+v", events)
	}
	if hasEventKind(events, streaming.EventDone) {
		t.Fatalf("did not expect EventDone, got %+v", events)
	}

	var endedWithCloseErr bool
	tracer.mu.Lock()
	ended := append([]error(nil), tracer.ended...)
	for _, err := range tracer.ended {
		if err != nil && strings.Contains(err.Error(), "stream ended without terminal event") {
			endedWithCloseErr = true
			break
		}
	}
	tracer.mu.Unlock()
	if !endedWithCloseErr {
		t.Fatalf("expected llm span to end with missing terminal event error, got %+v", ended)
	}
}

func TestStream_ProviderErrorEventWithoutDetailsFails(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks: []string{"partial"},
				rawEvents:  []llm.Event{{Kind: llm.EventError}},
			},
		},
	}
	tracer := &recordingTracer{}
	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
		Session:  NewSession(SessionConfig{Model: "fake-model"}),
		Tracer:   tracer,
	})

	ch, err := rt.Stream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "provider emitted error event without details") {
		t.Fatalf("expected provider error event failure, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) {
		t.Fatalf("expected EventError, got %+v", events)
	}
	if hasEventKind(events, streaming.EventDone) {
		t.Fatalf("did not expect EventDone, got %+v", events)
	}

	var endedWithProviderErr bool
	tracer.mu.Lock()
	ended := append([]error(nil), tracer.ended...)
	for _, err := range tracer.ended {
		if err != nil && strings.Contains(err.Error(), "provider emitted error event without details") {
			endedWithProviderErr = true
			break
		}
	}
	tracer.mu.Unlock()
	if !endedWithProviderErr {
		t.Fatalf("expected llm span to end with provider error, got %+v", ended)
	}
}

func TestStream_RecoversPanicAsEventError(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{textChunks: []string{"unreached"}},
		},
	}
	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
		Session:  NewSession(SessionConfig{Model: "fake-model"}),
		StreamingMiddleware: []StreamingMiddleware{
			func(StreamingHandler) StreamingHandler {
				return func(context.Context, *StreamingStep, chan<- streaming.Event) (*StreamingStepResult, error) {
					panic("middleware exploded")
				}
			},
		},
	})

	ch, err := rt.Stream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "runtime: stream panic: middleware exploded") {
		t.Fatalf("expected recovered panic error, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) {
		t.Fatalf("expected EventError, got %+v", events)
	}
	if hasEventKind(events, streaming.EventDone) {
		t.Fatalf("did not expect EventDone, got %+v", events)
	}
}

func TestStream_WithToolCall(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				toolCalls: []llm.ToolCall{
					{ID: "call_1", Name: "greet", Input: map[string]any{"name": "Alice"}},
				},
				usage:        llm.Usage{InputTokens: 15, OutputTokens: 10, TotalTokens: 25},
				finishReason: "tool_calls",
			},
			{
				textChunks:   []string{"I greeted Alice for you."},
				usage:        llm.Usage{InputTokens: 20, OutputTokens: 10, TotalTokens: 30},
				finishReason: "stop",
			},
		},
	}

	registry := tools.NewRegistry()
	registry.Register(testToolWithHandler("greet", func(_ context.Context, input map[string]any) (any, error) {
		return "Hello, " + input["name"].(string) + "!", nil
	}))

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	ch, err := rt.Stream(context.Background(), "Greet Alice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	var finalText string
	var toolCalls int
	var toolResults int
	var gotDone bool

	for _, e := range events {
		switch e.Kind {
		case streaming.EventToolCall:
			toolCalls++
		case streaming.EventToolResult:
			toolResults++
			if e.Result.Name != "greet" {
				t.Errorf("expected tool result name 'greet', got %q", e.Result.Name)
			}
		case streaming.EventFinalText:
			finalText = e.Final.Content
		case streaming.EventDone:
			gotDone = true
		}
	}

	if toolCalls != 1 {
		t.Errorf("expected 1 tool_call event, got %d", toolCalls)
	}
	if toolResults != 1 {
		t.Errorf("expected 1 tool_result event, got %d", toolResults)
	}
	if finalText != "I greeted Alice for you." {
		t.Errorf("expected final text 'I greeted Alice for you.', got %q", finalText)
	}
	if !gotDone {
		t.Error("expected EventDone")
	}

	msgs := rt.ConversationMessages()
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages (user, assistant, tool, assistant), got %d", len(msgs))
	}
}

func TestStream_ContinuesAfterModelDrivenTransfer(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks: []string{"Billing should take this."},
				toolCalls: []llm.ToolCall{
					{
						ID:   "transfer_1",
						Name: transferToolName("billing"),
						Input: map[string]any{
							"input":  "Handle the refund and ask for the invoice ID.",
							"reason": "billing owns refund approvals",
							"metadata": map[string]any{
								"priority": "high",
							},
						},
					},
				},
				finishReason: "tool_calls",
			},
			{
				textChunks:   []string{"I can help with your refund."},
				finishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing"},
	})
	root.RegisterTool(testTool("root_tool"))
	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	billing.RegisterTool(testTool("billing_tool"))

	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "I need a refund")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("stream error: %v", streamErr)
	}

	var finalText string
	var toolResults []string
	var gotDone bool
	for _, event := range events {
		switch event.Kind {
		case streaming.EventFinalText:
			finalText = event.Final.Content
		case streaming.EventToolResult:
			toolResults = append(toolResults, event.Result.Content)
		case streaming.EventDone:
			gotDone = true
		}
	}

	if finalText != "I can help with your refund." {
		t.Fatalf("expected billing final text, got %q", finalText)
	}
	if !gotDone {
		t.Fatal("expected EventDone")
	}
	if len(toolResults) != 1 || !strings.Contains(toolResults[0], "billing") {
		t.Fatalf("expected one transfer tool result mentioning billing, got %v", toolResults)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("expected 2 provider requests, got %d", len(provider.requests))
	}
	if provider.requests[1].Execution == nil || provider.requests[1].Execution.AgentID != "billing" {
		t.Fatalf("expected second streamed step agent id billing, got %#v", provider.requests[1].Execution)
	}
	if got := provider.requests[1].Execution.Metadata[TransferReasonMetadataKey]; got != "billing owns refund approvals" {
		t.Fatalf("expected streamed transfer reason metadata, got %q", got)
	}
	lastMsg := provider.requests[1].Messages[len(provider.requests[1].Messages)-1]
	if lastMsg.Role != "user" || lastMsg.Content != "Handle the refund and ask for the invoice ID." {
		t.Fatalf("expected streamed transfer task as trailing user message, got %#v", lastMsg)
	}
	if rt.agent == nil || rt.agent.Name() != "billing" {
		t.Fatalf("expected runtime agent to publish as billing, got %#v", rt.agent)
	}
}

func TestStream_ContinuesAfterModelDrivenConsult(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				toolCalls: []llm.ToolCall{
					{
						ID:   "consult_1",
						Name: consultToolName("research"),
						Input: map[string]any{
							"input":  "Find what information is needed for duplicate charge handling.",
							"reason": "research can summarize intake requirements",
							"metadata": map[string]any{
								"mode": "consult",
							},
						},
					},
				},
				finishReason: "tool_calls",
			},
			{
				textChunks:   []string{"Ask for invoice ID, charge date, and amount."},
				finishReason: "stop",
			},
			{
				textChunks:   []string{"Please provide the invoice ID, charge date, and amount."},
				finishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"research"}})
	research := NewAgent(AgentConfig{Name: "research", Model: "research-model"})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(research)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "I was charged twice")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("stream error: %v", streamErr)
	}

	var finalText string
	var consultResult string
	for _, event := range events {
		switch event.Kind {
		case streaming.EventFinalText:
			finalText = event.Final.Content
		case streaming.EventToolResult:
			if event.Result.Name == consultToolName("research") {
				consultResult = event.Result.Content
			}
		}
	}
	if finalText != "Please provide the invoice ID, charge date, and amount." {
		t.Fatalf("expected final parent text, got %q", finalText)
	}
	if !strings.Contains(consultResult, "invoice ID") {
		t.Fatalf("expected streamed consult tool result, got %q", consultResult)
	}
	if !hasEventKind(events, streaming.EventDone) {
		t.Fatal("expected EventDone")
	}
	if provider.requests[1].Execution == nil || provider.requests[1].Execution.AgentID != "research" {
		t.Fatalf("expected streamed child consult request agent id research, got %#v", provider.requests[1].Execution)
	}
	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to remain on root agent after streamed consult, got %#v", rt.agent)
	}
}

func TestStream_ModelDrivenTransferFallsBackToResponseTextWhenInputOmitted(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks: []string{"Please continue this billing case."},
				toolCalls: []llm.ToolCall{
					{ID: "transfer_1", Name: transferToolName("billing"), Input: map[string]any{}},
				},
				finishReason: "tool_calls",
			},
			{
				textChunks:   []string{"Billing continued."},
				finishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "start")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("stream error: %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventDone) {
		t.Fatal("expected EventDone")
	}
	lastMsg := provider.requests[1].Messages[len(provider.requests[1].Messages)-1]
	if lastMsg.Role != "user" || lastMsg.Content != "Please continue this billing case." {
		t.Fatalf("expected fallback transfer input as trailing user message, got %#v", lastMsg)
	}
}

func TestTransferStreamDoesNotCommitActiveAgentOnStreamError(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				rawEvents: []llm.Event{
					{Kind: llm.EventError, Err: errors.New("stream failed")},
				},
			},
		},
	}

	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing"},
	})
	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	ch, err := rt.TransferStream(context.Background(), resolver, TransferRequest{
		AgentName: "billing",
		Input:     "handle billing",
	})
	if err != nil {
		t.Fatalf("TransferStream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "stream failed") {
		t.Fatalf("expected stream failure, got %v", streamErr)
	}
	if hasEventKind(events, streaming.EventDone) {
		t.Fatalf("did not expect EventDone after stream failure, got %+v", events)
	}
	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to remain on root after failed transfer stream, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "" {
		t.Fatalf("did not expect active agent metadata to be committed, got %q", got)
	}
}

func TestTransferStreamCommitsActiveAgentOnEventDone(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{textChunks: []string{"billing done"}},
		},
	}

	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing"},
	})
	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	ch, err := rt.TransferStream(context.Background(), resolver, TransferRequest{
		AgentName: "billing",
		Input:     "handle billing",
	})
	if err != nil {
		t.Fatalf("TransferStream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("Collect: %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventDone) {
		t.Fatalf("expected EventDone, got %+v", events)
	}
	if rt.agent == nil || rt.agent.Name() != "billing" {
		t.Fatalf("expected runtime to commit billing after successful transfer stream, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "billing" {
		t.Fatalf("expected active agent billing after stream done, got %q", got)
	}
}

func TestStream_ModelDrivenTransferRejectsMixedToolCalls(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				toolCalls: []llm.ToolCall{
					{ID: "transfer_1", Name: transferToolName("billing"), Input: map[string]any{}},
					{ID: "tool_1", Name: "lookup", Input: map[string]any{"q": "refund"}},
				},
				finishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	root.RegisterTool(testTool("lookup"))
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "start")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "cannot be combined with other tool calls") {
		t.Fatalf("expected mixed transfer/tool call error, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) {
		t.Fatalf("expected EventError, got %+v", events)
	}
	if hasEventKind(events, streaming.EventDone) {
		t.Fatalf("did not expect EventDone on transfer failure, got %+v", events)
	}
	if hasEventKind(events, streaming.EventFinalText) {
		t.Fatalf("did not expect EventFinalText on transfer failure, got %+v", events)
	}
}

func TestStream_ModelDrivenTransferRejectsInvalidPayloadTypes(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				toolCalls: []llm.ToolCall{
					{
						ID:   "transfer_1",
						Name: transferToolName("billing"),
						Input: map[string]any{
							"input":  123,
							"reason": true,
						},
					},
				},
				finishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "start")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), "transfer input field must be a string") {
		t.Fatalf("expected invalid transfer payload error, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) || hasEventKind(events, streaming.EventDone) {
		t.Fatalf("expected EventError without EventDone, got %+v", events)
	}
}

func TestStream_ModelDrivenTransferRejectsUnavailableTarget(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				toolCalls: []llm.ToolCall{
					{ID: "transfer_1", Name: transferToolName("legal"), Input: map[string]any{}},
				},
				finishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	ch, err := rt.Stream(context.Background(), "start")
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil || !strings.Contains(streamErr.Error(), `transfer target "legal" is not available to agent "root"`) {
		t.Fatalf("expected unavailable transfer target error, got %v", streamErr)
	}
	if !hasEventKind(events, streaming.EventError) || hasEventKind(events, streaming.EventDone) {
		t.Fatalf("expected EventError without EventDone, got %+v", events)
	}
}

func TestStream_ToolCallFragmentsUseProviderIndex(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				rawEvents: []llm.Event{
					{Kind: llm.EventToolCall, Tool: &llm.ToolCallDelta{Index: 0, ID: "call_1", Name: "greet"}},
					{Kind: llm.EventToolCall, Tool: &llm.ToolCallDelta{Index: 0, Input: `{"na`}},
					{Kind: llm.EventToolCall, Tool: &llm.ToolCallDelta{Index: 0, Input: `me":"Alice"}`}},
				},
				usage:        llm.Usage{InputTokens: 15, OutputTokens: 10, TotalTokens: 25},
				finishReason: "tool_calls",
			},
			{
				textChunks:   []string{"I greeted Alice for you."},
				usage:        llm.Usage{InputTokens: 20, OutputTokens: 10, TotalTokens: 30},
				finishReason: "stop",
			},
		},
	}

	registry := tools.NewRegistry()
	registry.Register(testToolWithHandler("greet", func(_ context.Context, input map[string]any) (any, error) {
		return "Hello, " + input["name"].(string) + "!", nil
	}))

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	ch, err := rt.Stream(context.Background(), "Greet Alice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr != nil {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	var toolCallFragments int
	var toolResults int
	var finalText string

	for _, e := range events {
		switch e.Kind {
		case streaming.EventToolCall:
			toolCallFragments++
			if e.Tool.Index != 0 {
				t.Fatalf("expected provider index 0, got %d", e.Tool.Index)
			}
		case streaming.EventToolResult:
			toolResults++
		case streaming.EventFinalText:
			finalText = e.Final.Content
		}
	}

	if toolCallFragments != 3 {
		t.Fatalf("expected 3 tool call fragments, got %d", toolCallFragments)
	}
	if toolResults != 1 {
		t.Fatalf("expected exactly 1 tool result, got %d", toolResults)
	}
	if finalText != "I greeted Alice for you." {
		t.Fatalf("expected final text %q, got %q", "I greeted Alice for you.", finalText)
	}
}

func hasEventKind(events []streaming.Event, kind streaming.EventKind) bool {
	for _, event := range events {
		if event.Kind == kind {
			return true
		}
	}
	return false
}

func TestStream_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{textChunks: []string{"hello"}, finishReason: "stop"},
		},
	}

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	registry := tools.NewRegistry()
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	ch, err := rt.Stream(ctx, "test")
	if err != nil {
		t.Fatalf("unexpected error from Stream(): %v", err)
	}

	events, _ := collectEvents(ch)
	hasError := false
	for _, e := range events {
		if e.Kind == streaming.EventError {
			hasError = true
		}
	}
	if !hasError {
		t.Error("expected an error event from cancelled context")
	}
}

func TestStream_EmitsErrorOnCheckpointFailure(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks:   []string{"hello"},
				finishReason: "stop",
			},
		},
	}

	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100}),
		Tools:      tools.NewRegistry(),
		Dispatcher: tools.NewDispatcher(tools.NewRegistry()),
		ConversationStore: &fakeConversationStore{
			saveErr: errors.New("boom"),
		},
	})

	ch, err := rt.Stream(context.Background(), "test")
	if err != nil {
		t.Fatalf("unexpected error from Stream(): %v", err)
	}

	events, streamErr := collectEvents(ch)
	if streamErr == nil {
		t.Fatal("expected stream error")
	}
	if got := streamErr.Error(); got != "runtime: checkpoint conversation: boom" {
		t.Fatalf("unexpected stream error: %v", streamErr)
	}

	for _, e := range events {
		if e.Kind == streaming.EventDone {
			t.Fatal("did not expect EventDone after checkpoint failure")
		}
	}
}

func TestStream_MaxStepsExceeded(t *testing.T) {
	provider := &fakeStreamProvider{
		steps: make([]streamResponseStep, 100),
	}
	for i := range provider.steps {
		provider.steps[i] = streamResponseStep{
			toolCalls: []llm.ToolCall{
				{ID: "call_loop", Name: "loop", Input: map[string]any{}},
			},
			finishReason: "tool_calls",
		}
	}

	registry := tools.NewRegistry()
	registry.Register(testToolWithHandler("loop", func(_ context.Context, _ map[string]any) (any, error) {
		return "looping", nil
	}))

	session := NewSession(SessionConfig{Model: "fake-model"})
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
		MaxSteps:   3,
	})

	ch, err := rt.Stream(context.Background(), "loop forever")
	if err != nil {
		t.Fatalf("unexpected error from Stream(): %v", err)
	}

	events, _ := collectEvents(ch)
	hasError := false
	for _, e := range events {
		if e.Kind == streaming.EventError {
			hasError = true
		}
	}
	if !hasError {
		t.Error("expected an error event for max steps exceeded")
	}
}

func TestStream_StopsWhenConsumerDoesNotDrain(t *testing.T) {
	prevTimeout := streamSendTimeout
	streamSendTimeout = 20 * time.Millisecond
	defer func() { streamSendTimeout = prevTimeout }()

	chunks := make([]string, 256)
	for i := range chunks {
		chunks[i] = "x"
	}

	provider := &fakeStreamProvider{
		steps: []streamResponseStep{
			{
				textChunks:   chunks,
				finishReason: "stop",
			},
		},
	}

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	registry := tools.NewRegistry()
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	ch, err := rt.Stream(context.Background(), "Hi")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	time.Sleep(100 * time.Millisecond)

	done := make(chan struct{})
	go func() {
		for range ch {
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("expected stream goroutine to exit when consumer stops draining")
	}
}
