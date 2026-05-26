package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/compaction/strategy/slidingwindow"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/store"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

type fakeProvider struct {
	responses []*llm.Response
	callIdx   int
	requests  []*llm.Request
}

func (f *fakeProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	f.requests = append(f.requests, cloneRequest(req))
	if f.callIdx >= len(f.responses) {
		return &llm.Response{
			Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
			FinishReason: "stop",
		}, nil
	}
	resp := f.responses[f.callIdx]
	f.callIdx++
	return resp, nil
}

func (f *fakeProvider) Stream(_ context.Context, req *llm.Request) (<-chan llm.Event, error) {
	f.requests = append(f.requests, cloneRequest(req))
	ch := make(chan llm.Event, 64)
	go func() {
		defer close(ch)
		if f.callIdx < len(f.responses) {
			resp := f.responses[f.callIdx]
			f.callIdx++
			for _, block := range resp.Content {
				if block.Type == "text" && block.Text != "" {
					ch <- llm.Event{Kind: llm.EventText, Text: &llm.TextDelta{Content: block.Text}}
				}
				if block.Type == "tool_call" && block.ToolCall != nil {
					inputJSON, _ := json.Marshal(block.ToolCall.Input)
					ch <- llm.Event{Kind: llm.EventToolCall, Tool: &llm.ToolCallDelta{
						ID:    block.ToolCall.ID,
						Name:  block.ToolCall.Name,
						Input: string(inputJSON),
					}}
				}
			}
			ch <- llm.Event{Kind: llm.EventUsage, Usage: &llm.UsageDelta{
				InputTokens:  resp.Usage.InputTokens,
				OutputTokens: resp.Usage.OutputTokens,
				TotalTokens:  resp.Usage.TotalTokens,
			}}
		}
		ch <- llm.Event{Kind: llm.EventDone}
	}()
	return ch, nil
}
func (f *fakeProvider) Models(_ context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (f *fakeProvider) Name() string                                      { return "fake" }

type blockingProvider struct {
	started  chan struct{}
	release  chan struct{}
	response *llm.Response
}

func (b *blockingProvider) Complete(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
	close(b.started)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-b.release:
		return b.response, nil
	}
}

func (b *blockingProvider) Stream(_ context.Context, _ *llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event)
	close(ch)
	return ch, nil
}

func (b *blockingProvider) Models(_ context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (b *blockingProvider) Name() string                                      { return "blocking" }

func cloneRequest(req *llm.Request) *llm.Request {
	if req == nil {
		return nil
	}
	messages := make([]llm.Message, len(req.Messages))
	copy(messages, req.Messages)
	toolsCopy := make([]llm.ToolDef, len(req.Tools))
	copy(toolsCopy, req.Tools)
	optionsCopy := make(map[string]any, len(req.Options))
	for k, v := range req.Options {
		optionsCopy[k] = v
	}
	var temperature *float32
	if req.Temperature != nil {
		v := *req.Temperature
		temperature = &v
	}
	var execution *llm.ExecutionContext
	if req.Execution != nil {
		execution = &llm.ExecutionContext{
			SessionID: req.Execution.SessionID,
			UserID:    req.Execution.UserID,
			AgentID:   req.Execution.AgentID,
			RunID:     req.Execution.RunID,
			StepIndex: req.Execution.StepIndex,
			Metadata:  cloneStringMap(req.Execution.Metadata),
		}
	}
	return &llm.Request{
		Model:       req.Model,
		Messages:    messages,
		Tools:       toolsCopy,
		MaxTokens:   req.MaxTokens,
		Temperature: temperature,
		Options:     optionsCopy,
		Execution:   execution,
	}
}

type fakeSessionStore struct {
	saveErr error
}

func (f *fakeSessionStore) SaveSession(_ context.Context, _ *store.SessionData) error {
	return f.saveErr
}

func (f *fakeSessionStore) LoadSession(_ context.Context, _ string) (*store.SessionData, error) {
	return nil, nil
}

func (f *fakeSessionStore) DeleteSession(_ context.Context, _ string) error {
	return nil
}

func (f *fakeSessionStore) ListSessions(_ context.Context, _, _ string) ([]store.SessionEntry, error) {
	return nil, nil
}

type fakeConversationStore struct {
	savedID       string
	savedMessages []llm.Message
	saveErr       error
}

func (f *fakeConversationStore) Save(_ context.Context, convID string, messages []llm.Message) error {
	if f.saveErr != nil {
		return f.saveErr
	}
	f.savedID = convID
	f.savedMessages = append([]llm.Message(nil), messages...)
	return nil
}

func (f *fakeConversationStore) Load(_ context.Context, _ string) ([]llm.Message, error) {
	return nil, nil
}

func (f *fakeConversationStore) Append(_ context.Context, _ string, _ []llm.Message) error {
	return nil
}

func (f *fakeConversationStore) Delete(_ context.Context, _ string) error {
	return nil
}

func TestRuntime_SimpleResponse(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Hello!"}},
				FinishReason: "stop",
				Usage:        llm.Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
			},
		},
	}

	session := NewSession(SessionConfig{
		Model:       "fake-model",
		MaxTokens:   100,
		Temperature: float32Ptr(0.5),
	})

	registry := tools.NewRegistry()
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	result, err := rt.Run(context.Background(), "Hi there")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", result)
	}

	msgs := rt.ConversationMessages()
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (user + assistant), got %d", len(msgs))
	}
}

func TestRuntime_NewRuntimeWithoutSessionUsesDefaultSession(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "ok"}},
				FinishReason: "stop",
			},
		},
	}

	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
	})

	if rt.SessionSnapshot().ID == "" {
		t.Fatal("expected default session to be created")
	}

	result, err := rt.Run(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "ok" {
		t.Fatalf("expected ok, got %q", result)
	}
}

func TestRuntime_RunContinuesAfterModelDrivenTransfer(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{Type: "text", Text: "Billing should take this."},
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
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
				},
				FinishReason: "tool_calls",
				Usage:        llm.Usage{InputTokens: 7, OutputTokens: 3, TotalTokens: 10},
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "I can help with your refund."}},
				FinishReason: "stop",
				Usage:        llm.Usage{InputTokens: 4, OutputTokens: 2, TotalTokens: 6},
			},
		},
	}

	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing"},
		BudgetConfig: &streaming.BudgetConfig{MaxTokens: 1000},
	})
	root.RegisterTool(tools.ToolDef{Name: "root_tool"})
	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
		BudgetConfig: &streaming.BudgetConfig{MaxTokens: 20},
	})
	billing.RegisterTool(tools.ToolDef{Name: "billing_tool"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	out, err := rt.Run(context.Background(), "I need a refund")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "I can help with your refund." {
		t.Fatalf("expected billing reply, got %q", out)
	}
	if len(provider.requests) != 2 {
		t.Fatalf("expected 2 provider requests, got %d", len(provider.requests))
	}

	firstTools := toolNames(provider.requests[0].Tools)
	if !containsString(firstTools, transferToolName("billing")) {
		t.Fatalf("expected first step tools to include %q, got %v", transferToolName("billing"), firstTools)
	}
	secondTools := toolNames(provider.requests[1].Tools)
	if !containsString(secondTools, "billing_tool") {
		t.Fatalf("expected second step tools to include billing_tool, got %v", secondTools)
	}
	if !containsString(secondTools, transferToolName("root")) {
		t.Fatalf("expected second step tools to include return-to-root transfer, got %v", secondTools)
	}
	if containsString(secondTools, "root_tool") {
		t.Fatalf("did not expect second step tools to include root_tool, got %v", secondTools)
	}
	if provider.requests[1].Execution == nil || provider.requests[1].Execution.AgentID != "billing" {
		t.Fatalf("expected second step agent id billing, got %#v", provider.requests[1].Execution)
	}
	if got := provider.requests[1].Execution.Metadata[TransferReasonMetadataKey]; got != "billing owns refund approvals" {
		t.Fatalf("expected transfer reason metadata, got %q", got)
	}
	if got := provider.requests[1].Execution.Metadata["priority"]; got != "high" {
		t.Fatalf("expected transfer metadata priority=high, got %q", got)
	}
	lastMsg := provider.requests[1].Messages[len(provider.requests[1].Messages)-1]
	if lastMsg.Role != "user" || lastMsg.Content != "Handle the refund and ask for the invoice ID." {
		t.Fatalf("expected transferred task as trailing user message, got %#v", lastMsg)
	}

	snap := rt.SessionSnapshot()
	if snap.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected active agent billing after run, got %q", snap.Metadata[ActiveAgentMetadataKey])
	}
	if snap.Budget.MaxTokens != 20 {
		t.Fatalf("expected transferred budget max tokens 20, got %d", snap.Budget.MaxTokens)
	}
	if snap.Budget.TotalInput != 11 || snap.Budget.TotalOutput != 5 {
		t.Fatalf("expected cumulative transferred budget usage 11/5, got %d/%d", snap.Budget.TotalInput, snap.Budget.TotalOutput)
	}
	if rt.agent == nil || rt.agent.Name() != "billing" {
		t.Fatalf("expected runtime agent to publish as billing, got %#v", rt.agent)
	}
}

func TestRuntime_ModelDrivenTransferCanReturnToRoot(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:   "to_billing",
							Name: transferToolName("billing"),
							Input: map[string]any{
								"input": "Handle the billing issue.",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:   "to_root",
							Name: transferToolName("root"),
							Input: map[string]any{
								"input": "The issue is not billing-specific; coordinator should continue.",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Back at root."}},
				FinishReason: "stop",
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
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	out, err := rt.Run(context.Background(), "start")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Back at root." {
		t.Fatalf("expected root final reply, got %q", out)
	}
	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to publish root after transfer back, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "root" {
		t.Fatalf("expected active agent root, got %q", got)
	}
	if len(provider.requests) != 3 {
		t.Fatalf("expected 3 provider requests, got %d", len(provider.requests))
	}
	if provider.requests[1].Execution == nil || provider.requests[1].Execution.AgentID != "billing" {
		t.Fatalf("expected second request to run as billing, got %#v", provider.requests[1].Execution)
	}
	if provider.requests[2].Execution == nil || provider.requests[2].Execution.AgentID != "root" {
		t.Fatalf("expected third request to run as root, got %#v", provider.requests[2].Execution)
	}
}

func TestRuntime_ModelDrivenTransferCanRouteToPeerSpecialist(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:   "to_billing",
							Name: transferToolName("billing"),
							Input: map[string]any{
								"input": "Handle the billing issue.",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:   "to_technical",
							Name: transferToolName("technical"),
							Input: map[string]any{
								"input": "The user now needs technical setup help.",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Technical final."}},
				FinishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing", "technical"},
	})
	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	technical := NewAgent(AgentConfig{
		Name:         "technical",
		Model:        "technical-model",
		SystemPrompt: "technical system",
	})
	registry := NewRegistry()
	registry.Register(root)
	registry.Register(billing)
	registry.Register(technical)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: registry,
	})

	out, err := rt.Run(context.Background(), "start")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Technical final." {
		t.Fatalf("expected technical final reply, got %q", out)
	}
	if rt.agent == nil || rt.agent.Name() != "technical" {
		t.Fatalf("expected runtime to publish technical, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "technical" {
		t.Fatalf("expected active agent technical, got %q", got)
	}
	if len(provider.requests) != 3 {
		t.Fatalf("expected 3 provider requests, got %d", len(provider.requests))
	}
	secondTools := toolNames(provider.requests[1].Tools)
	if !containsString(secondTools, transferToolName("technical")) {
		t.Fatalf("expected billing step to expose peer technical transfer, got %v", secondTools)
	}
	if provider.requests[2].Execution == nil || provider.requests[2].Execution.AgentID != "technical" {
		t.Fatalf("expected third request to run as technical, got %#v", provider.requests[2].Execution)
	}
}

func TestRuntime_RunContinuesAfterModelDrivenConsult(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
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
				},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Ask for invoice ID, charge date, and amount."}},
				FinishReason: "stop",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Please provide the invoice ID, charge date, and amount."}},
				FinishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"research"}})
	research := NewAgent(AgentConfig{Name: "research", Model: "research-model"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(research)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	out, err := rt.Run(context.Background(), "I was charged twice")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Please provide the invoice ID, charge date, and amount." {
		t.Fatalf("expected parent final answer, got %q", out)
	}
	if len(provider.requests) != 3 {
		t.Fatalf("expected 3 provider requests, got %d", len(provider.requests))
	}

	firstTools := toolNames(provider.requests[0].Tools)
	if !containsString(firstTools, consultToolName("research")) {
		t.Fatalf("expected first step tools to include %q, got %v", consultToolName("research"), firstTools)
	}
	if provider.requests[1].Execution == nil || provider.requests[1].Execution.AgentID != "research" {
		t.Fatalf("expected child consult request agent id research, got %#v", provider.requests[1].Execution)
	}
	if got := provider.requests[1].Execution.Metadata[ConsultReasonMetadataKey]; got != "research can summarize intake requirements" {
		t.Fatalf("expected consult reason metadata, got %q", got)
	}
	childLastMsg := provider.requests[1].Messages[len(provider.requests[1].Messages)-1]
	if childLastMsg.Role != "user" || childLastMsg.Content != "Find what information is needed for duplicate charge handling." {
		t.Fatalf("expected consult task as child user message, got %#v", childLastMsg)
	}

	parentMessages := provider.requests[2].Messages
	var sawConsultResult bool
	for _, msg := range parentMessages {
		if msg.Role == "tool" && msg.Name == consultToolName("research") && strings.Contains(msg.Content, "invoice ID") {
			sawConsultResult = true
		}
	}
	if !sawConsultResult {
		t.Fatalf("expected parent continuation to include consult tool result, got %#v", parentMessages)
	}
	if rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey] != "" {
		t.Fatalf("did not expect consult to update active agent metadata, got %v", rt.SessionSnapshot().Metadata)
	}
	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to remain on root agent after consult, got %#v", rt.agent)
	}
}

func TestRuntime_ModelDrivenTransferFallsBackToResponseTextWhenInputOmitted(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{Type: "text", Text: "Please continue this billing case."},
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "transfer_1",
							Name:  transferToolName("billing"),
							Input: map[string]any{},
						},
					},
				},
				FinishReason: "tool_calls",
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "Billing continued."}},
				FinishReason: "stop",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	out, err := rt.Run(context.Background(), "start")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Billing continued." {
		t.Fatalf("expected billing continuation, got %q", out)
	}

	lastMsg := provider.requests[1].Messages[len(provider.requests[1].Messages)-1]
	if lastMsg.Role != "user" || lastMsg.Content != "Please continue this billing case." {
		t.Fatalf("expected fallback transfer input as trailing user message, got %#v", lastMsg)
	}
}

func TestRuntime_ModelDrivenTransferRejectsMixedToolCalls(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "transfer_1",
							Name:  transferToolName("billing"),
							Input: map[string]any{},
						},
					},
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "tool_1",
							Name:  "lookup",
							Input: map[string]any{"q": "refund"},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	root.RegisterTool(tools.ToolDef{Name: "lookup"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	_, err := rt.Run(context.Background(), "start")
	if err == nil || !containsError(err, `cannot be combined with other tool calls`) {
		t.Fatalf("expected mixed transfer/tool call error, got %v", err)
	}
}

func TestRuntime_ModelDrivenTransferRejectsInvalidPayloadTypes(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:   "transfer_1",
							Name: transferToolName("billing"),
							Input: map[string]any{
								"input":  123,
								"reason": true,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	_, err := rt.Run(context.Background(), "start")
	if err == nil || !containsError(err, `transfer input field must be a string`) {
		t.Fatalf("expected invalid transfer payload error, got %v", err)
	}
}

func TestRuntime_ModelDrivenTransferRejectsUnavailableTarget(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "transfer_1",
							Name:  transferToolName("legal"),
							Input: map[string]any{},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	resolver := NewRegistry()
	resolver.Register(root)
	resolver.Register(billing)

	rt := NewRuntime(RuntimeConfig{
		Provider:         provider,
		Agent:            root,
		SubagentResolver: resolver,
	})

	_, err := rt.Run(context.Background(), "start")
	if err == nil || !containsError(err, `transfer target "legal" is not available to agent "root"`) {
		t.Fatalf("expected unavailable transfer target error, got %v", err)
	}
}

func TestRuntime_BuildRequestOmitsUnsetTemperature(t *testing.T) {
	session := NewSession(SessionConfig{
		Model:     "fake-model",
		MaxTokens: 100,
	})

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  session,
	})

	req := rt.newRunExecutor().buildRequest(nil, nil, 0)
	if req.Temperature != nil {
		t.Fatalf("expected nil temperature, got %v", *req.Temperature)
	}
}

func toolNames(defs []llm.ToolDef) []string {
	names := make([]string, 0, len(defs))
	for _, def := range defs {
		names = append(names, def.Name)
	}
	return names
}

func containsString(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

func containsError(err error, want string) bool {
	return err != nil && strings.Contains(err.Error(), want)
}

func TestRuntime_BuildRequestPreservesZeroTemperature(t *testing.T) {
	session := NewSession(SessionConfig{
		Model:       "fake-model",
		MaxTokens:   100,
		Temperature: float32Ptr(0),
	})

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  session,
	})

	req := rt.newRunExecutor().buildRequest(nil, nil, 0)
	if req.Temperature == nil {
		t.Fatal("expected temperature to be set")
	}
	if *req.Temperature != 0 {
		t.Fatalf("expected temperature 0, got %v", *req.Temperature)
	}
}

func TestRuntime_BuildRequestIncludesExecutionContext(t *testing.T) {
	session := NewSession(SessionConfig{
		Model:     "fake-model",
		MaxTokens: 100,
	})
	session.Metadata["source"] = "test"

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  session,
		UserID:   "user-1",
		AgentID:  "agent-1",
	})

	req := rt.newRunExecutor().buildRequest(nil, nil, 3)
	if req.Execution == nil {
		t.Fatal("expected execution context to be set")
	}
	if req.Execution.SessionID != session.ID {
		t.Fatalf("expected session id %q, got %q", session.ID, req.Execution.SessionID)
	}
	if req.Execution.UserID != "user-1" {
		t.Fatalf("expected user id user-1, got %q", req.Execution.UserID)
	}
	if req.Execution.AgentID != "agent-1" {
		t.Fatalf("expected agent id agent-1, got %q", req.Execution.AgentID)
	}
	if req.Execution.RunID == "" {
		t.Fatal("expected run id to be set")
	}
	if req.Execution.StepIndex != 3 {
		t.Fatalf("expected step index 3, got %d", req.Execution.StepIndex)
	}
	if req.Execution.Metadata["source"] != "test" {
		t.Fatalf("expected metadata source=test, got %#v", req.Execution.Metadata)
	}
}

func float32Ptr(v float32) *float32 {
	return &v
}

func TestRuntime_WithToolCall(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{
				Content: []llm.ContentBlock{
					{
						Type: "tool_call",
						ToolCall: &llm.ToolCall{
							ID:    "call_1",
							Name:  "greet",
							Input: map[string]any{"name": "Alice"},
						},
					},
				},
				FinishReason: "tool_calls",
				Usage:        llm.Usage{InputTokens: 15, OutputTokens: 10},
			},
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "I greeted Alice for you."}},
				FinishReason: "stop",
				Usage:        llm.Usage{InputTokens: 20, OutputTokens: 10},
			},
		},
	}

	registry := tools.NewRegistry()
	registry.Register(tools.ToolDef{
		Name: "greet",
		Handler: func(_ context.Context, input map[string]any) (any, error) {
			return "Hello, " + input["name"].(string) + "!", nil
		},
	})

	session := NewSession(SessionConfig{Model: "fake-model", MaxTokens: 100})
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
	})

	result, err := rt.Run(context.Background(), "Greet Alice")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "I greeted Alice for you." {
		t.Errorf("expected 'I greeted Alice for you.', got %q", result)
	}

	msgs := rt.ConversationMessages()
	// user, assistant(tool_call), tool(result), assistant(final)
	if len(msgs) != 4 {
		t.Fatalf("expected 4 messages, got %d", len(msgs))
	}
}

func TestRuntime_MaxSteps(t *testing.T) {
	provider := &fakeProvider{
		responses: make([]*llm.Response, 100),
	}
	for i := range provider.responses {
		provider.responses[i] = &llm.Response{
			Content: []llm.ContentBlock{
				{
					Type: "tool_call",
					ToolCall: &llm.ToolCall{
						ID: "call", Name: "loop", Input: map[string]any{},
					},
				},
			},
			FinishReason: "tool_calls",
		}
	}

	registry := tools.NewRegistry()
	registry.Register(tools.ToolDef{
		Name: "loop",
		Handler: func(_ context.Context, _ map[string]any) (any, error) {
			return "looping", nil
		},
	})

	session := NewSession(SessionConfig{Model: "fake-model"})
	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      registry,
		Dispatcher: tools.NewDispatcher(registry),
		MaxSteps:   3,
	})

	_, err := rt.Run(context.Background(), "loop forever")
	if err == nil {
		t.Fatal("expected max steps error")
	}
}

func TestRuntime_CheckpointUsesSessionID(t *testing.T) {
	session := NewSession(SessionConfig{Model: "fake-model"})
	conv := NewConversation(TrimSlidingWindow, 10, 100)
	store := &fakeConversationStore{}

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{
			responses: []*llm.Response{
				{
					Content:      []llm.ContentBlock{{Type: "text", Text: "ok"}},
					FinishReason: "stop",
				},
			},
		},
		Session:           session,
		Conversation:      conv,
		Tools:             tools.NewRegistry(),
		Dispatcher:        tools.NewDispatcher(tools.NewRegistry()),
		ConversationStore: store,
		SessionStore:      &fakeSessionStore{},
	})

	if _, err := rt.Run(context.Background(), "hello"); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if store.savedID != session.ID {
		t.Fatalf("expected checkpoint ID %q to match session ID, got %q", session.ID, store.savedID)
	}
}

func TestRuntime_RunReturnsErrorOnCheckpointFailure(t *testing.T) {
	store := &fakeConversationStore{saveErr: errors.New("boom")}

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{
			responses: []*llm.Response{
				{
					Content:      []llm.ContentBlock{{Type: "text", Text: "ok"}},
					FinishReason: "stop",
				},
			},
		},
		Session:           NewSession(SessionConfig{Model: "fake-model"}),
		Tools:             tools.NewRegistry(),
		Dispatcher:        tools.NewDispatcher(tools.NewRegistry()),
		ConversationStore: store,
	})

	_, err := rt.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected checkpoint error")
	}
	if got := err.Error(); got != "runtime: checkpoint conversation: boom" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRuntime_RunReturnsErrorOnSessionSaveFailure(t *testing.T) {
	sessionStore := &fakeSessionStore{saveErr: errors.New("boom")}

	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{
			responses: []*llm.Response{
				{
					Content:      []llm.ContentBlock{{Type: "text", Text: "ok"}},
					FinishReason: "stop",
				},
			},
		},
		Session:      NewSession(SessionConfig{Model: "fake-model"}),
		Tools:        tools.NewRegistry(),
		Dispatcher:   tools.NewDispatcher(tools.NewRegistry()),
		SessionStore: sessionStore,
	})

	_, err := rt.Run(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected session save error")
	}
	if got := err.Error(); got != "runtime: save session: boom" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRuntime_DefaultCompactorCompactsConversationAfterTurn(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{Content: []llm.ContentBlock{{Type: "text", Text: "one"}}, FinishReason: "stop"},
			{Content: []llm.ContentBlock{{Type: "text", Text: "two"}}, FinishReason: "stop"},
			{Content: []llm.ContentBlock{{Type: "text", Text: "three"}}, FinishReason: "stop"},
		},
	}

	session := NewSession(SessionConfig{
		Model:        "fake-model",
		TrimStrategy: TrimSlidingWindow,
		MaxHistory:   2,
	})

	rt := NewRuntime(RuntimeConfig{
		Provider:   provider,
		Session:    session,
		Tools:      tools.NewRegistry(),
		Dispatcher: tools.NewDispatcher(tools.NewRegistry()),
	})

	for _, input := range []string{"a", "b", "c"} {
		if _, err := rt.Run(context.Background(), input); err != nil {
			t.Fatalf("Run(%q): %v", input, err)
		}
	}

	msgs := rt.ConversationMessages()
	if got := len(msgs); got != 4 {
		t.Fatalf("expected compacted retained conversation to contain 2 retained turns, got %d messages", got)
	}
	if len(provider.requests) != 3 {
		t.Fatalf("expected 3 provider requests, got %d", len(provider.requests))
	}
	if msgs[0].Content != "b" || msgs[1].Content != "two" || msgs[2].Content != "c" || msgs[3].Content != "three" {
		t.Fatalf("expected compaction after the final turn to keep the latest retained turns, got %+v", msgs)
	}
}

func TestRuntime_PreflightGuardForcesCompactionNearContextLimit(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}, FinishReason: "stop"},
		},
	}

	session := NewSession(SessionConfig{
		Model:        "fake-model",
		TrimStrategy: TrimSlidingWindow,
		MaxHistory:   2,
	})
	conv := NewConversation(TrimSlidingWindow, 10, 100)
	conv.Append(llm.Message{Role: "user", Content: "1"})
	conv.Append(llm.Message{Role: "assistant", Content: "2"})
	conv.Append(llm.Message{Role: "user", Content: "3"})

	rt := NewRuntime(RuntimeConfig{
		Provider:           provider,
		Session:            session,
		Conversation:       conv,
		Tools:              tools.NewRegistry(),
		Dispatcher:         tools.NewDispatcher(tools.NewRegistry()),
		Compactor:          compaction.NewEngine(nil, slidingwindow.New(2), nil),
		ModelContextLimit:  24,
		OutputTokenReserve: 5,
	})

	if _, err := rt.Run(context.Background(), "4"); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(provider.requests) != 1 {
		t.Fatalf("expected 1 provider request, got %d", len(provider.requests))
	}
	if got := len(provider.requests[0].Messages); got != 2 {
		t.Fatalf("expected preflight-compacted request to contain 2 messages, got %d", got)
	}
}

func TestRuntime_RunPersistsToCapturedTargetAfterLoadState(t *testing.T) {
	provider := &blockingProvider{
		started: make(chan struct{}),
		release: make(chan struct{}),
		response: &llm.Response{
			Content:      []llm.ContentBlock{{Type: "text", Text: "done-a"}},
			FinishReason: "stop",
			Usage:        llm.Usage{InputTokens: 1, OutputTokens: 1, TotalTokens: 2},
		},
	}
	store := &fakeConversationStore{}

	sessionA := NewSession(SessionConfig{Model: "model-a"})
	convA := NewConversation(TrimSlidingWindow, 10, 100)
	rt := NewRuntime(RuntimeConfig{
		Provider:          provider,
		Session:           sessionA,
		Conversation:      convA,
		Tools:             tools.NewRegistry(),
		Dispatcher:        tools.NewDispatcher(tools.NewRegistry()),
		ConversationStore: store,
		SessionStore:      &fakeSessionStore{},
	})

	resultCh := make(chan string, 1)
	errCh := make(chan error, 1)
	go func() {
		result, err := rt.Run(context.Background(), "run-a")
		resultCh <- result
		errCh <- err
	}()

	<-provider.started

	sessionB := NewSession(SessionConfig{Model: "model-b"})
	convB := NewConversation(TrimSlidingWindow, 10, 100)
	convB.Append(llm.Message{Role: "system", Content: "session-b"})
	rt.LoadState(sessionB.Snapshot(), convB.Snapshot())

	close(provider.release)

	if err := <-errCh; err != nil {
		t.Fatalf("unexpected run error: %v", err)
	}
	if result := <-resultCh; result != "done-a" {
		t.Fatalf("expected run result %q, got %q", "done-a", result)
	}

	if store.savedID != sessionA.ID {
		t.Fatalf("expected captured session ID %q, got %q", sessionA.ID, store.savedID)
	}
	if got := rt.SessionSnapshot().ID; got != sessionB.ID {
		t.Fatalf("expected active runtime session %q, got %q", sessionB.ID, got)
	}
	activeMsgs := rt.ConversationMessages()
	if len(activeMsgs) != 1 || activeMsgs[0].Content != "session-b" {
		t.Fatalf("expected active runtime conversation to stay on session B, got %+v", activeMsgs)
	}
	if len(store.savedMessages) != 2 || store.savedMessages[0].Content != "run-a" || store.savedMessages[1].Content != "done-a" {
		t.Fatalf("expected persisted conversation for session A, got %+v", store.savedMessages)
	}
}

func TestRuntimeSnapshotsHandleNilState(t *testing.T) {
	var rt Runtime

	sessionSnap := rt.SessionSnapshot()
	if sessionSnap.Metadata == nil {
		t.Fatal("expected nil session snapshot to include empty metadata map")
	}

	convSnap := rt.ConversationSnapshot()
	if len(convSnap.Messages) != 0 {
		t.Fatalf("expected empty conversation snapshot, got %+v", convSnap.Messages)
	}

	if msgs := rt.ConversationMessages(); len(msgs) != 0 {
		t.Fatalf("expected empty conversation messages, got %+v", msgs)
	}
}
