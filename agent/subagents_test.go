package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
)

type subagentTestProvider struct {
	responses   []*llm.Response
	lastRequest *llm.Request
	callIdx     int
}

func (p *subagentTestProvider) Complete(_ context.Context, req *llm.Request) (*llm.Response, error) {
	p.lastRequest = req
	if p.callIdx >= len(p.responses) {
		return &llm.Response{
			Content:      []llm.ContentBlock{{Type: "text", Text: "done"}},
			FinishReason: "stop",
		}, nil
	}
	resp := p.responses[p.callIdx]
	p.callIdx++
	return resp, nil
}

func (p *subagentTestProvider) Stream(_ context.Context, req *llm.Request) (<-chan llm.Event, error) {
	p.lastRequest = req
	ch := make(chan llm.Event, 2)
	ch <- llm.Event{Kind: llm.EventText, Text: &llm.TextDelta{Content: "stream"}}
	ch <- llm.Event{Kind: llm.EventDone}
	close(ch)
	return ch, nil
}

func (p *subagentTestProvider) Models(context.Context) ([]llm.ModelInfo, error) { return nil, nil }
func (p *subagentTestProvider) Name() string                                    { return "subagent-test" }

type staticResolver map[string]*Agent

func (r staticResolver) Get(name string) (*Agent, bool) {
	a, ok := r[name]
	return a, ok
}

var _ SubagentResolver = (staticResolver)(nil)

func TestRuntimeConsultUsesExplicitContextOnly(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "consulted"}},
				FinishReason: "stop",
			},
		},
	}

	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent: NewAgent(AgentConfig{
			Name:         "root",
			Model:        "root-model",
			SystemPrompt: "root system",
			Subagents:    []string{"researcher"},
		}),
		UserID:  "user_1",
		AgentID: "root",
	})
	parent.SetSessionMetadata("parent_only", "yes")
	parent.AppendConversationMessage(llm.Message{Role: "assistant", Content: "hidden parent history"})

	resolver := staticResolver{
		"researcher": NewAgent(AgentConfig{
			Name:         "researcher",
			Model:        "child-model",
			SystemPrompt: "child system",
		}),
	}

	out, err := parent.Consult(context.Background(), resolver, ConsultRequest{
		AgentName: "researcher",
		Input:     "find sources",
		Context:   []llm.Message{{Role: "user", Content: "explicit context"}},
		Metadata:  map[string]string{"scope": "consult"},
	})
	if err != nil {
		t.Fatalf("Consult: %v", err)
	}
	if out != "consulted" {
		t.Fatalf("expected consulted, got %q", out)
	}
	if provider.lastRequest == nil || provider.lastRequest.Execution == nil {
		t.Fatal("expected execution context on consult request")
	}
	if provider.lastRequest.Execution.SessionID != parent.SessionSnapshot().ID {
		t.Fatalf("expected consult session id %q, got %q", parent.SessionSnapshot().ID, provider.lastRequest.Execution.SessionID)
	}
	if provider.lastRequest.Execution.AgentID != "researcher" {
		t.Fatalf("expected consult agent id researcher, got %q", provider.lastRequest.Execution.AgentID)
	}
	if provider.lastRequest.Execution.Metadata["scope"] != "consult" {
		t.Fatalf("expected consult metadata to include scope=consult, got %v", provider.lastRequest.Execution.Metadata)
	}
	if _, ok := provider.lastRequest.Execution.Metadata["parent_only"]; ok {
		t.Fatalf("did not expect consult to inherit parent metadata, got %v", provider.lastRequest.Execution.Metadata)
	}

	var joined []string
	for _, msg := range provider.lastRequest.Messages {
		joined = append(joined, msg.Content)
	}
	text := strings.Join(joined, "\n")
	if !strings.Contains(text, "explicit context") {
		t.Fatalf("expected explicit context in child request, got %q", text)
	}
	if strings.Contains(text, "hidden parent history") {
		t.Fatalf("did not expect parent history to leak into consult request, got %q", text)
	}
}

func TestRuntimeTransferSetsActiveAgentAndMergesMetadata(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "billing reply"}},
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
	root.RegisterTool(testToolWithHandler("root_tool", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}))
	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent:    root,
		UserID:   "user_1",
		AgentID:  "root",
	})
	parent.SetSessionMetadata("tenant", "acme")

	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	billing.RegisterTool(testToolWithHandler("billing_tool", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}))
	resolver := staticResolver{
		"billing": billing,
	}

	out, err := parent.Transfer(context.Background(), resolver, TransferRequest{
		AgentName: "billing",
		Input:     "handle refund",
		Context:   []llm.Message{{Role: "user", Content: "refund case"}},
		Metadata:  map[string]string{"reason": "refund"},
	})
	if err != nil {
		t.Fatalf("Transfer: %v", err)
	}
	if out != "billing reply" {
		t.Fatalf("expected billing reply, got %q", out)
	}

	snap := parent.SessionSnapshot()
	if snap.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected active agent billing, got %q", snap.Metadata[ActiveAgentMetadataKey])
	}
	if provider.lastRequest == nil || provider.lastRequest.Execution == nil {
		t.Fatal("expected execution context on transfer request")
	}
	if provider.lastRequest.Execution.SessionID != snap.ID {
		t.Fatalf("expected transfer session id %q, got %q", snap.ID, provider.lastRequest.Execution.SessionID)
	}
	if provider.lastRequest.Execution.AgentID != "billing" {
		t.Fatalf("expected transfer agent id billing, got %q", provider.lastRequest.Execution.AgentID)
	}
	if provider.lastRequest.Execution.Metadata["tenant"] != "acme" {
		t.Fatalf("expected parent metadata to carry into transfer, got %v", provider.lastRequest.Execution.Metadata)
	}
	if provider.lastRequest.Execution.Metadata["reason"] != "refund" {
		t.Fatalf("expected transfer metadata to include reason=refund, got %v", provider.lastRequest.Execution.Metadata)
	}
	if provider.lastRequest.Execution.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected transfer metadata to include active agent, got %v", provider.lastRequest.Execution.Metadata)
	}
	if parent.agent == nil || parent.agent.Name() != "billing" {
		t.Fatalf("expected parent runtime to be rebound to billing agent, got %#v", parent.agent)
	}
	if _, ok := parent.tools.Get("billing_tool"); !ok {
		t.Fatal("expected billing tool to be active after transfer")
	}
	if _, ok := parent.tools.Get("root_tool"); ok {
		t.Fatal("did not expect root tool to remain active after transfer")
	}
}

func TestRuntimeActiveAgentFallbacks(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Agent: NewAgent(AgentConfig{Name: "root"}),
	})
	resolver := staticResolver{
		"billing": NewAgent(AgentConfig{Name: "billing"}),
	}

	if got := rt.ActiveAgent(resolver, "root"); got != "root" {
		t.Fatalf("expected root fallback on missing metadata, got %q", got)
	}

	rt.SetSessionMetadata(ActiveAgentMetadataKey, "unknown")
	if got := rt.ActiveAgent(resolver, "root"); got != "root" {
		t.Fatalf("expected root fallback on unknown metadata, got %q", got)
	}

	rt.SetSessionMetadata(ActiveAgentMetadataKey, "billing")
	if got := rt.ActiveAgent(resolver, "root"); got != "billing" {
		t.Fatalf("expected active agent billing, got %q", got)
	}
}

func TestRuntimeConsultStreamUsesCoreAPI(t *testing.T) {
	provider := &subagentTestProvider{}
	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent:    NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"researcher"}}),
	})
	resolver := staticResolver{
		"researcher": NewAgent(AgentConfig{Name: "researcher", Model: "child-model"}),
	}

	ch, err := parent.ConsultStream(context.Background(), resolver, ConsultRequest{
		AgentName: "researcher",
		Input:     "stream this",
	})
	if err != nil {
		t.Fatalf("ConsultStream: %v", err)
	}

	events, collectErr := streaming.Collect(context.Background(), ch)
	if collectErr != nil {
		t.Fatalf("Collect: %v", collectErr)
	}

	var sawFinal bool
	for _, event := range events {
		if event.Kind == streaming.EventFinalText && event.Final.Content == "stream" {
			sawFinal = true
		}
	}
	if !sawFinal {
		t.Fatalf("expected final streamed child text, got %+v", events)
	}
}

func TestRuntimeConsultRejectsUndeclaredSubagent(t *testing.T) {
	provider := &subagentTestProvider{}
	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent: NewAgent(AgentConfig{
			Name:      "root",
			Model:     "root-model",
			Subagents: []string{"billing"},
		}),
	})
	resolver := staticResolver{
		"legal": NewAgent(AgentConfig{Name: "legal", Model: "legal-model"}),
	}

	_, err := parent.Consult(context.Background(), resolver, ConsultRequest{
		AgentName: "legal",
		Input:     "review this",
	})
	if err == nil || !strings.Contains(err.Error(), `is not allowed to call subagent "legal"`) {
		t.Fatalf("expected undeclared subagent error, got %v", err)
	}
}

func TestRuntimeTransferRejectsUndeclaredSubagent(t *testing.T) {
	provider := &subagentTestProvider{}
	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent: NewAgent(AgentConfig{
			Name:      "root",
			Model:     "root-model",
			Subagents: []string{"billing"},
		}),
	})
	resolver := staticResolver{
		"legal": NewAgent(AgentConfig{Name: "legal", Model: "legal-model"}),
	}

	_, err := parent.Transfer(context.Background(), resolver, TransferRequest{
		AgentName: "legal",
		Input:     "take over",
	})
	if err == nil || !strings.Contains(err.Error(), `is not allowed to transfer to agent "legal"`) {
		t.Fatalf("expected undeclared subagent error, got %v", err)
	}
}

func TestRuntimeTransferAllowsReturnToRootAgent(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "root reply"}},
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
	if err := rt.ApplyTransfer(registry, TransferRequest{
		AgentName: "billing",
		Input:     "billing work",
	}); err != nil {
		t.Fatalf("ApplyTransfer to billing: %v", err)
	}

	out, err := rt.Transfer(context.Background(), registry, TransferRequest{
		AgentName: "root",
		Input:     "route this back to the coordinator",
	})
	if err != nil {
		t.Fatalf("Transfer back to root: %v", err)
	}
	if out != "root reply" {
		t.Fatalf("expected root reply, got %q", out)
	}
	if rt.agent == nil || rt.agent.Name() != "root" {
		t.Fatalf("expected runtime to return to root, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "root" {
		t.Fatalf("expected active agent metadata root, got %q", got)
	}
}

func TestRuntimeTransferAllowsRootDeclaredPeerSpecialist(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "technical reply"}},
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
	if err := rt.ApplyTransfer(registry, TransferRequest{
		AgentName: "billing",
		Input:     "billing work",
	}); err != nil {
		t.Fatalf("ApplyTransfer to billing: %v", err)
	}

	out, err := rt.Transfer(context.Background(), registry, TransferRequest{
		AgentName: "technical",
		Input:     "user now has a device setup issue",
	})
	if err != nil {
		t.Fatalf("Transfer to peer specialist: %v", err)
	}
	if out != "technical reply" {
		t.Fatalf("expected technical reply, got %q", out)
	}
	if rt.agent == nil || rt.agent.Name() != "technical" {
		t.Fatalf("expected runtime to bind technical, got %#v", rt.agent)
	}
	if got := rt.SessionSnapshot().Metadata[ActiveAgentMetadataKey]; got != "technical" {
		t.Fatalf("expected active agent technical, got %q", got)
	}
	if provider.lastRequest == nil || provider.lastRequest.Execution.AgentID != "technical" {
		t.Fatalf("expected request to run as technical, got %#v", provider.lastRequest)
	}
}

func TestRuntimeApplyTransferRebindsStateWithoutRunning(t *testing.T) {
	root := NewAgent(AgentConfig{
		Name:         "root",
		Model:        "root-model",
		SystemPrompt: "root system",
		Subagents:    []string{"billing"},
	})
	root.RegisterTool(testToolWithHandler("root_tool", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}))

	billing := NewAgent(AgentConfig{
		Name:         "billing",
		Model:        "billing-model",
		SystemPrompt: "billing system",
	})
	billing.RegisterTool(testToolWithHandler("billing_tool", func(context.Context, map[string]any) (any, error) {
		return "ok", nil
	}))

	rt := NewRuntime(RuntimeConfig{
		Agent: root,
	})
	rt.AppendConversationMessage(llm.Message{Role: "user", Content: "old context"})

	err := rt.ApplyTransfer(staticResolver{
		"billing": billing,
	}, TransferRequest{
		AgentName: "billing",
		Context: []llm.Message{
			{Role: "user", Content: "billing-only context"},
		},
		Metadata: map[string]string{"reason": "refund"},
	})
	if err != nil {
		t.Fatalf("ApplyTransfer: %v", err)
	}

	if rt.agent == nil || rt.agent.Name() != "billing" {
		t.Fatalf("expected runtime to bind billing agent, got %#v", rt.agent)
	}
	snap := rt.SessionSnapshot()
	if snap.Config.Model != "billing-model" {
		t.Fatalf("expected billing model after transfer, got %q", snap.Config.Model)
	}
	if snap.Config.SystemPrompt != "billing system" {
		t.Fatalf("expected billing system prompt after transfer, got %q", snap.Config.SystemPrompt)
	}
	if snap.Metadata[ActiveAgentMetadataKey] != "billing" {
		t.Fatalf("expected active agent billing, got %q", snap.Metadata[ActiveAgentMetadataKey])
	}
	if snap.Metadata["reason"] != "refund" {
		t.Fatalf("expected transfer metadata to be merged, got %v", snap.Metadata)
	}
	msgs := rt.ConversationMessages()
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[0].Content != "billing system" || msgs[1].Content != "billing-only context" {
		t.Fatalf("expected transfer context under billing system prompt, got %+v", msgs)
	}
	if _, ok := rt.tools.Get("billing_tool"); !ok {
		t.Fatal("expected billing tool registry after ApplyTransfer")
	}
	if _, ok := rt.tools.Get("root_tool"); ok {
		t.Fatal("did not expect root tool registry after ApplyTransfer")
	}
}

func TestRuntimeTransferUsesFilterWhenContextNotProvided(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "billing reply"}},
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
	parent := NewRuntime(RuntimeConfig{
		Provider:            provider,
		Agent:               root,
		TransferInputFilter: RemoveToolTransferInput(),
	})
	parent.AppendConversationMessage(llm.Message{
		Role: "assistant",
		ToolCalls: []llm.ToolCall{{
			ID:    "call_1",
			Name:  "lookup",
			Input: map[string]any{"id": "123"},
		}},
	})
	parent.AppendConversationMessage(llm.Message{
		Role:       "tool",
		Content:    "tool output",
		ToolCallID: "call_1",
		Name:       "lookup",
	})
	parent.AppendConversationMessage(llm.Message{
		Role:    "assistant",
		Content: "human-facing summary",
	})

	resolver := staticResolver{
		"billing": NewAgent(AgentConfig{
			Name:         "billing",
			Model:        "billing-model",
			SystemPrompt: "billing system",
		}),
	}

	_, err := parent.Transfer(context.Background(), resolver, TransferRequest{
		AgentName: "billing",
		Input:     "continue",
	})
	if err != nil {
		t.Fatalf("Transfer: %v", err)
	}

	if provider.lastRequest == nil {
		t.Fatal("expected request to be captured")
	}
	for _, msg := range provider.lastRequest.Messages {
		if msg.Role == "tool" {
			t.Fatalf("did not expect tool-role message in filtered transfer input: %+v", provider.lastRequest.Messages)
		}
		if len(msg.ToolCalls) > 0 || msg.ToolCallID != "" {
			t.Fatalf("did not expect tool-call chatter in filtered transfer input: %+v", provider.lastRequest.Messages)
		}
	}

	var sawSummary bool
	for _, msg := range provider.lastRequest.Messages {
		if msg.Content == "human-facing summary" {
			sawSummary = true
		}
	}
	if !sawSummary {
		t.Fatalf("expected filtered transfer input to keep non-tool conversation messages: %+v", provider.lastRequest.Messages)
	}
}

func TestRuntimeTransferExplicitContextOverridesFilter(t *testing.T) {
	provider := &subagentTestProvider{
		responses: []*llm.Response{
			{
				Content:      []llm.ContentBlock{{Type: "text", Text: "billing reply"}},
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
	parent := NewRuntime(RuntimeConfig{
		Provider: provider,
		Agent:    root,
		TransferInputFilter: func(data TransferInputData) (TransferInputData, error) {
			data.Messages = nil
			return data, nil
		},
	})

	resolver := staticResolver{
		"billing": NewAgent(AgentConfig{
			Name:         "billing",
			Model:        "billing-model",
			SystemPrompt: "billing system",
		}),
	}

	_, err := parent.Transfer(context.Background(), resolver, TransferRequest{
		AgentName: "billing",
		Input:     "continue",
		Context: []llm.Message{
			{Role: "user", Content: "explicit transfer context"},
		},
	})
	if err != nil {
		t.Fatalf("Transfer: %v", err)
	}

	if provider.lastRequest == nil {
		t.Fatal("expected request to be captured")
	}
	var sawExplicit bool
	for _, msg := range provider.lastRequest.Messages {
		if msg.Content == "explicit transfer context" {
			sawExplicit = true
		}
	}
	if !sawExplicit {
		t.Fatalf("expected explicit context to override transfer filter, got %+v", provider.lastRequest.Messages)
	}
}
