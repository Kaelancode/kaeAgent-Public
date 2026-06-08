package agent

import (
	"context"
	"testing"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/store"
	"github.com/Kaelancode/kaeAgent-Public/store/inmem"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestEngineHooksBindProviderAndTools(t *testing.T) {
	provider := &fakeProvider{
		responses: []*llm.Response{
			{Content: []llm.ContentBlock{{Type: "text", Text: "provider ok"}}},
		},
	}
	registry := tools.NewRegistry()
	registry.Register(testToolWithHandler("lookup", func(_ context.Context, input map[string]any) (any, error) {
		return input["q"], nil
	}))
	rt := NewRuntime(RuntimeConfig{
		Provider: provider,
		Session:  NewSession(SessionConfig{Model: "model"}),
		Tools:    registry,
	})

	hooks := rt.newRunExecutor().engineTurnInput("hello").Hooks
	resp, err := hooks.Complete(context.Background(), &llm.Request{Model: "model"})
	if err != nil {
		t.Fatalf("Complete hook: %v", err)
	}
	if resp.Content[0].Text != "provider ok" {
		t.Fatalf("unexpected provider response: %+v", resp)
	}

	results, err := hooks.ExecuteTools(context.Background(), agentengine.ToolStep{
		StepIndex: 1,
		Calls: []tools.ToolCall{
			{ID: "call_1", Name: "lookup", Input: map[string]any{"q": "answer"}},
		},
		MaxConcurrency: 1,
	})
	if err != nil {
		t.Fatalf("ExecuteTools hook: %v", err)
	}
	if len(results) != 1 || results[0].Err != nil || results[0].Content != "answer" {
		t.Fatalf("unexpected tool results: %+v", results)
	}
}

func TestEngineHooksResolveSubagentAndFilterTransfer(t *testing.T) {
	root := NewAgent(AgentConfig{Name: "root", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model"})
	billing.RegisterTool(testTool("billing_lookup"))
	resolver := staticSubagentResolver{"billing": billing}

	rt := NewRuntime(RuntimeConfig{
		Provider:         &fakeProvider{},
		Agent:            root,
		RootAgent:        root,
		SubagentResolver: resolver,
		Session:          NewSession(SessionConfig{Model: "root-model"}),
		TransferInputFilter: func(data TransferInputData) (TransferInputData, error) {
			data.Input = data.Input + " filtered"
			data.Messages = []llm.Message{{Role: "user", Content: "filtered history"}}
			data.Metadata = map[string]string{"filtered": "true"}
			return data, nil
		},
	})

	hooks := rt.newRunExecutor().engineTurnInput("hello").Hooks
	view, ok := hooks.ResolveSubagent(context.Background(), "billing")
	if !ok {
		t.Fatal("expected billing subagent to resolve")
	}
	if view.Name != "billing" || view.Model != "billing-model" {
		t.Fatalf("unexpected subagent view: %+v", view)
	}
	if got := engineToolNames(view.Tools); !sameStrings(got, []string{"billing_lookup"}) {
		t.Fatalf("expected billing tools, got %v", got)
	}

	filtered, err := hooks.FilterTransfer(context.Background(), agentengine.TransferInputData{
		FromAgent: "root",
		ToAgent:   "billing",
		Input:     "handoff",
		Messages:  []llm.Message{{Role: "user", Content: "raw history"}},
		Metadata:  map[string]string{"raw": "true"},
	})
	if err != nil {
		t.Fatalf("FilterTransfer hook: %v", err)
	}
	if filtered.Input != "handoff filtered" {
		t.Fatalf("unexpected filtered input: %+v", filtered)
	}
	if len(filtered.Messages) != 1 || filtered.Messages[0].Content != "filtered history" {
		t.Fatalf("unexpected filtered messages: %+v", filtered.Messages)
	}
	if filtered.Metadata["filtered"] != "true" {
		t.Fatalf("unexpected filtered metadata: %+v", filtered.Metadata)
	}
}

func TestEngineHooksCompactCheckpointAndSaveSession(t *testing.T) {
	convStore := inmem.NewConversationStore()
	sessionStore := &capturingSessionStore{}
	compactor := &capturingCompactor{}
	session := NewSession(SessionConfig{Model: "model"})

	rt := NewRuntime(RuntimeConfig{
		Provider:          &fakeProvider{},
		Session:           session,
		ConversationStore: convStore,
		SessionStore:      sessionStore,
		Compactor:         compactor,
		UserID:            "user-1",
		AgentID:           "agent-1",
	})
	rt.AppendConversationMessage(llm.Message{Role: "user", Content: "hello"})

	hooks := rt.newRunExecutor().engineTurnInput("hello").Hooks
	out, err := hooks.Compact(context.Background(), compaction.Input{
		SessionID: session.ID,
		Messages:  []llm.Message{{Role: "user", Content: "compact me"}},
	}, false)
	if err != nil {
		t.Fatalf("Compact hook: %v", err)
	}
	if !compactor.compactCalled || out.Messages[0].Content != "compact me" {
		t.Fatalf("unexpected compact result: %+v", out)
	}
	if _, err := hooks.Compact(context.Background(), compaction.Input{Messages: out.Messages}, true); err != nil {
		t.Fatalf("ForceCompact hook: %v", err)
	}
	if !compactor.forceCalled {
		t.Fatal("expected force compaction hook to call ForceCompact")
	}

	if err := hooks.Checkpoint(context.Background(), agentengine.State{}); err != nil {
		t.Fatalf("Checkpoint hook: %v", err)
	}
	loaded, err := convStore.Load(context.Background(), session.ID)
	if err != nil {
		t.Fatalf("load checkpoint: %v", err)
	}
	if len(loaded) != 1 || loaded[0].Content != "hello" {
		t.Fatalf("unexpected checkpoint messages: %+v", loaded)
	}

	if err := hooks.SaveSession(context.Background(), agentengine.State{}); err != nil {
		t.Fatalf("SaveSession hook: %v", err)
	}
	if sessionStore.saved == nil || sessionStore.saved.ID != session.ID || sessionStore.saved.UserID != "user-1" || sessionStore.saved.AgentID != "agent-1" {
		t.Fatalf("unexpected saved session: %+v", sessionStore.saved)
	}
}

type capturingCompactor struct {
	compactCalled bool
	forceCalled   bool
}

func (c *capturingCompactor) Compact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	c.compactCalled = true
	return compaction.Output{Messages: compaction.CloneMessages(input.Messages)}, nil
}

func (c *capturingCompactor) ForceCompact(_ context.Context, input compaction.Input) (compaction.Output, error) {
	c.forceCalled = true
	return compaction.Output{Messages: compaction.CloneMessages(input.Messages), Compacted: true}, nil
}

type capturingSessionStore struct {
	saved *store.SessionData
}

func (s *capturingSessionStore) SaveSession(_ context.Context, data *store.SessionData) error {
	cp := *data
	cp.Metadata = cloneStringMap(data.Metadata)
	s.saved = &cp
	return nil
}

func (s *capturingSessionStore) LoadSession(context.Context, string) (*store.SessionData, error) {
	return nil, nil
}

func (s *capturingSessionStore) DeleteSession(context.Context, string) error {
	return nil
}

func (s *capturingSessionStore) ListSessions(context.Context, string, string) ([]store.SessionEntry, error) {
	return nil, nil
}

type staticSubagentResolver map[string]*Agent

func (r staticSubagentResolver) Get(name string) (*Agent, bool) {
	agentDef, ok := r[name]
	return agentDef, ok
}
