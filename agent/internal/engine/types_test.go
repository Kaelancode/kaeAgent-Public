package engine

import (
	"context"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestTurnInputAcceptsNeutralEngineValues(t *testing.T) {
	temp := float32(0.2)
	input := TurnInput{
		UserMessage: "hello",
		State: State{
			Generation: 1,
			SessionID:  "session",
			RunID:      "run",
			UserID:     "user",
			AgentID:    "root",
			ActiveAgent: AgentView{
				Name:         "root",
				Model:        "model",
				SystemPrompt: "system",
				MaxSteps:     3,
				Tools: []tools.ToolDef{
					{Name: "lookup"},
				},
				Subagents: []string{"billing"},
			},
			RootAgent: AgentView{Name: "root"},
			Messages:  []llm.Message{{Role: "user", Content: "hello"}},
			Metadata:  map[string]string{"active_agent": "root"},
			Budget:    BudgetSnapshot{MaxTokens: 100, TotalInput: 1},
		},
		Config: Config{
			Model:              "model",
			MaxTokens:          100,
			Temperature:        &temp,
			MaxSteps:           3,
			MaxToolConcurrency: 2,
			ModelContextLimit:  1000,
			OutputTokenReserve: 100,
		},
	}

	if input.State.ActiveAgent.Tools[0].Name != "lookup" {
		t.Fatalf("expected tool definition to be retained, got %+v", input.State.ActiveAgent.Tools)
	}
	if input.State.Budget.TotalInput != 1 {
		t.Fatalf("expected budget snapshot to be retained, got %+v", input.State.Budget)
	}
}

func TestHooksAreRuntimeSuppliedFunctions(t *testing.T) {
	called := false
	hooks := Hooks{
		Complete: func(context.Context, *llm.Request) (*llm.Response, error) {
			called = true
			return &llm.Response{Content: []llm.ContentBlock{{Type: "text", Text: "ok"}}}, nil
		},
		ExecuteTools: func(context.Context, ToolStep) ([]tools.ToolResult, error) {
			return []tools.ToolResult{{CallID: "call", Name: "tool", Content: "result"}}, nil
		},
		ResolveSubagent: func(_ context.Context, name string) (AgentView, bool) {
			return AgentView{Name: name}, true
		},
		FilterTransfer: func(_ context.Context, input TransferInputData) (TransferInputResult, error) {
			return TransferInputResult{Input: input.Input, Metadata: input.Metadata}, nil
		},
		Compact: func(_ context.Context, input compaction.Input, _ bool) (compaction.Output, error) {
			return compaction.Output{Messages: input.Messages}, nil
		},
	}

	resp, err := hooks.Complete(context.Background(), &llm.Request{})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if !called || resp.Content[0].Text != "ok" {
		t.Fatalf("expected complete hook to be called, got called=%v resp=%+v", called, resp)
	}
}

func TestCommandsAreDataOnly(t *testing.T) {
	out := TurnOutput{
		FinalText: "done",
		State:     State{SessionID: "session"},
		Commands: []Command{
			{Kind: CommandAppendUserMessage, Data: map[string]any{"content": "hello"}},
			{Kind: CommandApplyTransfer, Data: map[string]any{"target_agent": "billing"}},
		},
		Events: []Event{
			{Kind: EventTransfer, Data: map[string]any{"from": "root", "to": "billing"}},
		},
	}

	if out.Commands[0].Kind != CommandAppendUserMessage {
		t.Fatalf("expected append command, got %+v", out.Commands[0])
	}
	if out.Commands[1].Data["target_agent"] != "billing" {
		t.Fatalf("expected command data to be retained, got %+v", out.Commands[1])
	}
	if out.Events[0].Kind != EventTransfer {
		t.Fatalf("expected transfer event, got %+v", out.Events[0])
	}
}
