package engine

import (
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/schema"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestBuildStepInputCopiesStateAndTools(t *testing.T) {
	state := State{
		SessionID: "session",
		RunID:     "run",
		UserID:    "user",
		AgentID:   "agent",
		ActiveAgent: AgentView{
			Name: "assistant",
		},
		Messages: []llm.Message{
			{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{
					{ID: "call", Name: "lookup", Input: map[string]any{"nested": map[string]any{"key": "value"}}},
				},
			},
		},
		Metadata: map[string]string{"source": "test"},
	}
	availableTools := []tools.ToolDef{
		{Name: "lookup", Tags: []string{"read"}},
	}

	step := BuildStepInput(state, availableTools, "openai", 2)
	step.Messages[0].ToolCalls[0].Input["nested"].(map[string]any)["key"] = "mutated"
	step.AvailableTools[0].Tags[0] = "mutated"
	step.Metadata["source"] = "mutated"

	if got := state.Messages[0].ToolCalls[0].Input["nested"].(map[string]any)["key"]; got != "value" {
		t.Fatalf("expected state messages not to alias step input, got %v", got)
	}
	if availableTools[0].Tags[0] != "read" {
		t.Fatalf("expected tool tags not to alias step input, got %v", availableTools[0].Tags)
	}
	if state.Metadata["source"] != "test" {
		t.Fatalf("expected metadata not to alias step input, got %v", state.Metadata)
	}
	if step.SessionID != "session" || step.ProviderName != "openai" || step.StepIndex != 2 || step.AgentName != "assistant" {
		t.Fatalf("unexpected step input: %+v", step)
	}
}

func TestBuildRequestCopiesPayloadAndPreservesExecutionContext(t *testing.T) {
	temp := float32(0)
	step := StepInput{
		SessionID: "session",
		RunID:     "run",
		StepIndex: 3,
		UserID:    "user",
		AgentID:   "agent",
		Messages: []llm.Message{
			{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{
					{ID: "call", Name: "lookup", Input: map[string]any{"key": "value"}},
				},
			},
		},
		AvailableTools: []tools.ToolDef{
			{
				Name:        "lookup",
				Description: "Lookup data",
				Schema: &schema.Schema{
					Type: "object",
					Properties: map[string]*schema.Schema{
						"key": {Type: "string"},
					},
				},
			},
		},
		Metadata: map[string]string{"source": "test"},
	}
	req := BuildRequest(step, Config{
		Model:       "model",
		MaxTokens:   100,
		Temperature: &temp,
	})

	if req.Model != "model" || req.MaxTokens != 100 {
		t.Fatalf("unexpected request config: %+v", req)
	}
	if req.Temperature == nil || *req.Temperature != 0 {
		t.Fatalf("expected explicit zero temperature, got %#v", req.Temperature)
	}
	if req.Execution == nil || req.Execution.SessionID != "session" || req.Execution.RunID != "run" || req.Execution.StepIndex != 3 {
		t.Fatalf("unexpected execution context: %+v", req.Execution)
	}
	if req.Execution.Metadata["source"] != "test" {
		t.Fatalf("unexpected execution metadata: %+v", req.Execution.Metadata)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "lookup" {
		t.Fatalf("unexpected tools: %+v", req.Tools)
	}

	req.Messages[0].ToolCalls[0].Input["key"] = "mutated"
	req.Execution.Metadata["source"] = "mutated"
	if step.Messages[0].ToolCalls[0].Input["key"] != "value" {
		t.Fatalf("expected request messages not to alias step input, got %v", step.Messages[0].ToolCalls[0].Input)
	}
	if step.Metadata["source"] != "test" {
		t.Fatalf("expected execution metadata not to alias step input, got %v", step.Metadata)
	}
}

func TestBuildRequestOmitsUnsetTemperature(t *testing.T) {
	req := BuildRequest(StepInput{}, Config{Model: "model"})
	if req.Temperature != nil {
		t.Fatalf("expected nil temperature, got %#v", req.Temperature)
	}
}

func TestToolDefsToLLMClonesSchemaParameters(t *testing.T) {
	defs := ToolDefsToLLM([]tools.ToolDef{
		{
			Name: "lookup",
			Schema: &schema.Schema{
				Type: "object",
				Properties: map[string]*schema.Schema{
					"query": {Type: "string"},
				},
			},
		},
	})

	if len(defs) != 1 || defs[0].Name != "lookup" {
		t.Fatalf("unexpected llm tool defs: %+v", defs)
	}
	props := defs[0].Parameters["properties"].(map[string]any)
	props["query"].(map[string]any)["type"] = "number"

	defsAgain := ToolDefsToLLM([]tools.ToolDef{
		{
			Name: "lookup",
			Schema: &schema.Schema{
				Type: "object",
				Properties: map[string]*schema.Schema{
					"query": {Type: "string"},
				},
			},
		},
	})
	propsAgain := defsAgain[0].Parameters["properties"].(map[string]any)
	if got := propsAgain["query"].(map[string]any)["type"]; got != "string" {
		t.Fatalf("expected cloned schema parameters, got %v", got)
	}
}
