package engine

import (
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestPlanStepCommandsFinal(t *testing.T) {
	commands := PlanStepCommands(StepInput{}, StepOutput{Text: "done"}, nil)
	assertCommandKinds(t, commands,
		CommandAppendAssistantMessage,
		CommandCheckpoint,
		CommandSaveSession,
		CommandEmitOutput,
		CommandTraceEvent,
		CommandEmitOutput,
	)

	msg := commands[0].Data["message"].(llm.Message)
	if msg.Role != "assistant" || msg.Content != "done" {
		t.Fatalf("unexpected assistant message command: %+v", msg)
	}
	emitFinal := commands[3].Data
	if emitFinal["type"] != "final_text" || emitFinal["text"] != "done" {
		t.Fatalf("unexpected final output command: %+v", emitFinal)
	}
	event := commands[4].Data["event"].(Event)
	if event.Kind != EventStepCompleted || event.Data["status"] != "final" {
		t.Fatalf("unexpected trace event command: %+v", event)
	}
	if commands[5].Data["type"] != "done" {
		t.Fatalf("unexpected done output command: %+v", commands[5])
	}
}

func TestPlanStepCommandsTools(t *testing.T) {
	calls := []tools.ToolCall{
		{ID: "call_1", Name: "lookup", Input: map[string]any{"nested": map[string]any{"key": "value"}}},
	}
	commands := PlanStepCommands(StepInput{StepIndex: 2}, StepOutput{
		Text:      "checking",
		ToolCalls: calls,
	}, nil)
	assertCommandKinds(t, commands,
		CommandAppendAssistantMessage,
		CommandCheckpoint,
		CommandExecuteTools,
	)

	msg := commands[0].Data["message"].(llm.Message)
	if msg.Content != "checking" || len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "lookup" {
		t.Fatalf("unexpected assistant tool message command: %+v", msg)
	}
	execCalls := commands[2].Data["calls"].([]tools.ToolCall)
	execCalls[0].Input["nested"].(map[string]any)["key"] = "mutated"
	if got := calls[0].Input["nested"].(map[string]any)["key"]; got != "value" {
		t.Fatalf("expected execute tool command not to alias original calls, got %v", got)
	}
	if commands[2].Data["step_index"] != 2 {
		t.Fatalf("unexpected execute tool step index: %+v", commands[2].Data)
	}
}

func TestPlanToolResultCommands(t *testing.T) {
	results := []tools.ToolResult{{CallID: "call_1", Name: "lookup", Content: "result"}}
	commands := PlanToolResultCommands(results)
	assertCommandKinds(t, commands, CommandAppendToolResults, CommandCheckpoint)

	got := commands[0].Data["results"].([]tools.ToolResult)
	if len(got) != 1 || got[0].Content != "result" {
		t.Fatalf("unexpected tool result command: %+v", got)
	}
}

func TestPlanStepCommandsTransfer(t *testing.T) {
	metadata := map[string]string{"topic": "billing"}
	commands := PlanStepCommands(StepInput{}, StepOutput{Text: "routing"}, &TransferPlan{
		Call:        tools.ToolCall{ID: "transfer_1", Name: "transfer_to_billing", Input: map[string]any{"input": "billing issue"}},
		TargetAgent: "billing",
		Input:       "billing issue",
		Reason:      "billing",
		Metadata:    metadata,
	})
	assertCommandKinds(t, commands,
		CommandAppendAssistantMessage,
		CommandCheckpoint,
		CommandAppendToolResults,
		CommandCheckpoint,
		CommandApplyTransfer,
	)

	msg := commands[0].Data["message"].(llm.Message)
	if msg.Content != "routing" || len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "transfer_to_billing" {
		t.Fatalf("unexpected transfer assistant command: %+v", msg)
	}
	results := commands[2].Data["results"].([]tools.ToolResult)
	if len(results) != 1 || results[0].Content != "Transferred control to billing." {
		t.Fatalf("unexpected transfer ack command: %+v", results)
	}
	transferData := commands[4].Data
	if transferData["target_agent"] != "billing" || transferData["input"] != "billing issue" {
		t.Fatalf("unexpected transfer command: %+v", transferData)
	}
	commandMetadata := transferData["metadata"].(map[string]string)
	commandMetadata["topic"] = "mutated"
	if metadata["topic"] != "billing" {
		t.Fatalf("expected transfer command metadata not to alias original metadata, got %v", metadata)
	}
}

func assertCommandKinds(t *testing.T, commands []Command, want ...CommandKind) {
	t.Helper()
	if len(commands) != len(want) {
		t.Fatalf("expected %d commands, got %+v", len(want), commands)
	}
	for i, kind := range want {
		if commands[i].Kind != kind {
			t.Fatalf("command %d: expected %s, got %+v", i, kind, commands[i])
		}
	}
}
