package agent

import (
	"strings"
	"testing"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestParseFinalCommandPlan(t *testing.T) {
	commands := agentengine.PlanStepCommands(
		agentengine.StepInput{},
		agentengine.StepOutput{Text: "done"},
		nil,
	)

	plan, err := parseFinalCommandPlan(commands)
	if err != nil {
		t.Fatalf("parseFinalCommandPlan: %v", err)
	}
	if plan.appendResponse.Kind != agentengine.CommandAppendAssistantMessage {
		t.Fatalf("unexpected append command: %+v", plan.appendResponse)
	}
	if len(plan.persist) != 2 ||
		plan.persist[0].Kind != agentengine.CommandCheckpoint ||
		plan.persist[1].Kind != agentengine.CommandSaveSession {
		t.Fatalf("unexpected persistence commands: %+v", plan.persist)
	}
	if plan.emitFinal.Kind != agentengine.CommandEmitOutput ||
		plan.traceFinal.Kind != agentengine.CommandTraceEvent ||
		plan.emitDone.Kind != agentengine.CommandEmitOutput {
		t.Fatalf("unexpected completion commands: %+v", plan)
	}
}

func TestParseToolCommandPlanRejectsReorderedCommands(t *testing.T) {
	commands := agentengine.PlanStepCommands(
		agentengine.StepInput{StepIndex: 1},
		agentengine.StepOutput{ToolCalls: []tools.ToolCall{{ID: "call_1", Name: "lookup"}}},
		nil,
	)
	commands[1], commands[2] = commands[2], commands[1]

	_, err := parseToolCommandPlan(commands)
	if err == nil || !strings.Contains(err.Error(), `command 1 must be "checkpoint", got "execute_tools"`) {
		t.Fatalf("expected reordered command error, got %v", err)
	}
}

func TestParseTransferCommandPlanRejectsMissingCommand(t *testing.T) {
	commands := agentengine.PlanStepCommands(
		agentengine.StepInput{},
		agentengine.StepOutput{},
		&agentengine.TransferPlan{
			Call:        tools.ToolCall{ID: "call_1", Name: "transfer_to_billing"},
			TargetAgent: "billing",
			Input:       "billing issue",
		},
	)

	_, err := parseTransferCommandPlan(commands[:len(commands)-1])
	if err == nil || !strings.Contains(err.Error(), "expected 5 commands, got 4") {
		t.Fatalf("expected missing command error, got %v", err)
	}
}

func TestParseToolResultCommandPlan(t *testing.T) {
	commands, err := parseToolResultCommandPlan(agentengine.PlanToolResultCommands([]tools.ToolResult{
		{CallID: "call_1", Name: "lookup", Content: "result"},
	}))
	if err != nil {
		t.Fatalf("parseToolResultCommandPlan: %v", err)
	}
	if len(commands) != 2 ||
		commands[0].Kind != agentengine.CommandAppendToolResults ||
		commands[1].Kind != agentengine.CommandCheckpoint {
		t.Fatalf("unexpected tool result commands: %+v", commands)
	}
}
