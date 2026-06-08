package agent

import (
	"context"
	"fmt"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
)

type finalCommandPlan struct {
	appendResponse agentengine.Command
	persist        []agentengine.Command
	emitFinal      agentengine.Command
	traceFinal     agentengine.Command
	emitDone       agentengine.Command
}

type toolCommandPlan struct {
	prepare      []agentengine.Command
	executeTools agentengine.Command
}

type transferCommandPlan struct {
	prepare       []agentengine.Command
	acknowledge   []agentengine.Command
	applyTransfer agentengine.Command
}

func parseFinalCommandPlan(commands []agentengine.Command) (finalCommandPlan, error) {
	expected := []agentengine.CommandKind{
		agentengine.CommandAppendAssistantMessage,
		agentengine.CommandCheckpoint,
		agentengine.CommandSaveSession,
		agentengine.CommandEmitOutput,
		agentengine.CommandTraceEvent,
		agentengine.CommandEmitOutput,
	}
	if err := validateCommandSequence("final", commands, expected); err != nil {
		return finalCommandPlan{}, err
	}
	return finalCommandPlan{
		appendResponse: commands[0],
		persist: []agentengine.Command{
			commands[1],
			commands[2],
		},
		emitFinal:  commands[3],
		traceFinal: commands[4],
		emitDone:   commands[5],
	}, nil
}

func parseToolCommandPlan(commands []agentengine.Command) (toolCommandPlan, error) {
	expected := []agentengine.CommandKind{
		agentengine.CommandAppendAssistantMessage,
		agentengine.CommandCheckpoint,
		agentengine.CommandExecuteTools,
	}
	if err := validateCommandSequence("tool", commands, expected); err != nil {
		return toolCommandPlan{}, err
	}
	return toolCommandPlan{
		prepare: []agentengine.Command{
			commands[0],
			commands[1],
		},
		executeTools: commands[2],
	}, nil
}

func parseTransferCommandPlan(commands []agentengine.Command) (transferCommandPlan, error) {
	expected := []agentengine.CommandKind{
		agentengine.CommandAppendAssistantMessage,
		agentengine.CommandCheckpoint,
		agentengine.CommandAppendToolResults,
		agentengine.CommandCheckpoint,
		agentengine.CommandApplyTransfer,
	}
	if err := validateCommandSequence("transfer", commands, expected); err != nil {
		return transferCommandPlan{}, err
	}
	return transferCommandPlan{
		prepare: []agentengine.Command{
			commands[0],
			commands[1],
		},
		acknowledge: []agentengine.Command{
			commands[2],
			commands[3],
		},
		applyTransfer: commands[4],
	}, nil
}

func parseToolResultCommandPlan(commands []agentengine.Command) ([]agentengine.Command, error) {
	expected := []agentengine.CommandKind{
		agentengine.CommandAppendToolResults,
		agentengine.CommandCheckpoint,
	}
	if err := validateCommandSequence("tool result", commands, expected); err != nil {
		return nil, err
	}
	return []agentengine.Command{commands[0], commands[1]}, nil
}

func validateCommandSequence(name string, commands []agentengine.Command, expected []agentengine.CommandKind) error {
	if len(commands) != len(expected) {
		return fmt.Errorf("runtime: invalid %s command plan: expected %d commands, got %d", name, len(expected), len(commands))
	}
	for i, kind := range expected {
		if commands[i].Kind != kind {
			return fmt.Errorf("runtime: invalid %s command plan: command %d must be %q, got %q", name, i, kind, commands[i].Kind)
		}
	}
	return nil
}

func (e *runExecutor) executeFinalCommandPlan(ctx context.Context, commands []agentengine.Command, output runOutputAdapter, trace *runTraceState, result *runLoopResult) error {
	plan, err := parseFinalCommandPlan(commands)
	if err != nil {
		return err
	}
	if _, err := e.applyEngineCommands(ctx, []agentengine.Command{plan.appendResponse}); err != nil {
		return err
	}
	if err := e.compactConversation(ctx); err != nil {
		return fmt.Errorf("runtime: compact conversation: %w", err)
	}
	if _, err := e.applyEngineCommands(ctx, plan.persist); err != nil {
		return err
	}
	e.rt.publishRunState(e.rs)
	if _, err := e.applyEngineCommandsWithOutput(ctx, []agentengine.Command{plan.emitFinal}, output); err != nil {
		return err
	}
	if _, err := e.applyEngineCommandsWithTrace(ctx, []agentengine.Command{plan.traceFinal}, trace, result); err != nil {
		return err
	}
	_, err = e.applyEngineCommandsWithOutput(ctx, []agentengine.Command{plan.emitDone}, output)
	return err
}

func (e *runExecutor) executeToolCommandPlan(ctx context.Context, commands []agentengine.Command, output runOutputAdapter, trace *runTraceState, step int) error {
	plan, err := parseToolCommandPlan(commands)
	if err != nil {
		return err
	}
	if _, err := e.applyEngineCommands(ctx, plan.prepare); err != nil {
		return err
	}
	applied, err := e.applyEngineCommands(ctx, []agentengine.Command{plan.executeTools})
	if err != nil {
		return err
	}
	toolResults := e.dispatchWithTracing(ctx, trace.agentSpan, step, applied.ToolCalls)
	resultCommands, err := parseToolResultCommandPlan(agentengine.PlanToolResultCommands(toolResults))
	if err != nil {
		return err
	}
	_, err = e.applyEngineCommandsWithOutput(ctx, resultCommands, output)
	return err
}

func (e *runExecutor) executeTransferCommandPlan(ctx context.Context, commands []agentengine.Command, output runOutputAdapter, trace *runTraceState, result *runLoopResult) error {
	plan, err := parseTransferCommandPlan(commands)
	if err != nil {
		return err
	}
	if _, err := e.applyEngineCommands(ctx, plan.prepare); err != nil {
		return err
	}
	if _, err := e.applyEngineCommandsWithOutput(ctx, plan.acknowledge, output); err != nil {
		return err
	}
	fromAgent := e.rs.agentID
	if e.rs.activeAgent != nil && e.rs.activeAgent.Name() != "" {
		fromAgent = e.rs.activeAgent.Name()
	}
	if _, err := e.applyEngineCommands(ctx, []agentengine.Command{plan.applyTransfer}); err != nil {
		return err
	}
	e.rotateTransferTrace(trace, fromAgent, result.Transfer.Request.AgentName, result.Transfer.Request.Input, result.Text)
	return nil
}
