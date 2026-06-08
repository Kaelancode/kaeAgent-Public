package agent

import (
	"context"
	"fmt"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

type appliedEngineCommands struct {
	ToolCalls   []tools.ToolCall
	ToolResults []tools.ToolResult
}

type engineCommandOptions struct {
	output runOutputAdapter
	trace  *runTraceState
	result *runLoopResult
}

func (e *runExecutor) applyEngineCommands(ctx context.Context, commands []agentengine.Command) (appliedEngineCommands, error) {
	return e.applyEngineCommandsWithOptions(ctx, commands, engineCommandOptions{})
}

func (e *runExecutor) applyEngineCommandsWithOutput(ctx context.Context, commands []agentengine.Command, output runOutputAdapter) (appliedEngineCommands, error) {
	return e.applyEngineCommandsWithOptions(ctx, commands, engineCommandOptions{output: output})
}

func (e *runExecutor) applyEngineCommandsWithTrace(ctx context.Context, commands []agentengine.Command, trace *runTraceState, result *runLoopResult) (appliedEngineCommands, error) {
	return e.applyEngineCommandsWithOptions(ctx, commands, engineCommandOptions{trace: trace, result: result})
}

func (e *runExecutor) applyEngineCommandsWithOptions(ctx context.Context, commands []agentengine.Command, opts engineCommandOptions) (appliedEngineCommands, error) {
	var applied appliedEngineCommands
	for _, command := range commands {
		out, err := e.applyEngineCommand(ctx, command, opts)
		if err != nil {
			return appliedEngineCommands{}, err
		}
		applied.ToolCalls = append(applied.ToolCalls, out.ToolCalls...)
		applied.ToolResults = append(applied.ToolResults, out.ToolResults...)
	}
	return applied, nil
}

func (e *runExecutor) applyEngineCommand(ctx context.Context, command agentengine.Command, opts engineCommandOptions) (appliedEngineCommands, error) {
	switch command.Kind {
	case agentengine.CommandAppendUserMessage:
		msg, ok := command.Data["message"].(llm.Message)
		if !ok {
			return appliedEngineCommands{}, fmt.Errorf("runtime: engine command append user message missing message")
		}
		if msg.Role != "user" {
			return appliedEngineCommands{}, fmt.Errorf("runtime: engine command append user message role must be user")
		}
		e.rs.conv.appendOwned(cloneMessage(msg))
		return appliedEngineCommands{}, nil

	case agentengine.CommandAppendAssistantMessage:
		msg, ok := command.Data["message"].(llm.Message)
		if !ok {
			return appliedEngineCommands{}, fmt.Errorf("runtime: engine command append assistant message missing message")
		}
		e.rs.conv.appendOwned(cloneMessage(msg))
		return appliedEngineCommands{}, nil

	case agentengine.CommandAppendToolResults:
		results, ok := command.Data["results"].([]tools.ToolResult)
		if !ok {
			return appliedEngineCommands{}, fmt.Errorf("runtime: engine command append tool results missing results")
		}
		for _, result := range results {
			content := tools.ResultToString(result)
			e.rs.conv.appendOwned(llm.Message{
				Role:       "tool",
				Content:    content,
				ToolCallID: result.CallID,
				Name:       result.Name,
			})
			if opts.output != nil {
				if err := opts.output.EmitToolResult(ctx, result.CallID, result.Name, content); err != nil {
					return appliedEngineCommands{}, err
				}
			}
		}
		return appliedEngineCommands{ToolResults: cloneToolResults(results)}, nil

	case agentengine.CommandExecuteTools:
		calls, ok := command.Data["calls"].([]tools.ToolCall)
		if !ok {
			return appliedEngineCommands{}, fmt.Errorf("runtime: engine command execute tools missing calls")
		}
		return appliedEngineCommands{ToolCalls: cloneToolCalls(calls)}, nil

	case agentengine.CommandApplyTransfer:
		req, err := transferRequestFromEngineCommand(command)
		if err != nil {
			return appliedEngineCommands{}, err
		}
		if err := e.applyTransfer(req); err != nil {
			return appliedEngineCommands{}, fmt.Errorf("runtime: transfer: %w", err)
		}
		return appliedEngineCommands{}, nil

	case agentengine.CommandCheckpoint:
		if err := e.checkpoint(ctx); err != nil {
			return appliedEngineCommands{}, err
		}
		return appliedEngineCommands{}, nil

	case agentengine.CommandSaveSession:
		if err := e.saveSessionData(ctx); err != nil {
			return appliedEngineCommands{}, err
		}
		return appliedEngineCommands{}, nil

	case agentengine.CommandEmitOutput:
		if opts.output == nil {
			return appliedEngineCommands{}, nil
		}
		if err := emitEngineOutput(ctx, opts.output, command); err != nil {
			return appliedEngineCommands{}, err
		}
		return appliedEngineCommands{}, nil

	case agentengine.CommandTraceEvent:
		if err := e.applyEngineTraceEvent(command, opts.trace, opts.result); err != nil {
			return appliedEngineCommands{}, err
		}
		return appliedEngineCommands{}, nil

	default:
		return appliedEngineCommands{}, fmt.Errorf("runtime: unsupported engine command %q", command.Kind)
	}
}

func (e *runExecutor) applyEngineTraceEvent(command agentengine.Command, trace *runTraceState, result *runLoopResult) error {
	event, ok := command.Data["event"].(agentengine.Event)
	if !ok {
		return fmt.Errorf("runtime: engine command trace event missing event")
	}
	if event.Kind == agentengine.EventStepCompleted && event.Data["status"] == "final" {
		if trace != nil && result != nil {
			e.recordFinalTrace(trace, result)
		}
		return nil
	}
	return nil
}

func emitEngineOutput(ctx context.Context, output runOutputAdapter, command agentengine.Command) error {
	kind, ok := command.Data["type"].(string)
	if !ok || kind == "" {
		return fmt.Errorf("runtime: engine command emit output missing type")
	}
	switch kind {
	case "final_text":
		text, ok := command.Data["text"].(string)
		if !ok {
			return fmt.Errorf("runtime: engine command emit output final_text missing text")
		}
		return output.EmitFinalText(ctx, text)
	case "done":
		return output.EmitDone(ctx)
	default:
		return fmt.Errorf("runtime: unsupported engine output type %q", kind)
	}
}

func transferRequestFromEngineCommand(command agentengine.Command) (TransferRequest, error) {
	target, ok := command.Data["target_agent"].(string)
	if !ok || target == "" {
		return TransferRequest{}, fmt.Errorf("runtime: engine command apply transfer missing target_agent")
	}
	input, ok := command.Data["input"].(string)
	if !ok {
		return TransferRequest{}, fmt.Errorf("runtime: engine command apply transfer missing input")
	}

	req := TransferRequest{
		AgentName: target,
		Input:     input,
	}
	if metadata, ok := command.Data["metadata"].(map[string]string); ok {
		req.Metadata = cloneStringMap(metadata)
	} else if command.Data["metadata"] != nil {
		return TransferRequest{}, fmt.Errorf("runtime: engine command apply transfer metadata must be map[string]string")
	}
	return req, nil
}

func cloneToolCalls(calls []tools.ToolCall) []tools.ToolCall {
	if calls == nil {
		return nil
	}
	out := make([]tools.ToolCall, len(calls))
	for i, call := range calls {
		out[i] = tools.ToolCall{
			ID:    call.ID,
			Name:  call.Name,
			Input: cloneMap(call.Input),
		}
	}
	return out
}

func cloneToolResults(results []tools.ToolResult) []tools.ToolResult {
	if results == nil {
		return nil
	}
	out := make([]tools.ToolResult, len(results))
	for i, result := range results {
		out[i] = result
	}
	return out
}
