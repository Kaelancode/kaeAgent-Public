package engine

import (
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func PlanStepCommands(step StepInput, output StepOutput, transfer *TransferPlan) []Command {
	if transfer != nil {
		return planTransferCommands(output, *transfer)
	}
	if len(output.ToolCalls) > 0 {
		return planToolCommands(step, output)
	}
	return planFinalCommands(output)
}

func PlanToolResultCommands(results []tools.ToolResult) []Command {
	return []Command{
		{
			Kind: CommandAppendToolResults,
			Data: map[string]any{
				"results": cloneToolResults(results),
			},
		},
		{Kind: CommandCheckpoint},
	}
}

func planFinalCommands(output StepOutput) []Command {
	return []Command{
		{
			Kind: CommandAppendAssistantMessage,
			Data: map[string]any{
				"message": llm.Message{Role: "assistant", Content: output.Text},
			},
		},
		{Kind: CommandCheckpoint},
		{Kind: CommandSaveSession},
		{
			Kind: CommandEmitOutput,
			Data: map[string]any{
				"type": "final_text",
				"text": output.Text,
			},
		},
		{
			Kind: CommandTraceEvent,
			Data: map[string]any{
				"event": Event{
					Kind: EventStepCompleted,
					Data: map[string]any{
						"status": "final",
						"text":   output.Text,
					},
				},
			},
		},
		{
			Kind: CommandEmitOutput,
			Data: map[string]any{
				"type": "done",
			},
		},
	}
}

func planToolCommands(step StepInput, output StepOutput) []Command {
	return []Command{
		{
			Kind: CommandAppendAssistantMessage,
			Data: map[string]any{
				"message": assistantToolCallMessage(output.Text, output.ToolCalls),
			},
		},
		{Kind: CommandCheckpoint},
		{
			Kind: CommandExecuteTools,
			Data: map[string]any{
				"step_index":      step.StepIndex,
				"calls":           cloneToolCalls(output.ToolCalls),
				"max_concurrency": 0,
			},
		},
	}
}

func planTransferCommands(output StepOutput, transfer TransferPlan) []Command {
	ack := transfer.Acknowledgment
	if ack == "" {
		ack = "Transferred control to " + transfer.TargetAgent + "."
	}
	return []Command{
		{
			Kind: CommandAppendAssistantMessage,
			Data: map[string]any{
				"message": assistantToolCallMessage(output.Text, []tools.ToolCall{transfer.Call}),
			},
		},
		{Kind: CommandCheckpoint},
		{
			Kind: CommandAppendToolResults,
			Data: map[string]any{
				"results": []tools.ToolResult{
					{
						CallID:  transfer.Call.ID,
						Name:    transfer.Call.Name,
						Content: ack,
					},
				},
			},
		},
		{Kind: CommandCheckpoint},
		{
			Kind: CommandApplyTransfer,
			Data: map[string]any{
				"target_agent": transfer.TargetAgent,
				"input":        transfer.Input,
				"reason":       transfer.Reason,
				"metadata":     cloneStringMap(transfer.Metadata),
			},
		},
	}
}

func assistantToolCallMessage(text string, toolCalls []tools.ToolCall) llm.Message {
	msg := llm.Message{Role: "assistant"}
	if text != "" {
		msg.Content = text
	}
	msg.ToolCalls = make([]llm.ToolCall, len(toolCalls))
	for i, call := range toolCalls {
		msg.ToolCalls[i] = llm.ToolCall{
			ID:    call.ID,
			Name:  call.Name,
			Input: cloneMap(call.Input),
		}
	}
	return msg
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
