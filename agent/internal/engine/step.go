package engine

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func ExecuteStep(ctx context.Context, step StepInput, cfg Config, hooks Hooks) (StepOutput, error) {
	if hooks.Complete == nil {
		return StepOutput{}, fmt.Errorf("engine: complete hook is nil")
	}

	req := BuildRequest(step, cfg)
	resp, err := hooks.Complete(ctx, req)
	if err != nil {
		return StepOutput{}, fmt.Errorf("engine: complete: %w", err)
	}
	if resp == nil {
		return StepOutput{}, fmt.Errorf("engine: complete returned nil response")
	}

	return StepOutput{
		Request:   req,
		Response:  resp,
		Text:      ExtractResponseText(resp),
		ToolCalls: ExtractToolCalls(resp),
		Usage:     resp.Usage,
	}, nil
}

func ExtractResponseText(resp *llm.Response) string {
	if resp == nil {
		return ""
	}
	for _, block := range resp.Content {
		if block.Type == "text" {
			return block.Text
		}
	}
	return ""
}

func ExtractToolCalls(resp *llm.Response) []tools.ToolCall {
	if resp == nil {
		return nil
	}

	var toolCalls []tools.ToolCall
	for _, block := range resp.Content {
		if block.Type == "tool_call" && block.ToolCall != nil {
			toolCalls = append(toolCalls, tools.ToolCall{
				ID:    block.ToolCall.ID,
				Name:  block.ToolCall.Name,
				Input: cloneMap(block.ToolCall.Input),
			})
		}
	}
	return toolCalls
}
