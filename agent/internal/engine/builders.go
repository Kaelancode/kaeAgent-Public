package engine

import (
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func BuildStepInput(state State, availableTools []tools.ToolDef, providerName string, stepIndex int) StepInput {
	return StepInput{
		SessionID:      state.SessionID,
		RunID:          state.RunID,
		StepIndex:      stepIndex,
		Messages:       cloneMessages(state.Messages),
		AvailableTools: cloneToolDefs(availableTools),
		ProviderName:   providerName,
		UserID:         state.UserID,
		AgentID:        state.AgentID,
		AgentName:      state.ActiveAgent.Name,
		Metadata:       cloneStringMap(state.Metadata),
	}
}

func BuildRequest(step StepInput, cfg Config) *llm.Request {
	return &llm.Request{
		Model:       cfg.Model,
		Messages:    cloneMessages(step.Messages),
		Tools:       ToolDefsToLLM(step.AvailableTools),
		MaxTokens:   cfg.MaxTokens,
		Temperature: cloneFloat32Ptr(cfg.Temperature),
		Execution: &llm.ExecutionContext{
			SessionID: step.SessionID,
			UserID:    step.UserID,
			AgentID:   step.AgentID,
			RunID:     step.RunID,
			StepIndex: step.StepIndex,
			Metadata:  cloneStringMap(step.Metadata),
		},
	}
}

func ToolDefsToLLM(stepTools []tools.ToolDef) []llm.ToolDef {
	llmTools := make([]llm.ToolDef, len(stepTools))
	for i, t := range stepTools {
		var params map[string]any
		if t.Schema != nil {
			params = t.Schema.ToMap()
		}
		llmTools[i] = llm.ToolDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  cloneMap(params),
		}
	}
	return llmTools
}

func cloneMessages(messages []llm.Message) []llm.Message {
	if messages == nil {
		return nil
	}

	out := make([]llm.Message, len(messages))
	for i, msg := range messages {
		out[i] = cloneMessage(msg)
	}
	return out
}

func cloneMessage(msg llm.Message) llm.Message {
	out := msg
	if len(msg.ToolCalls) > 0 {
		out.ToolCalls = make([]llm.ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			out.ToolCalls[i] = llm.ToolCall{
				ID:    tc.ID,
				Name:  tc.Name,
				Input: cloneMap(tc.Input),
			}
		}
	}
	return out
}

func cloneToolDefs(defs []tools.ToolDef) []tools.ToolDef {
	if defs == nil {
		return nil
	}

	out := make([]tools.ToolDef, len(defs))
	for i, def := range defs {
		out[i] = def
		out[i].Tags = append([]string(nil), def.Tags...)
	}
	return out
}

func cloneFloat32Ptr(v *float32) *float32 {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func cloneStringMap(input map[string]string) map[string]string {
	if input == nil {
		return nil
	}

	out := make(map[string]string, len(input))
	for k, v := range input {
		out[k] = v
	}
	return out
}

func cloneMap(input map[string]any) map[string]any {
	if input == nil {
		return nil
	}

	out := make(map[string]any, len(input))
	for k, v := range input {
		out[k] = cloneValue(v)
	}
	return out
}

func cloneSlice(input []any) []any {
	if input == nil {
		return nil
	}

	out := make([]any, len(input))
	for i, v := range input {
		out[i] = cloneValue(v)
	}
	return out
}

func cloneValue(v any) any {
	switch typed := v.(type) {
	case map[string]any:
		return cloneMap(typed)
	case []any:
		return cloneSlice(typed)
	default:
		return typed
	}
}
