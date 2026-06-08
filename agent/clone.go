package agent

import "github.com/Kaelancode/kaeAgent-Public/llm"

func cloneSessionConfig(cfg SessionConfig) SessionConfig {
	out := cfg
	if cfg.Temperature != nil {
		v := *cfg.Temperature
		out.Temperature = &v
	}
	if cfg.BudgetConfig != nil {
		budgetCfg := *cfg.BudgetConfig
		out.BudgetConfig = &budgetCfg
	}
	return out
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
