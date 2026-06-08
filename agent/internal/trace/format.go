package trace

import (
	"encoding/json"
	"strings"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func ProviderName(providerName string) string {
	switch providerName {
	case "claude":
		return "anthropic"
	case "gemini":
		return "gcp.gemini"
	default:
		return providerName
	}
}

func MessagesToOTelFormat(msgs []llm.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		entry := map[string]any{"role": m.Role}
		parts := make([]map[string]any, 0)
		if m.Content != "" {
			parts = append(parts, TextPart(m.Content))
		}
		for _, tc := range m.ToolCalls {
			parts = append(parts, map[string]any{
				"type":      "tool_call",
				"id":        tc.ID,
				"name":      tc.Name,
				"arguments": tc.Input,
			})
		}
		if m.ToolCallID != "" {
			parts = append(parts, map[string]any{
				"type":   "tool_call_response",
				"id":     m.ToolCallID,
				"result": truncate(m.Content),
			})
		}
		if len(parts) > 0 {
			entry["parts"] = parts
		}
		result = append(result, entry)
	}
	return result
}

func ExtractContentFromOTelMsg(msg map[string]any) string {
	parts, ok := msg["parts"].([]map[string]any)
	if !ok {
		if content, ok := msg["content"].(string); ok {
			return content
		}
		return ""
	}
	for _, p := range parts {
		if p["type"] == "text" {
			if content, ok := p["content"].(string); ok {
				return content
			}
		}
	}
	return ""
}

func TextMessageJSON(role, content string) string {
	return MustJSON([]map[string]any{{
		"role":  role,
		"parts": []map[string]any{TextPart(content)},
	}})
}

func AssistantTextOutput(text, finishReason string) (string, string) {
	msg := map[string]any{
		"role":          "assistant",
		"finish_reason": finishReason,
		"parts":         []map[string]any{TextPart(text)},
	}
	return MustJSON([]map[string]any{msg}), text
}

func AssistantToolCallsOutputFromResponse(resp *llm.Response) (string, string) {
	outputParts := []map[string]any{}
	var toolNames []string
	if resp != nil {
		for _, block := range resp.Content {
			if block.Type != "tool_call" || block.ToolCall == nil {
				continue
			}
			outputParts = append(outputParts, ToolCallPart(block.ToolCall.ID, block.ToolCall.Name))
			toolNames = append(toolNames, block.ToolCall.Name)
		}
	}
	if len(outputParts) == 0 {
		outputParts = append(outputParts, TextPart(""))
	}
	finishReason := ""
	if resp != nil {
		finishReason = resp.FinishReason
	}
	msg := map[string]any{
		"role":          "assistant",
		"finish_reason": finishReason,
		"parts":         outputParts,
	}
	return MustJSON([]map[string]any{msg}), ToolCallsSummary(toolNames)
}

func AssistantToolCallsOutputFromTools(calls []tools.ToolCall, finishReason string) (string, string) {
	outputParts := make([]map[string]any, 0, len(calls))
	toolNames := make([]string, 0, len(calls))
	for _, call := range calls {
		outputParts = append(outputParts, ToolCallPart(call.ID, call.Name))
		toolNames = append(toolNames, call.Name)
	}
	msg := map[string]any{
		"role":          "assistant",
		"finish_reason": finishReason,
		"parts":         outputParts,
	}
	return MustJSON([]map[string]any{msg}), ToolCallsSummary(toolNames)
}

func ToolCallsSummary(names []string) string {
	return "tool_calls: " + strings.Join(names, ", ")
}

func ToolCallPart(id, name string) map[string]any {
	return map[string]any{
		"type": "tool_call",
		"tool_call": map[string]any{
			"id":   id,
			"name": name,
		},
	}
}

func TextPart(content string) map[string]any {
	return map[string]any{"type": "text", "content": truncate(content)}
}

func MustJSON(v any) string {
	data, _ := json.Marshal(v)
	return string(data)
}

func truncate(content string) string {
	if len(content) > 2000 {
		return content[:1997] + "..."
	}
	return content
}
