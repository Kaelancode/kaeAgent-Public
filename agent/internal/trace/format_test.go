package trace

import (
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestProviderName(t *testing.T) {
	tests := map[string]string{
		"claude":     "anthropic",
		"gemini":     "gcp.gemini",
		"openai":     "openai",
		"gcp.gemini": "gcp.gemini",
	}
	for input, want := range tests {
		if got := ProviderName(input); got != want {
			t.Fatalf("ProviderName(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestMessagesToOTelFormat(t *testing.T) {
	msgs := []llm.Message{
		{Role: "system", Content: "system prompt"},
		{
			Role: "assistant",
			ToolCalls: []llm.ToolCall{{
				ID:    "call_1",
				Name:  "lookup",
				Input: map[string]any{"id": "123"},
			}},
		},
		{Role: "tool", Content: "tool result", ToolCallID: "call_1"},
	}

	formatted := MessagesToOTelFormat(msgs)
	if len(formatted) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(formatted))
	}
	if got := ExtractContentFromOTelMsg(formatted[0]); got != "system prompt" {
		t.Fatalf("expected system content, got %q", got)
	}
	parts, ok := formatted[1]["parts"].([]map[string]any)
	if !ok || len(parts) != 1 {
		t.Fatalf("expected assistant tool call part, got %#v", formatted[1]["parts"])
	}
	if parts[0]["type"] != "tool_call" || parts[0]["name"] != "lookup" {
		t.Fatalf("unexpected tool call part: %#v", parts[0])
	}
}

func TestAssistantOutputs(t *testing.T) {
	textJSON, textSummary := AssistantTextOutput("hello", "stop")
	if textSummary != "hello" {
		t.Fatalf("expected text summary hello, got %q", textSummary)
	}
	if textJSON == "" {
		t.Fatal("expected text output JSON")
	}

	toolJSON, toolSummary := AssistantToolCallsOutputFromTools([]tools.ToolCall{
		{ID: "call_1", Name: "lookup"},
		{ID: "call_2", Name: "search"},
	}, "tool_calls")
	if toolSummary != "tool_calls: lookup, search" {
		t.Fatalf("unexpected tool summary: %q", toolSummary)
	}
	if toolJSON == "" {
		t.Fatal("expected tool output JSON")
	}
}
