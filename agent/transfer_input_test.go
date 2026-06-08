package agent

import (
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestRecentWindowTransferInputPreservesSystemAndKeepsRecentNonSystem(t *testing.T) {
	filter := RecentWindowTransferInput(3)

	out, err := filter(TransferInputData{
		Messages: []llm.Message{
			{Role: "system", Content: "system-1"},
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: "a1"},
			{Role: "user", Content: "u2"},
			{Role: "assistant", Content: "a2"},
			{Role: "system", Content: "system-2"},
			{Role: "user", Content: "u3"},
		},
	})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	if len(out.Messages) != 5 {
		t.Fatalf("expected 5 messages after filtering, got %d", len(out.Messages))
	}

	// Expected order is preserved from the original transcript for the retained window.
	want := []struct {
		role    string
		content string
	}{
		{"system", "system-1"},
		{"system", "system-2"},
		{"user", "u2"},
		{"assistant", "a2"},
		{"user", "u3"},
	}

	for i, msg := range out.Messages {
		if msg.Role != want[i].role || msg.Content != want[i].content {
			t.Fatalf("message %d: expected %s %q, got %s %q", i, want[i].role, want[i].content, msg.Role, msg.Content)
		}
	}
}

func TestRecentWindowTransferInputZeroKeepsOnlySystem(t *testing.T) {
	filter := RecentWindowTransferInput(0)

	out, err := filter(TransferInputData{
		Messages: []llm.Message{
			{Role: "system", Content: "system-1"},
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: "a1"},
			{Role: "system", Content: "system-2"},
		},
	})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	if len(out.Messages) != 2 {
		t.Fatalf("expected 2 system messages, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[0].Content != "system-1" {
		t.Fatalf("unexpected first system message: %+v", out.Messages[0])
	}
	if out.Messages[1].Role != "system" || out.Messages[1].Content != "system-2" {
		t.Fatalf("unexpected second system message: %+v", out.Messages[1])
	}
}

func TestNestTransferHistoryPreservesSystemAndWrapsTranscript(t *testing.T) {
	filter := NestTransferHistory()

	out, err := filter(TransferInputData{
		Messages: []llm.Message{
			{Role: "system", Content: "system-1"},
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: "a1"},
			{Role: "tool", Content: "tool output"},
		},
	})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	if len(out.Messages) != 2 {
		t.Fatalf("expected system message plus nested history message, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[0].Content != "system-1" {
		t.Fatalf("unexpected preserved system message: %+v", out.Messages[0])
	}
	if out.Messages[1].Role != "assistant" {
		t.Fatalf("expected nested history message to be assistant role, got %+v", out.Messages[1])
	}
	if !strings.Contains(out.Messages[1].Content, DefaultTransferHistoryStart) {
		t.Fatalf("expected nested history start marker, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, DefaultTransferHistoryEnd) {
		t.Fatalf("expected nested history end marker, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, "user: u1") {
		t.Fatalf("expected user message in nested history, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, "assistant: a1") {
		t.Fatalf("expected assistant message in nested history, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, "tool: tool output") {
		t.Fatalf("expected tool message in nested history, got %q", out.Messages[1].Content)
	}
}

func TestNestTransferHistoryOnlySystemLeavesSystemMessages(t *testing.T) {
	filter := NestTransferHistory()

	out, err := filter(TransferInputData{
		Messages: []llm.Message{
			{Role: "system", Content: "system-1"},
			{Role: "system", Content: "system-2"},
		},
	})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	if len(out.Messages) != 2 {
		t.Fatalf("expected only original system messages, got %d", len(out.Messages))
	}
	if out.Messages[0].Content != "system-1" || out.Messages[1].Content != "system-2" {
		t.Fatalf("unexpected system-only output: %+v", out.Messages)
	}
}

func TestComposeTransferInputChainsFiltersLeftToRight(t *testing.T) {
	filter := ComposeTransferInput(
		RemoveToolTransferInput(),
		RecentWindowTransferInput(2),
		NestTransferHistory(),
	)

	out, err := filter(TransferInputData{
		Messages: []llm.Message{
			{Role: "system", Content: "system-1"},
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: "a1"},
			{
				Role: "assistant",
				ToolCalls: []llm.ToolCall{{
					ID:    "call_1",
					Name:  "lookup",
					Input: map[string]any{"id": "123"},
				}},
			},
			{Role: "tool", Content: "tool output", ToolCallID: "call_1"},
			{Role: "user", Content: "u2"},
			{Role: "assistant", Content: "a2"},
		},
	})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}

	if len(out.Messages) != 2 {
		t.Fatalf("expected system plus nested-history message, got %d", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[0].Content != "system-1" {
		t.Fatalf("unexpected preserved system message: %+v", out.Messages[0])
	}
	if out.Messages[1].Role != "assistant" {
		t.Fatalf("expected nested history message, got %+v", out.Messages[1])
	}
	if strings.Contains(out.Messages[1].Content, "tool output") {
		t.Fatalf("did not expect tool chatter after composition, got %q", out.Messages[1].Content)
	}
	if strings.Contains(out.Messages[1].Content, "u1") || strings.Contains(out.Messages[1].Content, "a1") {
		t.Fatalf("did not expect older non-system messages after recent-window trim, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, "user: u2") {
		t.Fatalf("expected retained recent user message, got %q", out.Messages[1].Content)
	}
	if !strings.Contains(out.Messages[1].Content, "assistant: a2") {
		t.Fatalf("expected retained recent assistant message, got %q", out.Messages[1].Content)
	}
}
