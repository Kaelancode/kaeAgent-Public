package summary

import (
	"context"
	"strings"
	"testing"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

func TestStrategy_SummarizesOlderTurnsAndKeepsRecentTurns(t *testing.T) {
	strategy := New(2, func(_ context.Context, turns [][]llm.Message) (string, error) {
		return "Older summary", nil
	})

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "a"},
			{Role: "assistant", Content: "one"},
			{Role: "user", Content: "b"},
			{Role: "assistant", Content: "two"},
			{Role: "user", Content: "c"},
			{Role: "assistant", Content: "three"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !out.Compacted {
		t.Fatalf("expected compacted output")
	}
	if len(out.Messages) != 6 {
		t.Fatalf("expected system + summary + 2 turns, got %d messages", len(out.Messages))
	}
	if out.Messages[1].Role != "system" {
		t.Fatalf("expected summary message to use system role, got %q", out.Messages[1].Role)
	}
	if !strings.Contains(out.Messages[1].Content, "Older summary") {
		t.Fatalf("expected summary content, got %q", out.Messages[1].Content)
	}
	if out.Messages[2].Content != "b" || out.Messages[3].Content != "two" || out.Messages[4].Content != "c" || out.Messages[5].Content != "three" {
		t.Fatalf("unexpected retained messages: %+v", out.Messages)
	}
}

func TestStrategy_DefaultSummarizerIncludesToolTurnContent(t *testing.T) {
	strategy := New(1, nil)

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "search"},
			{Role: "assistant", Content: ""},
			{Role: "tool", Name: "lookup", Content: "result payload"},
			{Role: "assistant", Content: "done"},
			{Role: "user", Content: "latest"},
			{Role: "assistant", Content: "current"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if len(out.Messages) != 4 {
		t.Fatalf("expected system + summary + latest turn, got %d messages", len(out.Messages))
	}
	summaryMsg := out.Messages[1]
	if summaryMsg.Role != "system" {
		t.Fatalf("expected summary role system, got %q", summaryMsg.Role)
	}
	if !strings.Contains(summaryMsg.Content, "Conversation summary:") {
		t.Fatalf("expected default summary prefix, got %q", summaryMsg.Content)
	}
	if !strings.Contains(summaryMsg.Content, "Tool(lookup): result payload") {
		t.Fatalf("expected tool content in summary, got %q", summaryMsg.Content)
	}
	if out.Messages[2].Content != "latest" || out.Messages[3].Content != "current" {
		t.Fatalf("expected latest turn to remain intact, got %+v", out.Messages)
	}
}

func TestStrategy_NoCompactionWhenTurnCountFits(t *testing.T) {
	strategy := New(2, func(_ context.Context, turns [][]llm.Message) (string, error) {
		return "unused", nil
	})

	in := []llm.Message{
		{Role: "system", Content: "sys"},
		{Role: "user", Content: "a"},
		{Role: "assistant", Content: "one"},
		{Role: "user", Content: "b"},
		{Role: "assistant", Content: "two"},
	}

	out, err := strategy.Compact(context.Background(), compaction.Input{Messages: in})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if out.Compacted {
		t.Fatalf("expected no compaction")
	}
	if len(out.Messages) != len(in) {
		t.Fatalf("expected original messages to be preserved")
	}
}

func TestStrategy_ReplacesPreviousSummaryInsteadOfAccumulating(t *testing.T) {
	strategy := New(2, func(_ context.Context, turns [][]llm.Message) (string, error) {
		if len(turns) != 2 {
			t.Fatalf("expected prior summary plus one newly summarized turn, got %d", len(turns))
		}
		if len(turns[0]) != 1 || turns[0][0].Role != "system" || !strings.Contains(turns[0][0].Content, "Turn 1 summary") {
			t.Fatalf("expected prior summary to be passed into summarizer, got %+v", turns[0])
		}
		return "Turn 1 summary\nTurn 2 summary", nil
	})

	out, err := strategy.Compact(context.Background(), compaction.Input{
		Messages: []llm.Message{
			{Role: "system", Content: "sys"},
			{Role: "system", Content: "Conversation summary:\nTurn 1 summary"},
			{Role: "user", Content: "b"},
			{Role: "assistant", Content: "two"},
			{Role: "user", Content: "c"},
			{Role: "assistant", Content: "three"},
			{Role: "user", Content: "d"},
			{Role: "assistant", Content: "four"},
		},
	})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !out.Compacted {
		t.Fatalf("expected compacted output")
	}
	if len(out.Messages) != 6 {
		t.Fatalf("expected system + summary + 2 turns, got %d messages", len(out.Messages))
	}

	summaryCount := 0
	for _, msg := range out.Messages {
		if msg.Role == "system" && strings.HasPrefix(msg.Content, "Conversation summary:") {
			summaryCount++
			if strings.Count(msg.Content, "Turn 1 summary") != 1 || strings.Count(msg.Content, "Turn 2 summary") != 1 {
				t.Fatalf("expected merged summary content, got %q", msg.Content)
			}
		}
	}
	if summaryCount != 1 {
		t.Fatalf("expected exactly one retained summary message, got %d", summaryCount)
	}
}
