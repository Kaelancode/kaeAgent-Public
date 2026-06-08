package multiagent

import (
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestPrepareAndApplyTransferContext(t *testing.T) {
	source := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "root",
			Model:        "source-model",
			SystemPrompt: "source prompt",
		}),
	})
	source.AppendConversationMessage(llm.Message{Role: "user", Content: "one"})
	source.AppendConversationMessage(llm.Message{Role: "assistant", Content: "two"})

	tc, err := PrepareTransferContext(source, 1, 3)
	if err != nil {
		t.Fatalf("PrepareTransferContext failed: %v", err)
	}
	if tc.Metadata[TransferSourceModelKey] != "source-model" {
		t.Fatalf("expected source model metadata, got %v", tc.Metadata)
	}

	target := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "child",
			Model:        "child-model",
			SystemPrompt: "child prompt",
		}),
	})
	ApplyTransferContext(target, tc)

	msgs := target.ConversationMessages()
	if len(msgs) != 3 {
		t.Fatalf("expected system prompt plus 2 transferred messages, got %d", len(msgs))
	}
	if msgs[0].Role != "system" || msgs[0].Content != "child prompt" {
		t.Fatalf("expected target system prompt preserved, got %#v", msgs[0])
	}
	if msgs[1].Content != "one" || msgs[2].Content != "two" {
		t.Fatalf("expected transferred messages appended, got %#v", msgs)
	}

	snap := target.SessionSnapshot()
	if snap.Metadata[TransferSourceSessionKey] == "" {
		t.Fatalf("expected source session metadata to be applied")
	}
}

func TestPrepareTransferContextValidatesBounds(t *testing.T) {
	source := agent.NewRuntime(agent.RuntimeConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "root",
			Model:        "source-model",
			SystemPrompt: "source prompt",
		}),
	})
	source.AppendConversationMessage(llm.Message{Role: "user", Content: "one"})

	tests := []struct {
		name       string
		start, end int
		want       string
	}{
		{name: "negative start", start: -1, end: 1, want: "non-negative"},
		{name: "negative end", start: 0, end: -1, want: "non-negative"},
		{name: "start exceeds end", start: 2, end: 1, want: "start exceeds end"},
		{name: "end exceeds count", start: 0, end: 3, want: "end exceeds message count"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := PrepareTransferContext(source, tt.start, tt.end)
			if err == nil {
				t.Fatal("expected bounds error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected error containing %q, got %v", tt.want, err)
			}
		})
	}
}

func TestPrepareTransferContextRejectsNilRuntime(t *testing.T) {
	_, err := PrepareTransferContext(nil, 0, 0)
	if err == nil {
		t.Fatal("expected nil runtime error")
	}
	if !strings.Contains(err.Error(), "runtime is nil") {
		t.Fatalf("expected nil runtime error, got %v", err)
	}
}
