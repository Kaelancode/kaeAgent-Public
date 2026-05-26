package multiagent

import (
	"testing"

	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/llm"
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

	tc := PrepareTransferContext(source, 1, 3)
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
