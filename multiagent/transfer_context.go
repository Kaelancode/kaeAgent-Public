// multiagent/transfer_context.go
package multiagent

import (
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
)

const (
	TransferSourceModelKey   = "source_model"
	TransferSourceSessionKey = "source_session"
)

// TransferContext packages explicitly selected context and metadata for transfer
// between agents.
type TransferContext struct {
	SourceSessionID string
	Messages        []llm.Message
	Metadata        map[string]string
}

// PrepareTransferContext extracts the relevant portion of a conversation for transfer.
// If start/end are both 0 the entire history is included.
func PrepareTransferContext(rt *agent.Runtime, start, end int) (TransferContext, error) {
	if rt == nil {
		return TransferContext{}, fmt.Errorf("multiagent: prepare transfer context: runtime is nil")
	}
	sess := rt.SessionSnapshot()
	allMessages := rt.ConversationMessages()

	var msgs []llm.Message
	if start == 0 && end == 0 {
		msgs = allMessages
	} else {
		if start < 0 || end < 0 {
			return TransferContext{}, fmt.Errorf("multiagent: prepare transfer context: invalid range [%d:%d]: indices must be non-negative", start, end)
		}
		if start > end {
			return TransferContext{}, fmt.Errorf("multiagent: prepare transfer context: invalid range [%d:%d]: start exceeds end", start, end)
		}
		if end > len(allMessages) {
			return TransferContext{}, fmt.Errorf("multiagent: prepare transfer context: invalid range [%d:%d]: end exceeds message count %d", start, end, len(allMessages))
		}
		msgs = allMessages[start:end]
	}

	meta := make(map[string]string)
	meta[TransferSourceModelKey] = sess.Config.Model
	meta[TransferSourceSessionKey] = sess.ID

	return TransferContext{
		SourceSessionID: sess.ID,
		Messages:        msgs,
		Metadata:        meta,
	}, nil
}

// ApplyTransferContext injects the selected transfer messages into a target runtime's
// conversation without modifying the target agent's system prompt. Metadata is
// merged into the target session.
func ApplyTransferContext(rt *agent.Runtime, tc TransferContext) {
	rt.AppendConversationMessages(tc.Messages)
	for k, v := range tc.Metadata {
		rt.SetSessionMetadata(k, v)
	}
}
