package inmem

import (
	"context"
	"sync"

	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/store"
)

type ConversationStore struct {
	mu    sync.RWMutex
	store map[string][]llm.Message
}

var _ store.ConversationStore = (*ConversationStore)(nil)

func NewConversationStore() *ConversationStore {
	return &ConversationStore{
		store: make(map[string][]llm.Message),
	}
}

func (s *ConversationStore) Save(_ context.Context, convID string, messages []llm.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.store[convID] = cloneMessages(messages)
	return nil
}

func (s *ConversationStore) Load(_ context.Context, convID string) ([]llm.Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	msgs, ok := s.store[convID]
	if !ok {
		return nil, nil
	}
	return cloneMessages(msgs), nil
}

func (s *ConversationStore) Append(_ context.Context, convID string, messages []llm.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	existing := s.store[convID]
	appended := cloneMessages(messages)
	s.store[convID] = append(existing, appended...)
	return nil
}

func (s *ConversationStore) Delete(_ context.Context, convID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.store, convID)
	return nil
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
