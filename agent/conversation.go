package agent

import (
	"encoding/json"
	"sync"

	"github.com/yourorg/agent-sdk/llm"
)

const charsPerToken = 4

type Conversation struct {
	mu       sync.RWMutex
	messages []llm.Message
	strategy TrimStrategy
	maxMsgs  int
	maxChars int
}

func NewConversation(strategy TrimStrategy, maxMessages int, tokenBudget int) *Conversation {
	return &Conversation{
		strategy: strategy,
		maxMsgs:  maxMessages,
		maxChars: tokenBudget * charsPerToken,
	}
}

func (c *Conversation) Append(msg llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append(c.messages, cloneMessage(msg))
}

func (c *Conversation) AppendAll(msgs []llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append(c.messages, cloneMessages(msgs)...)
}

func (c *Conversation) Messages() []llm.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return cloneMessages(c.messages)
}

func (c *Conversation) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.messages)
}

func (c *Conversation) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = nil
}

func (c *Conversation) ReplaceMessages(messages []llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = cloneMessages(messages)
}

func (c *Conversation) appendOwned(msg llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append(c.messages, msg)
}

func (c *Conversation) appendAllOwned(msgs []llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append(c.messages, msgs...)
}

func (c *Conversation) messagesOwned() []llm.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return cloneMessages(c.messages)
}

func (c *Conversation) replaceMessagesOwned(messages []llm.Message) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = cloneMessages(messages)
}

func (c *Conversation) SetSystem(content string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	sysMsg := llm.Message{Role: "system", Content: content}
	if len(c.messages) > 0 && c.messages[0].Role == "system" {
		c.messages[0] = sysMsg
	} else {
		c.messages = append([]llm.Message{sysMsg}, c.messages...)
	}
}

func (c *Conversation) Slice(start, end int) []llm.Message {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if start < 0 {
		start = 0
	}
	if end > len(c.messages) {
		end = len(c.messages)
	}
	if start >= end {
		return nil
	}

	out := make([]llm.Message, end-start)
	copy(out, c.messages[start:end])
	return cloneMessages(out)
}

type ConversationState struct {
	Messages []llm.Message `json:"messages"`
	Strategy TrimStrategy  `json:"strategy"`
	MaxMsgs  int           `json:"max_msgs"`
	MaxChars int           `json:"max_chars"`
}

func (c *Conversation) Snapshot() ConversationState {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return ConversationState{
		Messages: cloneMessages(c.messages),
		Strategy: c.strategy,
		MaxMsgs:  c.maxMsgs,
		MaxChars: c.maxChars,
	}
}

func (c *Conversation) Restore(s ConversationState) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.messages = cloneMessages(s.Messages)
	c.strategy = s.Strategy
	c.maxMsgs = s.MaxMsgs
	c.maxChars = s.MaxChars
}

func (c *Conversation) MarshalJSON() ([]byte, error) {
	return json.Marshal(c.Snapshot())
}

func (c *Conversation) UnmarshalJSON(data []byte) error {
	var legacyState struct {
		ID       string        `json:"id"`
		Messages []llm.Message `json:"messages"`
		Strategy TrimStrategy  `json:"strategy"`
		MaxMsgs  int           `json:"max_msgs"`
		MaxChars int           `json:"max_chars"`
	}
	if err := json.Unmarshal(data, &legacyState); err != nil {
		return err
	}
	c.Restore(ConversationState{
		Messages: legacyState.Messages,
		Strategy: legacyState.Strategy,
		MaxMsgs:  legacyState.MaxMsgs,
		MaxChars: legacyState.MaxChars,
	})
	return nil
}

func NewConversationFromState(state ConversationState) *Conversation {
	return &Conversation{
		messages: cloneMessages(state.Messages),
		strategy: state.Strategy,
		maxMsgs:  state.MaxMsgs,
		maxChars: state.MaxChars,
	}
}
