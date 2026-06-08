package agent

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func (e *runExecutor) compactConversation(ctx context.Context) error {
	if e.rt.compactor == nil {
		return nil
	}
	result, err := e.rt.compactor.Compact(ctx, compaction.Input{
		SessionID: e.rs.sessionID,
		Messages:  e.rs.conv.messagesOwned(),
		Tools:     toLLMToolDefs(e.availableToolDefs()),
	})
	if err != nil {
		return err
	}
	if result.Compacted {
		e.rs.conv.replaceMessagesOwned(result.Messages)
	}
	return nil
}

func (e *runExecutor) prepareMessagesForRequest(ctx context.Context, sessionID string, messages []llm.Message, stepTools []tools.ToolDef) ([]llm.Message, error) {
	current := cloneMessages(messages)
	if !e.shouldForcePreflightCompaction(current, stepTools) {
		if err := ensureSingleSystemMessage(current); err != nil {
			return nil, err
		}
		return current, nil
	}
	if e.rt.compactor == nil {
		return nil, fmt.Errorf("request exceeds model context limit and no compactor is configured")
	}

	result, err := e.rt.compactor.ForceCompact(ctx, compaction.Input{
		SessionID: sessionID,
		Messages:  current,
		Tools:     toLLMToolDefs(stepTools),
	})
	if err != nil {
		return nil, err
	}
	if result.Compacted {
		e.rs.conv.replaceMessagesOwned(result.Messages)
		current = cloneMessages(result.Messages)
	}
	if e.shouldForcePreflightCompaction(current, stepTools) {
		return nil, fmt.Errorf("request still exceeds model context limit after compaction")
	}
	if err := ensureSingleSystemMessage(current); err != nil {
		return nil, err
	}
	return current, nil
}

func ensureSingleSystemMessage(messages []llm.Message) error {
	count := 0
	for _, msg := range messages {
		if msg.Role == "system" {
			count++
		}
	}
	if count > 1 {
		return fmt.Errorf("runtime: expected at most one system message, got %d", count)
	}
	return nil
}

func (e *runExecutor) shouldForcePreflightCompaction(messages []llm.Message, stepTools []tools.ToolDef) bool {
	if e.rt.modelContextLimit <= 0 {
		return false
	}
	return compaction.EstimatePromptTokens(messages, toLLMToolDefs(stepTools), nil)+e.rt.outputReserve > e.rt.modelContextLimit
}
