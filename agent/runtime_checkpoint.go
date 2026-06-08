package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/store"
)

func (e *runExecutor) checkpoint(ctx context.Context) error {
	if e.rt.conversationStore == nil {
		return nil
	}
	if err := e.rt.conversationStore.Save(ctx, e.rs.sessionID, e.rs.conv.messagesOwned()); err != nil {
		e.rt.logger.Error().Err(err).Str("session_id", e.rs.sessionID).Msg("conversation checkpoint failed")
		return fmt.Errorf("runtime: checkpoint conversation: %w", err)
	}
	return nil
}

func (e *runExecutor) saveSessionData(ctx context.Context) error {
	if e.rt.sessionStore == nil {
		return nil
	}
	data := e.sessionStoreData()
	if err := e.rt.sessionStore.SaveSession(ctx, data); err != nil {
		e.rt.logger.Error().Err(err).Str("session_id", e.rs.sessionID).Msg("session save failed")
		return fmt.Errorf("runtime: save session: %w", err)
	}
	return nil
}

func (e *runExecutor) sessionStoreData() *store.SessionData {
	configJSON, _ := json.Marshal(cloneSessionConfig(e.rs.config))
	budgetJSON, _ := json.Marshal(e.rs.budget.Snapshot())
	return &store.SessionData{
		ID:       e.rs.sessionID,
		UserID:   e.rs.userID,
		AgentID:  e.rs.agentID,
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: cloneStringMap(e.rs.metadata),
	}
}
