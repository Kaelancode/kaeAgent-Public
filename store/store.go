package store

import (
	"context"
	"encoding/json"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

type ConversationStore interface {
	Save(ctx context.Context, convID string, messages []llm.Message) error
	Load(ctx context.Context, convID string) ([]llm.Message, error)
	Append(ctx context.Context, convID string, messages []llm.Message) error
	Delete(ctx context.Context, convID string) error
}

type SessionData struct {
	ID       string            `json:"id"`
	UserID   string            `json:"user_id"`
	AgentID  string            `json:"agent_id"`
	Config   json.RawMessage   `json:"config"`
	Budget   json.RawMessage   `json:"budget"`
	Metadata map[string]string `json:"metadata"`
}

type SessionEntry struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	AgentID   string    `json:"agent_id"`
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type SessionStore interface {
	SaveSession(ctx context.Context, data *SessionData) error
	LoadSession(ctx context.Context, sessionID string) (*SessionData, error)
	DeleteSession(ctx context.Context, sessionID string) error
	ListSessions(ctx context.Context, userID, agentID string) ([]SessionEntry, error)
}
