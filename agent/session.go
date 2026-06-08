package agent

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/Kaelancode/kaeAgent-Public/store"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
)

type TrimStrategy string

const (
	TrimSlidingWindow TrimStrategy = "sliding_window"
	TrimTokenCount    TrimStrategy = "token_count"

	defaultSessionMaxHistory  = 50
	defaultSessionTokenBudget = 128000
	defaultSessionMaxTokens   = 4096
)

type SessionConfig struct {
	Model        string                  `json:"model"`
	SystemPrompt string                  `json:"system_prompt,omitempty"`
	MaxTokens    int                     `json:"max_tokens"`
	Temperature  *float32                `json:"temperature,omitempty"`
	TrimStrategy TrimStrategy            `json:"trim_strategy"`
	MaxHistory   int                     `json:"max_history"`
	TokenBudget  int                     `json:"token_budget"`
	BudgetConfig *streaming.BudgetConfig `json:"budget_config,omitempty"`
}

type Session struct {
	mu       sync.RWMutex
	ID       string
	Config   SessionConfig
	Budget   *streaming.Budget
	Metadata map[string]string
}

func NewSession(cfg SessionConfig) *Session {
	id := generateSessionID()

	var budget *streaming.Budget
	if cfg.BudgetConfig != nil {
		budget = streaming.NewBudget(*cfg.BudgetConfig)
	} else {
		budget = streaming.NewBudget(streaming.BudgetConfig{})
	}

	if cfg.TrimStrategy == "" {
		cfg.TrimStrategy = TrimSlidingWindow
	}
	if cfg.MaxHistory <= 0 {
		cfg.MaxHistory = defaultSessionMaxHistory
	}
	if cfg.TokenBudget <= 0 {
		cfg.TokenBudget = defaultSessionTokenBudget
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = defaultSessionMaxTokens
	}

	return &Session{
		ID:       id,
		Config:   cfg,
		Budget:   budget,
		Metadata: make(map[string]string),
	}
}

func (s *Session) SetMeta(key, value string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Metadata[key] = value
}

func (s *Session) GetMeta(key string) (string, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.Metadata[key]
	return v, ok
}

func (s *Session) CheckBudget() error {
	if s.Budget == nil {
		return nil
	}
	if err := s.Budget.Check(); err != nil {
		return fmt.Errorf("session %s: %w", s.ID, err)
	}
	return nil
}

func generateSessionID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return "sess_" + hex.EncodeToString(b)
}

type SessionSnapshot struct {
	ID       string                   `json:"id"`
	Config   SessionConfig            `json:"config"`
	Budget   streaming.BudgetSnapshot `json:"budget"`
	Metadata map[string]string        `json:"metadata"`
}

func (s *Session) Snapshot() SessionSnapshot {
	s.mu.RLock()
	defer s.mu.RUnlock()

	meta := make(map[string]string, len(s.Metadata))
	for k, v := range s.Metadata {
		meta[k] = v
	}

	return SessionSnapshot{
		ID:       s.ID,
		Config:   cloneSessionConfig(s.Config),
		Budget:   s.Budget.Snapshot(),
		Metadata: meta,
	}
}

func (s *Session) Restore(snap SessionSnapshot) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ID = snap.ID
	s.Config = cloneSessionConfig(snap.Config)
	s.Budget.Restore(snap.Budget)
	s.Metadata = make(map[string]string, len(snap.Metadata))
	for k, v := range snap.Metadata {
		s.Metadata[k] = v
	}
}

func (s *Session) MarshalJSON() ([]byte, error) {
	return json.Marshal(s.Snapshot())
}

func (s *Session) UnmarshalJSON(data []byte) error {
	var snap SessionSnapshot
	if err := json.Unmarshal(data, &snap); err != nil {
		return err
	}
	s.Restore(snap)
	return nil
}

func NewSessionFromSnapshot(snap SessionSnapshot) *Session {
	return &Session{
		ID:       snap.ID,
		Config:   cloneSessionConfig(snap.Config),
		Budget:   streaming.NewBudgetFromSnapshot(snap.Budget),
		Metadata: cloneStringMap(snap.Metadata),
	}
}

func (s *Session) ToStoreData(userID, agentID string) *store.SessionData {
	snap := s.Snapshot()
	configJSON, _ := json.Marshal(snap.Config)
	budgetJSON, _ := json.Marshal(snap.Budget)
	return &store.SessionData{
		ID:       snap.ID,
		UserID:   userID,
		AgentID:  agentID,
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: snap.Metadata,
	}
}

func SessionFromStoreData(data *store.SessionData) (*Session, error) {
	var config SessionConfig
	if err := json.Unmarshal(data.Config, &config); err != nil {
		return nil, fmt.Errorf("session from store: unmarshal config: %w", err)
	}
	var budgetSnap streaming.BudgetSnapshot
	if len(data.Budget) > 0 {
		if err := json.Unmarshal(data.Budget, &budgetSnap); err != nil {
			return nil, fmt.Errorf("session from store: unmarshal budget: %w", err)
		}
	}
	snap := SessionSnapshot{
		ID:       data.ID,
		Config:   config,
		Budget:   budgetSnap,
		Metadata: data.Metadata,
	}
	return NewSessionFromSnapshot(snap), nil
}
