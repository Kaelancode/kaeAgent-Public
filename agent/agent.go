package agent

import (
	"sync"

	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

type AgentConfig struct {
	Name         string
	Model        string
	SystemPrompt string
	MaxTokens    int
	Temperature  *float32
	TrimStrategy TrimStrategy
	MaxHistory   int
	TokenBudget  int
	BudgetConfig *streaming.BudgetConfig
	Subagents    []string
	MaxSteps     int
}

type Agent struct {
	mu        sync.RWMutex
	config    AgentConfig
	tools     *tools.Registry
	subagents []string
}

func NewAgent(cfg AgentConfig) *Agent {
	return &Agent{
		config:    cloneAgentConfig(cfg),
		tools:     tools.NewRegistry(),
		subagents: append([]string(nil), cfg.Subagents...),
	}
}

func (a *Agent) Snapshot() AgentConfig {
	a.mu.RLock()
	defer a.mu.RUnlock()
	cfg := cloneAgentConfig(a.config)
	cfg.Subagents = append([]string(nil), a.subagents...)
	return cfg
}

func (a *Agent) SessionConfig() SessionConfig {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return SessionConfig{
		Model:        a.config.Model,
		SystemPrompt: a.config.SystemPrompt,
		MaxTokens:    a.config.MaxTokens,
		Temperature:  cloneFloat32Ptr(a.config.Temperature),
		TrimStrategy: a.config.TrimStrategy,
		MaxHistory:   a.config.MaxHistory,
		TokenBudget:  a.config.TokenBudget,
		BudgetConfig: cloneBudgetConfig(a.config.BudgetConfig),
	}
}

func (a *Agent) Name() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config.Name
}

func (a *Agent) MaxSteps() int {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config.MaxSteps
}

func (a *Agent) RegisterTool(t tools.ToolDef) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.tools.Register(t)
}

func (a *Agent) ToolRegistry() *tools.Registry {
	a.mu.RLock()
	defer a.mu.RUnlock()
	registry := tools.NewRegistry()
	for _, t := range a.tools.All() {
		registry.Register(t)
	}
	return registry
}

func (a *Agent) Tools() []tools.ToolDef {
	return a.ToolRegistry().All()
}

func (a *Agent) AddSubagent(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.subagents = append(a.subagents, name)
}

func (a *Agent) Subagents() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return append([]string(nil), a.subagents...)
}

func (a *Agent) HasSubagent(name string) bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	for _, subagent := range a.subagents {
		if subagent == name {
			return true
		}
	}
	return false
}

func cloneAgentConfig(cfg AgentConfig) AgentConfig {
	return AgentConfig{
		Name:         cfg.Name,
		Model:        cfg.Model,
		SystemPrompt: cfg.SystemPrompt,
		MaxTokens:    cfg.MaxTokens,
		Temperature:  cloneFloat32Ptr(cfg.Temperature),
		TrimStrategy: cfg.TrimStrategy,
		MaxHistory:   cfg.MaxHistory,
		TokenBudget:  cfg.TokenBudget,
		BudgetConfig: cloneBudgetConfig(cfg.BudgetConfig),
		Subagents:    append([]string(nil), cfg.Subagents...),
		MaxSteps:     cfg.MaxSteps,
	}
}

func cloneFloat32Ptr(v *float32) *float32 {
	if v == nil {
		return nil
	}
	out := *v
	return &out
}

func cloneBudgetConfig(cfg *streaming.BudgetConfig) *streaming.BudgetConfig {
	if cfg == nil {
		return nil
	}
	clone := *cfg
	return &clone
}
