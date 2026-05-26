// multiagent/router.go
package multiagent

import (
	"fmt"
	"sync"

	"github.com/yourorg/agent-sdk/agent"
)

// AgentConfig describes a registered agent capability for the multiagent
// workflow/compatibility layer. For model-driven consult/transfer in normal
// Runtime.Run usage, prefer agent.Agent plus agent.Registry directly.
type AgentConfig struct {
	Agent        *agent.Agent
	Name         string
	Description  string
	SystemPrompt string
	Model        string
	Tags         []string
	MaxSteps     int
	Subagents    []string
}

func (c AgentConfig) Definition() *agent.Agent {
	return c.Agent
}

func (c AgentConfig) materializeAgent() *agent.Agent {
	if c.Agent != nil {
		return c.Agent
	}
	return agent.NewAgent(agent.AgentConfig{
		Name:         c.Name,
		Model:        c.Model,
		SystemPrompt: c.SystemPrompt,
		MaxSteps:     c.MaxSteps,
		Subagents:    append([]string(nil), c.Subagents...),
	})
}

// Router is a discovery/lookup helper for registered agents.
// It is not the owner of orchestration policy: the calling agent is expected to
// choose which subagent to consult or transfer to.
type Router struct {
	mu     sync.RWMutex
	agents map[string]AgentConfig
	tagIdx map[string][]string // tag → agent names
}

// NewRouter creates an empty router.
func NewRouter() *Router {
	return &Router{
		agents: make(map[string]AgentConfig),
		tagIdx: make(map[string][]string),
	}
}

// Register adds an agent configuration to the router.
//
// Router registration is for workflow helpers, tag lookup, and compatibility
// orchestrator calls. It does not register model-driven consult/transfer tools
// on a core runtime; those come from the active agent's declared subagents and
// RuntimeConfig.SubagentResolver.
func (r *Router) Register(cfg AgentConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()

	cfg.Agent = cfg.materializeAgent()
	if prev, exists := r.agents[cfg.Name]; exists {
		for _, tag := range prev.Tags {
			r.tagIdx[tag] = removeAgentName(r.tagIdx[tag], cfg.Name)
			if len(r.tagIdx[tag]) == 0 {
				delete(r.tagIdx, tag)
			}
		}
	}

	r.agents[cfg.Name] = cfg
	for _, tag := range cfg.Tags {
		if !containsAgentName(r.tagIdx[tag], cfg.Name) {
			r.tagIdx[tag] = append(r.tagIdx[tag], cfg.Name)
		}
	}
}

// Route is a convenience helper that returns the first registered agent
// matching the tag. Prefer Get(name) for explicit caller-chosen subagent
// invocation, or RouteAll(tag) when the caller wants to inspect candidates.
func (r *Router) Route(tag string) (AgentConfig, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names, ok := r.tagIdx[tag]
	if !ok || len(names) == 0 {
		return AgentConfig{}, fmt.Errorf("router: no agent registered for tag %q", tag)
	}

	cfg, ok := r.agents[names[0]]
	if !ok {
		return AgentConfig{}, fmt.Errorf("router: agent %q not found", names[0])
	}
	return cfg, nil
}

// RouteAll returns all registered candidate agents matching a tag so the
// caller can choose explicitly.
func (r *Router) RouteAll(tag string) []AgentConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := r.tagIdx[tag]
	configs := make([]AgentConfig, 0, len(names))
	for _, name := range names {
		if cfg, ok := r.agents[name]; ok {
			configs = append(configs, cfg)
		}
	}
	return configs
}

// Get retrieves an agent config by name.
func (r *Router) Get(name string) (AgentConfig, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cfg, ok := r.agents[name]
	return cfg, ok
}

// List returns all registered agent names.
func (r *Router) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.agents))
	for name := range r.agents {
		names = append(names, name)
	}
	return names
}

func containsAgentName(names []string, target string) bool {
	for _, name := range names {
		if name == target {
			return true
		}
	}
	return false
}

func removeAgentName(names []string, target string) []string {
	if len(names) == 0 {
		return nil
	}
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name != target {
			out = append(out, name)
		}
	}
	return out
}
