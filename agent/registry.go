package agent

import "sync"

type Registry struct {
	mu     sync.RWMutex
	agents map[string]*Agent
}

func NewRegistry() *Registry {
	return &Registry{
		agents: make(map[string]*Agent),
	}
}

func (r *Registry) Register(agentDef *Agent) {
	if agentDef == nil {
		return
	}

	name := agentDef.Name()
	if name == "" {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents[name] = agentDef
}

func (r *Registry) Get(name string) (*Agent, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	agentDef, ok := r.agents[name]
	return agentDef, ok
}

func (r *Registry) List() []*Agent {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]*Agent, 0, len(r.agents))
	for _, agentDef := range r.agents {
		out = append(out, agentDef)
	}
	return out
}

func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]string, 0, len(r.agents))
	for name := range r.agents {
		out = append(out, name)
	}
	return out
}

func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.agents)
}

var _ SubagentResolver = (*Registry)(nil)
