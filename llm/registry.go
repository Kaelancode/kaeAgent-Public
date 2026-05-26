// llm/registry.go
package llm

import (
	"fmt"
	"strings"
	"sync"
)

// ProviderRegistry maps model-name prefixes to Provider implementations.
type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[string]Provider // prefix → Provider
}

// NewProviderRegistry creates an empty registry.
func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers: make(map[string]Provider),
	}
}

// Register associates a prefix (e.g. "gpt-", "claude-", "gemini-") with a Provider.
func (r *ProviderRegistry) Register(prefix string, p Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[prefix] = p
}

// Resolve finds the Provider whose registered prefix matches the given model name.
// It returns the longest matching prefix to handle overlapping registrations.
func (r *ProviderRegistry) Resolve(model string) (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var best Provider
	bestLen := 0

	for prefix, p := range r.providers {
		if strings.HasPrefix(model, prefix) && len(prefix) > bestLen {
			best = p
			bestLen = len(prefix)
		}
	}

	if best == nil {
		return nil, fmt.Errorf("registry: no provider registered for model %q", model)
	}
	return best, nil
}

// List returns all registered prefix→provider-name pairs.
func (r *ProviderRegistry) List() map[string]string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make(map[string]string, len(r.providers))
	for prefix, p := range r.providers {
		out[prefix] = p.Name()
	}
	return out
}
