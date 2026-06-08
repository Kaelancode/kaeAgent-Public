// tools/registry.go
package tools

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

// Registry manages tool definitions with name and tag-based lookup.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]ToolDef
}

// NewRegistry creates an empty tool registry.
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]ToolDef),
	}
}

// Register adds a tool to the registry.
func (r *Registry) Register(t ToolDef) error {
	if strings.TrimSpace(t.Name) == "" {
		return fmt.Errorf("tools: tool name is required")
	}
	if t.Schema == nil {
		return fmt.Errorf("tools: tool %q schema is required", t.Name)
	}
	if t.Handler == nil {
		return fmt.Errorf("tools: tool %q handler is required", t.Name)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[t.Name]; exists {
		return fmt.Errorf("tools: tool %q already registered", t.Name)
	}
	r.tools[t.Name] = t
	return nil
}

// Get retrieves a tool by exact name.
func (r *Registry) Get(name string) (ToolDef, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// ByTag returns all tools that have the given tag.
func (r *Registry) ByTag(tag string) []ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var result []ToolDef
	for _, t := range sortedTools(r.tools) {
		for _, tt := range t.Tags {
			if tt == tag {
				result = append(result, t)
				break
			}
		}
	}
	return result
}

// All returns every registered tool.
func (r *Registry) All() []ToolDef {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]ToolDef, 0, len(r.tools))
	for _, t := range sortedTools(r.tools) {
		result = append(result, t)
	}
	return result
}

// ToProviderFormat converts all registered tools into the format expected by the
// specified provider. Supported providers: "openai", "qwen", "anthropic", "gcp.gemini".
func (r *Registry) ToProviderFormat(providerName string) any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	switch providerName {
	case "openai", "qwen":
		return r.toOpenAIFormat()
	case "anthropic":
		return r.toClaudeFormat()
	case "gcp.gemini":
		return r.toGeminiFormat()
	default:
		return r.toOpenAIFormat()
	}
}

func (r *Registry) toOpenAIFormat() []map[string]any {
	result := make([]map[string]any, 0, len(r.tools))
	for _, t := range sortedTools(r.tools) {
		params := schemaToMap(t.Schema)
		result = append(result, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  params,
			},
		})
	}
	return result
}

func (r *Registry) toClaudeFormat() []map[string]any {
	result := make([]map[string]any, 0, len(r.tools))
	for _, t := range sortedTools(r.tools) {
		params := schemaToMap(t.Schema)
		result = append(result, map[string]any{
			"name":         t.Name,
			"description":  t.Description,
			"input_schema": params,
		})
	}
	return result
}

func (r *Registry) toGeminiFormat() map[string]any {
	decls := make([]map[string]any, 0, len(r.tools))
	for _, t := range sortedTools(r.tools) {
		params := schemaToMap(t.Schema)
		decls = append(decls, map[string]any{
			"name":        t.Name,
			"description": t.Description,
			"parameters":  params,
		})
	}
	return map[string]any{
		"functionDeclarations": decls,
	}
}

func schemaToMap(s *schema.Schema) map[string]any {
	if s == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return s.ToMap()
}

// Names returns a formatted list of tool names for debugging.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.tools))
	for name := range r.tools {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Count returns the number of registered tools.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.tools)
}

// Remove unregisters a tool by name. Returns an error if not found.
func (r *Registry) Remove(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.tools[name]; !ok {
		return fmt.Errorf("tools: tool %q not found in registry", name)
	}
	delete(r.tools, name)
	return nil
}

func sortedTools(input map[string]ToolDef) []ToolDef {
	names := make([]string, 0, len(input))
	for name := range input {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]ToolDef, 0, len(names))
	for _, name := range names {
		out = append(out, input[name])
	}
	return out
}
