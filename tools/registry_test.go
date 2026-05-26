package tools

import (
	"context"
	"testing"

	"github.com/yourorg/agent-sdk/schema"
)

func TestRegistry_RegisterAndGet(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{
		Name:        "test_tool",
		Description: "a test tool",
		Tags:        []string{"test", "utility"},
		Handler:     func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil },
	})

	tool, ok := r.Get("test_tool")
	if !ok {
		t.Fatal("expected to find test_tool")
	}
	if tool.Name != "test_tool" {
		t.Errorf("expected test_tool, got %s", tool.Name)
	}
}

func TestRegistry_ByTag(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{Name: "a", Tags: []string{"web", "read-only"}})
	r.Register(ToolDef{Name: "b", Tags: []string{"web"}})
	r.Register(ToolDef{Name: "c", Tags: []string{"finance"}})

	web := r.ByTag("web")
	if len(web) != 2 {
		t.Errorf("expected 2 web tools, got %d", len(web))
	}

	finance := r.ByTag("finance")
	if len(finance) != 1 {
		t.Errorf("expected 1 finance tool, got %d", len(finance))
	}

	none := r.ByTag("nonexistent")
	if len(none) != 0 {
		t.Errorf("expected 0 tools for nonexistent tag, got %d", len(none))
	}
}

func TestRegistry_ToProviderFormat_OpenAI(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{
		Name:        "search",
		Description: "Search the web",
		Schema:      &schema.Schema{Type: "object", Properties: map[string]*schema.Schema{"query": {Type: "string"}}},
	})

	result := r.ToProviderFormat("openai")
	tools, ok := result.([]map[string]any)
	if !ok {
		t.Fatalf("expected []map[string]any, got %T", result)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0]["type"] != "function" {
		t.Errorf("expected type=function, got %v", tools[0]["type"])
	}
	fn := tools[0]["function"].(map[string]any)
	if fn["name"] != "search" {
		t.Errorf("expected name=search, got %v", fn["name"])
	}
}

func TestRegistry_ToProviderFormat_Anthropic(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{Name: "search", Description: "Search"})

	result := r.ToProviderFormat("anthropic")
	tools, ok := result.([]map[string]any)
	if !ok {
		t.Fatalf("expected []map[string]any, got %T", result)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if _, has := tools[0]["input_schema"]; !has {
		t.Error("expected input_schema key for Claude format")
	}
}

func TestRegistry_ToProviderFormat_GcpGemini(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{Name: "search", Description: "Search"})

	result := r.ToProviderFormat("gcp.gemini")
	m, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", result)
	}
	decls, ok := m["functionDeclarations"]
	if !ok {
		t.Fatal("expected functionDeclarations key")
	}
	arr := decls.([]map[string]any)
	if len(arr) != 1 {
		t.Errorf("expected 1 declaration, got %d", len(arr))
	}
}

func TestRegistry_Remove(t *testing.T) {
	r := NewRegistry()
	r.Register(ToolDef{Name: "to_remove"})

	err := r.Remove("to_remove")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Count() != 0 {
		t.Errorf("expected 0 tools after remove, got %d", r.Count())
	}

	err = r.Remove("nonexistent")
	if err == nil {
		t.Error("expected error for removing nonexistent tool")
	}
}
