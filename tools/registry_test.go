package tools

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

func validTestTool(name string) ToolDef {
	return ToolDef{
		Name:    name,
		Schema:  &schema.Schema{Type: "object"},
		Handler: func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil },
	}
}

func TestRegistry_RegisterAndGet(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(ToolDef{
		Name:        "test_tool",
		Description: "a test tool",
		Tags:        []string{"test", "utility"},
		Schema:      &schema.Schema{Type: "object"},
		Handler:     func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil },
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

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
	a := validTestTool("a")
	a.Tags = []string{"web", "read-only"}
	b := validTestTool("b")
	b.Tags = []string{"web"}
	c := validTestTool("c")
	c.Tags = []string{"finance"}
	if err := r.Register(a); err != nil {
		t.Fatalf("Register a: %v", err)
	}
	if err := r.Register(b); err != nil {
		t.Fatalf("Register b: %v", err)
	}
	if err := r.Register(c); err != nil {
		t.Fatalf("Register c: %v", err)
	}

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
	if err := r.Register(ToolDef{
		Name:        "search",
		Description: "Search the web",
		Schema:      &schema.Schema{Type: "object", Properties: map[string]*schema.Schema{"query": {Type: "string"}}},
		Handler:     func(_ context.Context, _ map[string]any) (any, error) { return "ok", nil },
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

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
	tool := validTestTool("search")
	tool.Description = "Search"
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register: %v", err)
	}

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
	tool := validTestTool("search")
	tool.Description = "Search"
	if err := r.Register(tool); err != nil {
		t.Fatalf("Register: %v", err)
	}

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
	if err := r.Register(validTestTool("to_remove")); err != nil {
		t.Fatalf("Register: %v", err)
	}

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

func TestRegistry_RegisterValidation(t *testing.T) {
	tests := []struct {
		name string
		tool ToolDef
		want string
	}{
		{name: "empty name", tool: ToolDef{Schema: &schema.Schema{Type: "object"}, Handler: func(context.Context, map[string]any) (any, error) { return nil, nil }}, want: "tool name is required"},
		{name: "nil schema", tool: ToolDef{Name: "bad", Handler: func(context.Context, map[string]any) (any, error) { return nil, nil }}, want: "schema is required"},
		{name: "nil handler", tool: ToolDef{Name: "bad", Schema: &schema.Schema{Type: "object"}}, want: "handler is required"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewRegistry()
			err := r.Register(tt.tool)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("expected %q in error, got %v", tt.want, err)
			}
			if r.Count() != 0 {
				t.Fatalf("expected invalid tool not to be registered")
			}
		})
	}
}

func TestRegistry_RegisterRejectsDuplicate(t *testing.T) {
	r := NewRegistry()
	if err := r.Register(validTestTool("dup")); err != nil {
		t.Fatalf("Register first: %v", err)
	}
	err := r.Register(validTestTool("dup"))
	if err == nil {
		t.Fatal("expected duplicate error")
	}
	if !strings.Contains(err.Error(), "already registered") {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Count() != 1 {
		t.Fatalf("expected original tool to remain, count=%d", r.Count())
	}
}

func TestRegistry_NamesSorted(t *testing.T) {
	r := NewRegistry()
	for _, name := range []string{"zeta", "alpha", "middle"} {
		if err := r.Register(validTestTool(name)); err != nil {
			t.Fatalf("Register %s: %v", name, err)
		}
	}

	got := r.Names()
	want := []string{"alpha", "middle", "zeta"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected sorted names %v, got %v", want, got)
	}
}
