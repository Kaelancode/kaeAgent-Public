// tools/tool.go
package tools

import (
	"context"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

// ToolDef describes a callable tool with its schema, tags, and handler.
type ToolDef struct {
	Name        string
	Description string
	Schema      *schema.Schema
	Tags        []string
	Handler     func(ctx context.Context, input map[string]any) (any, error)
}

// ToolCall represents an invocation of a tool with resolved input.
type ToolCall struct {
	ID    string
	Name  string
	Input map[string]any
}

// ToolResult is the outcome of executing a ToolCall.
type ToolResult struct {
	CallID  string
	Name    string
	Content any
	Err     error
}
