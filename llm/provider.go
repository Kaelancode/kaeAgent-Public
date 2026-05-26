package llm

import "context"

// Provider is the pluggable model-execution boundary used by the agent runtime.
// Implementations may call a vendor API directly, route through an internal
// gateway, submit work to a queue-backed worker system, or use local inference,
// as long as they preserve the normalized Request/Response/Event contract.
type Provider interface {
	Complete(ctx context.Context, req *Request) (*Response, error)
	Stream(ctx context.Context, req *Request) (<-chan Event, error)
	Models(ctx context.Context) ([]ModelInfo, error)
	Name() string
}

// Request is the normalized model-generation payload sent from the runtime to a
// Provider. Fields such as Model, Messages, Tools, MaxTokens, and Temperature
// describe model behavior. Options is reserved for provider-specific generation
// knobs such as top_p, top_k, or vendor flags; it should not become the primary
// home for runtime/business policy.
//
// Built-in direct providers are expected to be wrapped for backend controls
// such as rate limiting, concurrency caps, and transport retry. Third-party
// Provider implementations may manage those controls however they choose, as
// long as they honor the Provider contract.
type Request struct {
	Model       string         `json:"model"`
	Messages    []Message      `json:"messages"`
	Tools       []ToolDef      `json:"tools,omitempty"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature *float32       `json:"temperature,omitempty"`
	Options     map[string]any `json:"options,omitempty"`
	Execution   *ExecutionContext `json:"execution,omitempty"`
}

// ExecutionContext is the typed container for operational request
// metadata when a Provider implementation needs routing, correlation, or
// scheduling context beyond the model payload itself.
type ExecutionContext struct {
	SessionID string            `json:"session_id,omitempty"`
	UserID    string            `json:"user_id,omitempty"`
	AgentID   string            `json:"agent_id,omitempty"`
	RunID     string            `json:"run_id,omitempty"`
	StepIndex int               `json:"step_index,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

func sendEvent(ctx context.Context, ch chan<- Event, event Event) bool {
	select {
	case <-ctx.Done():
		return false
	case ch <- event:
		return true
	}
}

type Response struct {
	Content      []ContentBlock `json:"content"`
	Usage        Usage          `json:"usage"`
	FinishReason string         `json:"finish_reason"`
	Raw          any            `json:"-"`
}

type ContentBlock struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ToolCall *ToolCall `json:"tool_call,omitempty"`
}

type ToolCall struct {
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type ModelInfo struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	ContextWindow int    `json:"context_window"`
	Provider      string `json:"provider"`
}

type Event struct {
	Kind  EventKind      `json:"kind"`
	Text  *TextDelta     `json:"text,omitempty"`
	Tool  *ToolCallDelta `json:"tool,omitempty"`
	Usage *UsageDelta    `json:"usage,omitempty"`
	Err   error          `json:"-"`
}

type EventKind string

const (
	EventText       EventKind = "text_delta"
	EventToolCall   EventKind = "tool_call"
	EventToolResult EventKind = "tool_result"
	EventUsage      EventKind = "usage"
	EventError      EventKind = "error"
	EventDone       EventKind = "done"
)

type TextDelta struct {
	Content string `json:"content"`
}

type ToolCallDelta struct {
	Index int    `json:"index"`
	ID    string `json:"id"`
	Name  string `json:"name"`
	Input string `json:"input"`
}

type UsageDelta struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}
