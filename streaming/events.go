// streaming/events.go
package streaming

// EventKind discriminates the type of a streaming event.
type EventKind string

const (
	EventText       EventKind = "text_delta"
	EventToolCall   EventKind = "tool_call"
	EventToolResult EventKind = "tool_result"
	EventUsage      EventKind = "usage"
	EventError      EventKind = "error"
	EventDone       EventKind = "done"
	EventFinalText  EventKind = "final_text"
)

// Event is the typed union emitted during streaming.
type Event struct {
	Kind   EventKind
	Text   *TextDelta
	Tool   *ToolCallDelta
	Result *ToolResultDelta
	Usage  *UsageDelta
	Final  *FinalTextDelta
	Err    error
}

// TextDelta carries a chunk of streamed text.
type TextDelta struct {
	Content string
}

// ToolCallDelta carries a streaming tool call fragment.
type ToolCallDelta struct {
	Index int
	ID    string
	Name  string
	Input string
}

// ToolResultDelta carries the result of a tool execution.
type ToolResultDelta struct {
	CallID  string
	Name    string
	Content string
}

// UsageDelta carries token usage information.
type UsageDelta struct {
	InputTokens  int
	OutputTokens int
	TotalTokens  int
}

type FinalTextDelta struct {
	Content string
}
