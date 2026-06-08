package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestApplyEngineCommandsAppendsMessagesInOrder(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandAppendAssistantMessage,
			Data: map[string]any{"message": llm.Message{Role: "assistant", Content: "checking"}},
		},
		{
			Kind: agentengine.CommandAppendToolResults,
			Data: map[string]any{"results": []tools.ToolResult{
				{CallID: "call_1", Name: "lookup", Content: "result"},
			}},
		},
	})
	if err != nil {
		t.Fatalf("applyEngineCommands: %v", err)
	}

	msgs := exec.rs.conv.messagesOwned()
	if len(msgs) != 2 {
		t.Fatalf("expected two messages, got %+v", msgs)
	}
	if msgs[0].Role != "assistant" || msgs[0].Content != "checking" {
		t.Fatalf("unexpected assistant message: %+v", msgs[0])
	}
	if msgs[1].Role != "tool" || msgs[1].ToolCallID != "call_1" || msgs[1].Name != "lookup" || msgs[1].Content != "result" {
		t.Fatalf("unexpected tool message: %+v", msgs[1])
	}
}

func TestApplyEngineCommandsAppendsUserMessage(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandAppendUserMessage,
			Data: map[string]any{"message": llm.Message{
				Role:    "user",
				Content: "hello",
			}},
		},
	})
	if err != nil {
		t.Fatalf("applyEngineCommands: %v", err)
	}
	msgs := exec.rs.conv.messagesOwned()
	if len(msgs) != 1 || msgs[0].Role != "user" || msgs[0].Content != "hello" {
		t.Fatalf("unexpected user messages: %+v", msgs)
	}
}

func TestApplyEngineCommandsReturnsExecuteToolCallsWithoutDispatch(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()
	calls := []tools.ToolCall{
		{ID: "call_1", Name: "lookup", Input: map[string]any{"nested": map[string]any{"key": "value"}}},
	}

	applied, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandExecuteTools,
			Data: map[string]any{"calls": calls},
		},
	})
	if err != nil {
		t.Fatalf("applyEngineCommands: %v", err)
	}
	if len(applied.ToolCalls) != 1 || applied.ToolCalls[0].Name != "lookup" {
		t.Fatalf("unexpected applied tool calls: %+v", applied.ToolCalls)
	}
	applied.ToolCalls[0].Input["nested"].(map[string]any)["key"] = "mutated"
	if got := calls[0].Input["nested"].(map[string]any)["key"]; got != "value" {
		t.Fatalf("expected execute command output not to alias original calls, got %v", got)
	}
	if exec.rs.conv.Len() != 0 {
		t.Fatalf("execute_tools command should not mutate conversation, got %d messages", exec.rs.conv.Len())
	}
}

func TestApplyEngineCommandsSurfacesCheckpointFailure(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider:          &fakeProvider{},
		Session:           NewSession(SessionConfig{Model: "model"}),
		ConversationStore: &fakeConversationStore{saveErr: errors.New("boom")},
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{{Kind: agentengine.CommandCheckpoint}})
	if err == nil || err.Error() != "runtime: checkpoint conversation: boom" {
		t.Fatalf("expected checkpoint failure, got %v", err)
	}
}

func TestApplyEngineCommandsSurfacesSessionSaveFailure(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider:     &fakeProvider{},
		Session:      NewSession(SessionConfig{Model: "model"}),
		SessionStore: &fakeSessionStore{saveErr: errors.New("boom")},
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{{Kind: agentengine.CommandSaveSession}})
	if err == nil || err.Error() != "runtime: save session: boom" {
		t.Fatalf("expected save session failure, got %v", err)
	}
}

func TestApplyEngineCommandsRejectsMalformedCommand(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{Kind: agentengine.CommandAppendAssistantMessage, Data: map[string]any{}},
	})
	if err == nil || !strings.Contains(err.Error(), "missing message") {
		t.Fatalf("expected malformed command error, got %v", err)
	}
}

func TestApplyEngineCommandsWithOutputEmitsToolResultsAfterAppend(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()
	output := &capturingRunOutput{}

	_, err := exec.applyEngineCommandsWithOutput(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandAppendToolResults,
			Data: map[string]any{"results": []tools.ToolResult{
				{CallID: "call_1", Name: "lookup", Content: "result"},
			}},
		},
	}, output)
	if err != nil {
		t.Fatalf("applyEngineCommandsWithOutput: %v", err)
	}
	msgs := exec.rs.conv.messagesOwned()
	if len(msgs) != 1 || msgs[0].Role != "tool" || msgs[0].Content != "result" {
		t.Fatalf("expected tool result to be appended before output assertion, got %+v", msgs)
	}
	if len(output.events) != 1 || output.events[0] != "tool:call_1:lookup:result" {
		t.Fatalf("unexpected output events: %+v", output.events)
	}
}

func TestApplyEngineCommandsWithOutputEmitsFinalAndDone(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()
	output := &capturingRunOutput{}

	_, err := exec.applyEngineCommandsWithOutput(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandEmitOutput,
			Data: map[string]any{"type": "final_text", "text": "done"},
		},
		{
			Kind: agentengine.CommandEmitOutput,
			Data: map[string]any{"type": "done"},
		},
	}, output)
	if err != nil {
		t.Fatalf("applyEngineCommandsWithOutput: %v", err)
	}
	if got := strings.Join(output.events, ","); got != "final:done,done" {
		t.Fatalf("unexpected output events: %v", output.events)
	}
}

func TestApplyEngineCommandsWithOutputSurfacesOutputError(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()
	output := &capturingRunOutput{err: errors.New("blocked")}

	_, err := exec.applyEngineCommandsWithOutput(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandEmitOutput,
			Data: map[string]any{"type": "final_text", "text": "done"},
		},
	}, output)
	if !errors.Is(err, output.err) {
		t.Fatalf("expected output error, got %v", err)
	}
}

func TestApplyEngineCommandsWithTraceRecordsFinalTrace(t *testing.T) {
	tracer := &recordingTracer{}
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
		Tracer:   tracer,
	})
	exec := rt.newRunExecutor()
	trace := exec.startRunTrace(context.Background(), "hello")

	_, err := exec.applyEngineCommandsWithTrace(trace.ctx, []agentengine.Command{
		{
			Kind: agentengine.CommandTraceEvent,
			Data: map[string]any{
				"event": agentengine.Event{
					Kind: agentengine.EventStepCompleted,
					Data: map[string]any{"status": "final"},
				},
			},
		},
	}, trace, &runLoopResult{
		Text:       "done",
		TokensUsed: llm.Usage{InputTokens: 3, OutputTokens: 4},
	})
	if err != nil {
		t.Fatalf("applyEngineCommandsWithTrace: %v", err)
	}
	if _, ok := findRecordedTraceEvent(tracer.events, "gen_ai.assistant.message"); !ok {
		t.Fatalf("expected final assistant trace event, got %+v", tracer.events)
	}
	if !recordedAttrEquals(tracer.attrs, "langfuse.observation.output", "done") {
		t.Fatalf("expected final trace output attribute, got %+v", tracer.attrs)
	}
	if !recordedAttrEquals(tracer.attrs, "gen_ai.usage.input_tokens", 3) {
		t.Fatalf("expected input token trace attr, got %+v", tracer.attrs)
	}
}

func TestApplyEngineCommandsTraceEventMalformed(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommandsWithTrace(context.Background(), []agentengine.Command{
		{Kind: agentengine.CommandTraceEvent, Data: map[string]any{}},
	}, nil, nil)
	if err == nil || !strings.Contains(err.Error(), "trace event missing event") {
		t.Fatalf("expected malformed trace event error, got %v", err)
	}
}

func TestApplyEngineCommandsApplyTransfer(t *testing.T) {
	root := NewAgent(AgentConfig{Name: "root", Model: "root-model", Subagents: []string{"billing"}})
	billing := NewAgent(AgentConfig{Name: "billing", Model: "billing-model", SystemPrompt: "billing system"})
	resolver := staticSubagentResolver{"billing": billing}

	rt := NewRuntime(RuntimeConfig{
		Provider:         &fakeProvider{},
		Agent:            root,
		RootAgent:        root,
		SubagentResolver: resolver,
		Session:          NewSession(SessionConfig{Model: "root-model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandApplyTransfer,
			Data: map[string]any{
				"target_agent": "billing",
				"input":        "billing issue",
				"metadata":     map[string]string{"topic": "billing"},
			},
		},
	})
	if err != nil {
		t.Fatalf("apply transfer command: %v", err)
	}
	if exec.rs.activeAgent.Name() != "billing" || exec.rs.agentID != "billing" {
		t.Fatalf("expected active billing agent, got agent_id=%q active=%v", exec.rs.agentID, exec.rs.activeAgent.Name())
	}
	if exec.rs.metadata[ActiveAgentMetadataKey] != "billing" || exec.rs.metadata["topic"] != "billing" {
		t.Fatalf("unexpected transfer metadata: %+v", exec.rs.metadata)
	}
	msgs := exec.rs.conv.messagesOwned()
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[0].Content != "billing system" || msgs[1].Content != "billing issue" {
		t.Fatalf("unexpected transfer conversation: %+v", msgs)
	}
}

func TestApplyEngineCommandsApplyTransferUnknownTarget(t *testing.T) {
	root := NewAgent(AgentConfig{Name: "root", Subagents: []string{"billing"}})
	rt := NewRuntime(RuntimeConfig{
		Provider:         &fakeProvider{},
		Agent:            root,
		RootAgent:        root,
		SubagentResolver: staticSubagentResolver{},
		Session:          NewSession(SessionConfig{Model: "root-model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{
			Kind: agentengine.CommandApplyTransfer,
			Data: map[string]any{
				"target_agent": "billing",
				"input":        "billing issue",
			},
		},
	})
	if err == nil || !strings.Contains(err.Error(), `runtime: transfer: runtime: subagent "billing" not found`) {
		t.Fatalf("expected unknown target transfer error, got %v", err)
	}
}

func TestApplyEngineCommandsApplyTransferMalformed(t *testing.T) {
	rt := NewRuntime(RuntimeConfig{
		Provider: &fakeProvider{},
		Session:  NewSession(SessionConfig{Model: "model"}),
	})
	exec := rt.newRunExecutor()

	_, err := exec.applyEngineCommands(context.Background(), []agentengine.Command{
		{Kind: agentengine.CommandApplyTransfer, Data: map[string]any{"input": "missing target"}},
	})
	if err == nil || !strings.Contains(err.Error(), "missing target_agent") {
		t.Fatalf("expected malformed apply transfer error, got %v", err)
	}
}

type capturingRunOutput struct {
	events []string
	err    error
}

func (o *capturingRunOutput) EmitToolResult(_ context.Context, callID, name, content string) error {
	if o.err != nil {
		return o.err
	}
	o.events = append(o.events, fmt.Sprintf("tool:%s:%s:%s", callID, name, content))
	return nil
}

func (o *capturingRunOutput) EmitFinalText(_ context.Context, text string) error {
	if o.err != nil {
		return o.err
	}
	o.events = append(o.events, "final:"+text)
	return nil
}

func (o *capturingRunOutput) EmitDone(context.Context) error {
	if o.err != nil {
		return o.err
	}
	o.events = append(o.events, "done")
	return nil
}

func (o *capturingRunOutput) EmitError(_ context.Context, err error) error {
	if o.err != nil {
		return o.err
	}
	o.events = append(o.events, "error:"+err.Error())
	return nil
}

func recordedAttrEquals(attrs []map[string]any, key string, want any) bool {
	for _, attrSet := range attrs {
		if got, ok := attrSet[key]; ok && got == want {
			return true
		}
	}
	return false
}
