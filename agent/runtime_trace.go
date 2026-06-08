package agent

import (
	"context"
	"strconv"

	agenttrace "github.com/Kaelancode/kaeAgent-Public/agent/internal/trace"
	"github.com/Kaelancode/kaeAgent-Public/observability"
)

type runTraceState struct {
	ctx       context.Context
	runSpan   observability.Span
	agentSpan observability.Span
}

func (e *runExecutor) startRunTrace(ctx context.Context, userMessage string) *runTraceState {
	trace := &runTraceState{ctx: ctx}
	if e.rt.tracer == nil {
		return trace
	}

	agentName := ""
	if e.rs.activeAgent != nil {
		agentName = e.rs.activeAgent.Name()
	}
	spanAttrs := map[string]string{
		"gen_ai.operation.name":  "invoke_agent",
		"gen_ai.conversation.id": e.rs.sessionID,
		"session.id":             e.rs.sessionID,
	}
	if agentName != "" {
		spanAttrs["gen_ai.agent.name"] = agentName
	}
	if e.rs.userID != "" {
		spanAttrs["user.id"] = e.rs.userID
	}
	if e.rs.agentID != "" {
		spanAttrs["gen_ai.agent.id"] = e.rs.agentID
	}
	spanName := "invoke_agent"
	if agentName != "" {
		spanName = "invoke_agent " + agentName
	}
	trace.ctx, trace.runSpan = e.rt.tracer.StartSpan(ctx, spanName, spanAttrs)
	trace.agentSpan = trace.runSpan

	inputJSON := agenttrace.TextMessageJSON("user", userMessage)
	e.rt.tracer.SetSpanAttributes(trace.ctx, trace.runSpan, map[string]any{
		"gen_ai.operation.name":      "invoke_agent",
		"gen_ai.input.messages":      inputJSON,
		"langfuse.observation.input": inputJSON,
	})
	e.rt.tracer.AddEvent(trace.ctx, trace.runSpan, "gen_ai.user.message", map[string]string{
		"role":                  "user",
		"content":               userMessage,
		"gen_ai.input.messages": inputJSON,
	})
	return trace
}

func (e *runExecutor) endRunTrace(trace *runTraceState, err error) {
	if e.rt.tracer == nil || trace == nil || trace.runSpan == nil {
		return
	}
	if trace.agentSpan != nil && trace.agentSpan != trace.runSpan {
		e.rt.tracer.EndSpan(trace.ctx, trace.agentSpan, err)
	}
	e.rt.tracer.EndSpan(trace.ctx, trace.runSpan, err)
}

func (e *runExecutor) recordStepStarted(trace *runTraceState, step *Step) {
	if e.rt.tracer == nil || trace == nil || trace.agentSpan == nil {
		return
	}
	e.rt.tracer.AddEvent(trace.ctx, trace.agentSpan, "agent.step.started", map[string]string{
		"gen_ai.step.index": strconv.Itoa(step.StepIndex),
		"gen_ai.agent.id":   step.AgentID,
		"gen_ai.agent.name": step.AgentName,
	})
}

func (e *runExecutor) recordStepCompleted(trace *runTraceState, stepIndex int, result *runLoopResult, err error) {
	if e.rt.tracer == nil || trace == nil || trace.agentSpan == nil {
		return
	}
	attrs := map[string]string{
		"gen_ai.step.index": strconv.Itoa(stepIndex),
	}
	if err != nil {
		attrs["status"] = "error"
		attrs["error"] = err.Error()
	} else {
		attrs["status"] = "ok"
		attrs["tool_calls"] = strconv.Itoa(len(result.ToolCalls))
		attrs["transfer"] = strconv.FormatBool(result.Transfer != nil)
	}
	e.rt.tracer.AddEvent(trace.ctx, trace.agentSpan, "agent.step.completed", attrs)
}

func (e *runExecutor) rotateTransferTrace(trace *runTraceState, fromAgent, toAgent, transferInput, fallbackText string) {
	if e.rt.tracer == nil || trace == nil || trace.runSpan == nil {
		return
	}
	e.rt.tracer.AddEvent(trace.ctx, trace.runSpan, "gen_ai.agent.transfer", map[string]string{
		"gen_ai.handoff.from_agent": fromAgent,
		"gen_ai.handoff.to_agent":   toAgent,
		"content":                   transferInput,
	})
	if trace.agentSpan != trace.runSpan {
		e.rt.tracer.EndSpan(trace.ctx, trace.agentSpan, nil)
	}
	agentSpanAttrs := map[string]string{
		"gen_ai.operation.name":  "invoke_agent",
		"gen_ai.conversation.id": e.rs.sessionID,
		"gen_ai.agent.name":      toAgent,
		"session.id":             e.rs.sessionID,
	}
	trace.ctx, trace.agentSpan = e.rt.tracer.StartSpan(trace.ctx, "invoke_agent "+toAgent, agentSpanAttrs)

	if transferInput == "" {
		transferInput = fallbackText
	}
	if transferInput == "" {
		transferInput = fromAgent + " requested transfer to " + toAgent
	}
	transferJSON := agenttrace.TextMessageJSON("user", transferInput)
	e.rt.tracer.SetSpanAttributes(trace.ctx, trace.agentSpan, map[string]any{
		"gen_ai.operation.name":      "invoke_agent",
		"gen_ai.input.messages":      transferJSON,
		"langfuse.observation.input": transferJSON,
	})
}

func (e *runExecutor) recordFinalTrace(trace *runTraceState, result *runLoopResult) {
	if e.rt.tracer == nil || trace == nil || trace.runSpan == nil {
		return
	}
	outputJSON, _ := agenttrace.AssistantTextOutput(result.Text, "")
	e.rt.tracer.AddEvent(trace.ctx, trace.runSpan, "gen_ai.assistant.message", map[string]string{
		"role":                   "assistant",
		"content":                result.Text,
		"gen_ai.output.messages": outputJSON,
	})
	e.rt.tracer.SetSpanAttributes(trace.ctx, trace.runSpan, map[string]any{
		"gen_ai.usage.input_tokens":   result.TokensUsed.InputTokens,
		"gen_ai.usage.output_tokens":  result.TokensUsed.OutputTokens,
		"gen_ai.output.messages":      outputJSON,
		"langfuse.observation.output": result.Text,
	})
	if trace.agentSpan != trace.runSpan {
		e.rt.tracer.SetSpanAttributes(trace.ctx, trace.agentSpan, map[string]any{
			"gen_ai.output.messages":      outputJSON,
			"langfuse.observation.output": result.Text,
		})
	}
}
