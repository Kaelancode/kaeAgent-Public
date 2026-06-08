package agent

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	agenttrace "github.com/Kaelancode/kaeAgent-Public/agent/internal/trace"
	"github.com/Kaelancode/kaeAgent-Public/observability"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func (e *runExecutor) dispatchWithTracing(ctx context.Context, parentSpan observability.Span, stepIndex int, calls []tools.ToolCall) []tools.ToolResult {
	groupID := fmt.Sprintf("%s/step/%d/tools", e.rs.runID, stepIndex)
	mode := "sequential"
	if e.rt.maxToolConcurrency > 1 && len(calls) > 1 {
		mode = "parallel"
	}
	toolIndexByCall := make(map[string]int, len(calls))
	for i, call := range calls {
		toolIndexByCall[call.ID+"\x00"+call.Name] = i
	}
	return tools.NewDispatcher(e.rs.tools).DispatchAllWith(ctx, calls, e.rt.maxToolConcurrency, func(ctx context.Context, call tools.ToolCall) tools.ToolResult {
		toolIndex, ok := toolIndexByCall[call.ID+"\x00"+call.Name]
		if !ok {
			toolIndex = 0
		}
		return e.dispatchOneWithTracing(ctx, parentSpan, call, stepIndex, toolIndex, groupID, mode)
	})
}

func (e *runExecutor) dispatchOneWithTracing(ctx context.Context, parentSpan observability.Span, call tools.ToolCall, stepIndex, toolIndex int, groupID, mode string) tools.ToolResult {
	var toolSpan observability.Span
	if e.rt.tracer != nil {
		if parentSpan != nil {
			e.rt.tracer.AddEvent(ctx, parentSpan, "agent.tool.started", map[string]string{
				"gen_ai.step.index":         strconv.Itoa(stepIndex),
				"gen_ai.tool.index":         strconv.Itoa(toolIndex),
				"gen_ai.tool.name":          call.Name,
				"gen_ai.tool.call_id":       call.ID,
				"gen_ai.execution.group.id": groupID,
				"gen_ai.execution.mode":     mode,
			})
		}
		spanAttrs := map[string]string{
			"gen_ai.operation.name":  "execute_tool",
			"gen_ai.provider.name":   agenttrace.ProviderName(e.rt.provider.Name()),
			"gen_ai.tool.name":       call.Name,
			"gen_ai.tool.call_id":    call.ID,
			"gen_ai.conversation.id": e.rs.sessionID,
			"session.id":             e.rs.sessionID,
		}
		if e.rs.userID != "" {
			spanAttrs["user.id"] = e.rs.userID
		}
		if e.rs.agentID != "" {
			spanAttrs["gen_ai.agent.id"] = e.rs.agentID
		}
		if e.rs.activeAgent != nil && e.rs.activeAgent.Name() != "" {
			spanAttrs["gen_ai.agent.name"] = e.rs.activeAgent.Name()
		}
		spanName := "execute_tool " + call.Name
		ctx, toolSpan = e.rt.tracer.StartSpan(ctx, spanName, spanAttrs)
		inputJSON := agenttrace.MustJSON(call.Input)
		e.rt.tracer.SetSpanAttributes(ctx, toolSpan, map[string]any{
			"gen_ai.operation.name":      "execute_tool",
			"gen_ai.step.index":          stepIndex,
			"gen_ai.tool.index":          toolIndex,
			"gen_ai.execution.group.id":  groupID,
			"gen_ai.execution.mode":      mode,
			"gen_ai.tool.input":          inputJSON,
			"langfuse.observation.input": inputJSON,
		})
		e.rt.tracer.AddEvent(ctx, toolSpan, "gen_ai.tool.message", map[string]string{
			"role":              "tool",
			"content":           inputJSON,
			"gen_ai.tool.input": inputJSON,
		})
	}

	result := e.dispatchToolCall(ctx, call)

	if e.rt.tracer != nil && parentSpan != nil {
		attrs := map[string]string{
			"gen_ai.step.index":         strconv.Itoa(stepIndex),
			"gen_ai.tool.index":         strconv.Itoa(toolIndex),
			"gen_ai.tool.name":          call.Name,
			"gen_ai.tool.call_id":       call.ID,
			"gen_ai.execution.group.id": groupID,
			"gen_ai.execution.mode":     mode,
			"status":                    "ok",
		}
		if result.Err != nil {
			attrs["status"] = "error"
			attrs["error"] = result.Err.Error()
		}
		e.rt.tracer.AddEvent(ctx, parentSpan, "agent.tool.completed", attrs)
	}

	if e.rt.tracer != nil && toolSpan != nil {
		if result.Err != nil {
			e.rt.tracer.AddEvent(ctx, toolSpan, "gen_ai.tool.message", map[string]string{
				"role":               "tool",
				"content":            result.Err.Error(),
				"gen_ai.tool.output": result.Err.Error(),
			})
			e.rt.tracer.SetSpanAttributes(ctx, toolSpan, map[string]any{
				"gen_ai.tool.output":          result.Err.Error(),
				"langfuse.observation.output": result.Err.Error(),
				"error.type":                  "tool_error",
			})
			e.rt.tracer.EndSpan(ctx, toolSpan, result.Err)
		} else {
			outputContent := tools.ResultToString(result)
			e.rt.tracer.AddEvent(ctx, toolSpan, "gen_ai.tool.message", map[string]string{
				"role":               "tool",
				"content":            outputContent,
				"gen_ai.tool.output": outputContent,
			})
			e.rt.tracer.SetSpanAttributes(ctx, toolSpan, map[string]any{
				"gen_ai.tool.output":          outputContent,
				"langfuse.observation.output": outputContent,
			})
			e.rt.tracer.EndSpan(ctx, toolSpan, nil)
		}
	}

	return result
}

func (e *runExecutor) dispatchToolCall(ctx context.Context, call tools.ToolCall) tools.ToolResult {
	if strings.HasPrefix(call.Name, consultToolPrefix) {
		return e.dispatchConsult(ctx, call)
	}
	return tools.NewDispatcher(e.rs.tools).Dispatch(ctx, call)
}

func (e *runExecutor) dispatchConsult(ctx context.Context, call tools.ToolCall) tools.ToolResult {
	target, ok := e.consultTargetForTool(call.Name)
	if !ok {
		agentName := e.rs.agentID
		if e.rs.activeAgent != nil && e.rs.activeAgent.Name() != "" {
			agentName = e.rs.activeAgent.Name()
		}
		return tools.ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("runtime: consult target %q is not available to agent %q", strings.TrimPrefix(call.Name, consultToolPrefix), agentName),
		}
	}
	if _, errs := consultToolSchema.CoerceAndValidate(call.Input); len(errs) > 0 {
		return tools.ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("runtime: consult validation failed for %q: %v", call.Name, errs),
		}
	}
	req, err := parseConsultRequest(target, call.Input)
	if err != nil {
		return tools.ToolResult{CallID: call.ID, Name: call.Name, Err: err}
	}

	child, err := e.prepareConsultRuntime(req)
	if err != nil {
		return tools.ToolResult{CallID: call.ID, Name: call.Name, Err: err}
	}
	out, err := child.Run(ctx, req.Input)
	if err != nil {
		return tools.ToolResult{
			CallID: call.ID,
			Name:   call.Name,
			Err:    fmt.Errorf("runtime: consult %q: %w", req.AgentName, err),
		}
	}
	return tools.ToolResult{CallID: call.ID, Name: call.Name, Content: out}
}
