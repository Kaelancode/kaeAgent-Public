package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	agentengine "github.com/yourorg/agent-sdk/agent/internal/engine"
	agentstreaming "github.com/yourorg/agent-sdk/agent/internal/streaming"
	agenttrace "github.com/yourorg/agent-sdk/agent/internal/trace"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

func (e *runExecutor) buildStepInfo(step int) *Step {
	return stepFromEngineStepInput(agentengine.BuildStepInput(
		engineStateFromRunState(e.rs),
		e.availableToolDefs(),
		e.rt.provider.Name(),
		step,
	))
}

func (e *runExecutor) buildStep(step int) *Step {
	return e.buildStepInfo(step)
}

func normalizeStepResult(result *StepResult) *runLoopResult {
	text := ""
	if result.Response != nil {
		text = extractResponseText(result.Response)
	}
	return &runLoopResult{
		Response:   result.Response,
		ToolCalls:  result.ToolCalls,
		Transfer:   result.Transfer,
		TokensUsed: result.TokensUsed,
		Text:       text,
	}
}

func normalizeStreamingStepResult(result *StreamingStepResult) *runLoopResult {
	text := result.StreamedText
	if text == "" && result.Response != nil {
		text = extractResponseText(result.Response)
	}
	return &runLoopResult{
		Response:   result.Response,
		ToolCalls:  result.ToolCalls,
		Transfer:   result.Transfer,
		TokensUsed: result.TokensUsed,
		Text:       text,
	}
}

func (e *runExecutor) finishStreamingLLMSpan(ctx context.Context, span observability.Span, req *llm.Request, text string, toolCalls []tools.ToolCall, usage llm.Usage, finishReason string, err error) {
	if e.rt.tracer == nil || span == nil {
		return
	}
	responseAttrs := map[string]any{
		"gen_ai.response.finish_reasons": []string{finishReason},
		"gen_ai.response.model":          req.Model,
		"gen_ai.usage.input_tokens":      usage.InputTokens,
		"gen_ai.usage.output_tokens":     usage.OutputTokens,
	}
	if text != "" {
		outputJSON, _ := agenttrace.AssistantTextOutput(text, finishReason)
		responseAttrs["gen_ai.output.messages"] = outputJSON
		responseAttrs["langfuse.observation.output"] = text
		e.rt.tracer.AddEvent(ctx, span, "gen_ai.assistant.message", map[string]string{
			"role":                   "assistant",
			"content":                text,
			"gen_ai.output.messages": outputJSON,
		})
	} else if len(toolCalls) > 0 {
		outputJSON, outputSummary := agenttrace.AssistantToolCallsOutputFromTools(toolCalls, finishReason)
		responseAttrs["gen_ai.output.messages"] = outputJSON
		responseAttrs["langfuse.observation.output"] = outputSummary
		e.rt.tracer.AddEvent(ctx, span, "gen_ai.assistant.message", map[string]string{
			"role":                   "assistant",
			"gen_ai.output.messages": outputJSON,
			"content":                outputSummary,
		})
	} else {
		outputJSON, _ := agenttrace.AssistantTextOutput("", finishReason)
		responseAttrs["gen_ai.output.messages"] = outputJSON
		responseAttrs["langfuse.observation.output"] = ""
		e.rt.tracer.AddEvent(ctx, span, "gen_ai.assistant.message", map[string]string{
			"role":                   "assistant",
			"gen_ai.output.messages": outputJSON,
		})
	}
	e.rt.tracer.SetSpanAttributes(ctx, span, responseAttrs)
	e.rt.tracer.EndSpan(ctx, span, err)
}

func (e *runExecutor) handleTransferStep(trace *runTraceState, output runOutputAdapter, result *runLoopResult) error {
	commands := agentengine.PlanStepCommands(agentengine.StepInput{}, agentengine.StepOutput{Text: result.Text}, transferPlanFromStep(result.Transfer))
	return e.executeTransferCommandPlan(trace.ctx, commands, output, trace, result)
}

func (e *runExecutor) handleFinalStep(trace *runTraceState, output runOutputAdapter, result *runLoopResult) (string, error) {
	commands := agentengine.PlanStepCommands(agentengine.StepInput{}, agentengine.StepOutput{Text: result.Text}, nil)
	if err := e.executeFinalCommandPlan(trace.ctx, commands, output, trace, result); err != nil {
		return "", err
	}
	return result.Text, nil
}

func (e *runExecutor) handleToolCallStep(trace *runTraceState, output runOutputAdapter, step int, result *runLoopResult) error {
	commands := agentengine.PlanStepCommands(agentengine.StepInput{StepIndex: step}, agentengine.StepOutput{
		Text:      result.Text,
		ToolCalls: result.ToolCalls,
	}, nil)
	return e.executeToolCommandPlan(trace.ctx, commands, output, trace, step)
}

func transferPlanFromStep(step *TransferStep) *agentengine.TransferPlan {
	if step == nil {
		return nil
	}
	return &agentengine.TransferPlan{
		Call:           step.Call,
		TargetAgent:    step.Request.AgentName,
		Input:          step.Request.Input,
		Metadata:       cloneStringMap(step.Request.Metadata),
		Acknowledgment: fmt.Sprintf("Transferred control to %s.", step.Request.AgentName),
	}
}

func (e *runExecutor) buildRequest(stepMessages []llm.Message, stepTools []tools.ToolDef, stepIndex int) *llm.Request {
	return agentengine.BuildRequest(agentengine.StepInput{
		SessionID:      e.rs.sessionID,
		RunID:          e.rs.runID,
		StepIndex:      stepIndex,
		Messages:       stepMessages,
		AvailableTools: stepTools,
		UserID:         e.rs.userID,
		AgentID:        e.rs.agentID,
		Metadata:       cloneStringMap(e.rs.metadata),
	}, e.engineConfig())
}

func (e *runExecutor) executeStep(ctx context.Context, step *Step) (*StepResult, error) {
	messages, err := e.prepareMessagesForRequest(ctx, step.SessionID, step.Messages, step.AvailTools)
	if err != nil {
		return nil, fmt.Errorf("runtime: preflight context guard: %w", err)
	}

	req := e.buildRequest(messages, step.AvailTools, step.StepIndex)
	e.rt.logger.Debug().Interface("request", req).Msg("llm.request")

	var llmSpan observability.Span
	if e.rt.tracer != nil {
		inputMessages := agenttrace.MessagesToOTelFormat(req.Messages)
		inputJSON := agenttrace.MustJSON(inputMessages)
		spanAttrs := map[string]string{
			"gen_ai.operation.name":  "chat",
			"gen_ai.provider.name":   agenttrace.ProviderName(e.rt.provider.Name()),
			"gen_ai.request.model":   req.Model,
			"gen_ai.conversation.id": step.SessionID,
			"session.id":             step.SessionID,
		}
		if step.UserID != "" {
			spanAttrs["user.id"] = step.UserID
		}
		if step.AgentID != "" {
			spanAttrs["gen_ai.agent.id"] = step.AgentID
		}
		if step.AgentName != "" {
			spanAttrs["gen_ai.agent.name"] = step.AgentName
		}
		spanName := "chat " + req.Model
		ctx, llmSpan = e.rt.tracer.StartSpan(ctx, spanName, spanAttrs)
		attrs := map[string]any{
			"gen_ai.operation.name":      "chat",
			"gen_ai.request.max_tokens":  req.MaxTokens,
			"gen_ai.request.stream":      false,
			"gen_ai.step.index":          step.StepIndex,
			"gen_ai.input.messages":      string(inputJSON),
			"langfuse.observation.input": string(inputJSON),
		}
		if req.Temperature != nil {
			attrs["gen_ai.request.temperature"] = float64(*req.Temperature)
		}
		if len(req.Tools) > 0 {
			toolsJSON, _ := json.Marshal(req.Tools)
			attrs["gen_ai.tool.definitions"] = string(toolsJSON)
		}
		e.rt.tracer.SetSpanAttributes(ctx, llmSpan, attrs)
		for _, msg := range inputMessages {
			role, _ := msg["role"].(string)
			eventName := "gen_ai.user.message"
			if role == "system" {
				eventName = "gen_ai.system.message"
			} else if role == "assistant" {
				eventName = "gen_ai.assistant.message"
			} else if role == "tool" {
				eventName = "gen_ai.tool.message"
			}
			msgJSON := agenttrace.MustJSON(msg)
			content := agenttrace.ExtractContentFromOTelMsg(msg)
			eventAttrs := map[string]string{
				"role":                  role,
				"gen_ai.input.messages": msgJSON,
			}
			if content != "" {
				eventAttrs["content"] = content
			}
			e.rt.tracer.AddEvent(ctx, llmSpan, eventName, eventAttrs)
		}
	}

	engineStep := agentengine.StepInput{
		SessionID:      step.SessionID,
		RunID:          e.rs.runID,
		StepIndex:      step.StepIndex,
		Messages:       messages,
		AvailableTools: step.AvailTools,
		ProviderName:   step.ProviderName,
		UserID:         step.UserID,
		AgentID:        step.AgentID,
		AgentName:      step.AgentName,
		Metadata:       cloneStringMap(e.rs.metadata),
	}
	engineOut, err := agentengine.ExecuteStep(ctx, engineStep, e.engineConfig(), e.engineHooks())
	if err != nil {
		e.rt.logger.Error().Err(err).Msg("llm.error")
		if e.rt.tracer != nil && llmSpan != nil {
			e.rt.tracer.EndSpan(ctx, llmSpan, err)
		}
		return nil, fmt.Errorf("runtime: provider complete: %w", err)
	}

	resp := engineOut.Response
	e.rt.logger.Debug().Interface("response", resp).Msg("llm.response")
	if e.rt.tracer != nil && llmSpan != nil {
		responseAttrs := map[string]any{
			"gen_ai.response.finish_reasons": []string{resp.FinishReason},
			"gen_ai.response.model":          req.Model,
			"gen_ai.usage.input_tokens":      resp.Usage.InputTokens,
			"gen_ai.usage.output_tokens":     resp.Usage.OutputTokens,
		}
		if text := extractResponseText(resp); text != "" {
			outputJSON, _ := agenttrace.AssistantTextOutput(text, resp.FinishReason)
			responseAttrs["gen_ai.output.messages"] = outputJSON
			responseAttrs["langfuse.observation.output"] = text
			e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
				"role":                   "assistant",
				"content":                text,
				"gen_ai.output.messages": outputJSON,
			})
		} else {
			outputJSON, outputSummary := agenttrace.AssistantToolCallsOutputFromResponse(resp)
			responseAttrs["gen_ai.output.messages"] = outputJSON
			responseAttrs["langfuse.observation.output"] = outputSummary
			e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
				"role":                   "assistant",
				"gen_ai.output.messages": outputJSON,
				"content":                outputSummary,
			})
		}
		e.rt.tracer.SetSpanAttributes(ctx, llmSpan, responseAttrs)
		e.rt.tracer.EndSpan(ctx, llmSpan, nil)
	}

	toolCalls := engineOut.ToolCalls

	transferStep, err := e.extractTransfer(toolCalls, engineOut.Text)
	if err != nil {
		return nil, err
	}
	if transferStep != nil {
		return &StepResult{
			Response:   resp,
			Transfer:   transferStep,
			TokensUsed: engineOut.Usage,
		}, nil
	}

	return &StepResult{
		Response:   resp,
		ToolCalls:  toolCalls,
		TokensUsed: engineOut.Usage,
	}, nil
}

func (e *runExecutor) executeStreamingStep(ctx context.Context, step *StreamingStep, out chan<- streaming.Event) (*StreamingStepResult, error) {
	messages, err := e.prepareMessagesForRequest(ctx, step.SessionID, step.Messages, step.AvailTools)
	if err != nil {
		return nil, fmt.Errorf("runtime: preflight context guard: %w", err)
	}

	req := e.buildRequest(messages, step.AvailTools, step.StepIndex)
	e.rt.logger.Debug().Interface("request", req).Msg("llm.stream.request")

	var llmSpan observability.Span
	if e.rt.tracer != nil {
		inputMessages := agenttrace.MessagesToOTelFormat(req.Messages)
		inputJSON := agenttrace.MustJSON(inputMessages)
		spanAttrs := map[string]string{
			"gen_ai.operation.name":  "chat",
			"gen_ai.provider.name":   agenttrace.ProviderName(e.rt.provider.Name()),
			"gen_ai.request.model":   req.Model,
			"gen_ai.conversation.id": step.SessionID,
			"session.id":             step.SessionID,
		}
		if step.UserID != "" {
			spanAttrs["user.id"] = step.UserID
		}
		if step.AgentID != "" {
			spanAttrs["gen_ai.agent.id"] = step.AgentID
		}
		if step.AgentName != "" {
			spanAttrs["gen_ai.agent.name"] = step.AgentName
		}
		spanName := "chat " + req.Model
		ctx, llmSpan = e.rt.tracer.StartSpan(ctx, spanName, spanAttrs)
		attrs := map[string]any{
			"gen_ai.operation.name":      "chat",
			"gen_ai.request.max_tokens":  req.MaxTokens,
			"gen_ai.request.stream":      true,
			"gen_ai.step.index":          step.StepIndex,
			"gen_ai.input.messages":      string(inputJSON),
			"langfuse.observation.input": string(inputJSON),
		}
		if req.Temperature != nil {
			attrs["gen_ai.request.temperature"] = float64(*req.Temperature)
		}
		if len(req.Tools) > 0 {
			toolsJSON, _ := json.Marshal(req.Tools)
			attrs["gen_ai.tool.definitions"] = string(toolsJSON)
		}
		e.rt.tracer.SetSpanAttributes(ctx, llmSpan, attrs)
		for _, msg := range inputMessages {
			role, _ := msg["role"].(string)
			eventName := "gen_ai.user.message"
			if role == "system" {
				eventName = "gen_ai.system.message"
			} else if role == "assistant" {
				eventName = "gen_ai.assistant.message"
			} else if role == "tool" {
				eventName = "gen_ai.tool.message"
			}
			msgJSON := agenttrace.MustJSON(msg)
			content := agenttrace.ExtractContentFromOTelMsg(msg)
			eventAttrs := map[string]string{
				"role":                  role,
				"gen_ai.input.messages": msgJSON,
			}
			if content != "" {
				eventAttrs["content"] = content
			}
			e.rt.tracer.AddEvent(ctx, llmSpan, eventName, eventAttrs)
		}
	}

	source, err := e.rt.provider.Stream(ctx, req)
	if err != nil {
		e.rt.logger.Error().Err(err).Msg("llm.stream.error")
		if e.rt.tracer != nil && llmSpan != nil {
			e.rt.tracer.EndSpan(ctx, llmSpan, err)
		}
		return nil, fmt.Errorf("runtime: provider stream: %w", err)
	}

	var textBuilder strings.Builder
	assembler := agentstreaming.NewToolCallAssembler()
	var usage llm.Usage
	var responseFinishReason string

	for {
		select {
		case <-ctx.Done():
			if e.rt.tracer != nil && llmSpan != nil {
				e.rt.tracer.EndSpan(ctx, llmSpan, ctx.Err())
			}
			return nil, fmt.Errorf("runtime: stream cancelled: %w", ctx.Err())
		case event, ok := <-source:
			if !ok {
				toolCalls, asmErr := assembler.Assemble()
				if asmErr != nil {
					e.finishStreamingLLMSpan(ctx, llmSpan, req, textBuilder.String(), nil, usage, responseFinishReason, asmErr)
					return nil, asmErr
				}
				err := fmt.Errorf("runtime: stream ended without terminal event")
				e.finishStreamingLLMSpan(ctx, llmSpan, req, textBuilder.String(), toolCalls, usage, responseFinishReason, err)
				return nil, err
			}

			switch event.Kind {
			case llm.EventText:
				if event.Text != nil {
					textBuilder.WriteString(event.Text.Content)
					if err := e.rt.sendStreamingEvent(ctx, out, streaming.Event{
						Kind: streaming.EventText,
						Text: &streaming.TextDelta{Content: event.Text.Content},
					}); err != nil {
						return nil, err
					}
				}
			case llm.EventToolCall:
				if event.Tool != nil && (event.Tool.ID != "" || event.Tool.Name != "" || event.Tool.Input != "") {
					assembler.AddFragment(event.Tool.Index, event.Tool)
					if err := e.rt.sendStreamingEvent(ctx, out, streaming.Event{
						Kind: streaming.EventToolCall,
						Tool: &streaming.ToolCallDelta{
							Index: event.Tool.Index,
							ID:    event.Tool.ID,
							Name:  event.Tool.Name,
							Input: event.Tool.Input,
						},
					}); err != nil {
						return nil, err
					}
				}
			case llm.EventUsage:
				if event.Usage != nil {
					usage.InputTokens += event.Usage.InputTokens
					usage.OutputTokens += event.Usage.OutputTokens
					usage.TotalTokens += event.Usage.TotalTokens
					if err := e.rt.sendStreamingEvent(ctx, out, streaming.Event{
						Kind: streaming.EventUsage,
						Usage: &streaming.UsageDelta{
							InputTokens:  event.Usage.InputTokens,
							OutputTokens: event.Usage.OutputTokens,
							TotalTokens:  event.Usage.TotalTokens,
						},
					}); err != nil {
						return nil, err
					}
				}
			case llm.EventError:
				if event.Err != nil {
					if e.rt.tracer != nil && llmSpan != nil {
						e.rt.tracer.EndSpan(ctx, llmSpan, event.Err)
					}
					return nil, fmt.Errorf("runtime: stream error: %w", event.Err)
				}
			case llm.EventDone:
				toolCalls, asmErr := assembler.Assemble()
				if asmErr != nil {
					return nil, asmErr
				}
				e.finishStreamingLLMSpan(ctx, llmSpan, req, textBuilder.String(), toolCalls, usage, responseFinishReason, nil)

				transferStep, transferErr := e.extractTransfer(toolCalls, textBuilder.String())
				if transferErr != nil {
					return nil, transferErr
				}
				return &StreamingStepResult{
					Response: &llm.Response{
						Usage:        usage,
						FinishReason: responseFinishReason,
					},
					ToolCalls:    toolCalls,
					Transfer:     transferStep,
					TokensUsed:   usage,
					StreamedText: textBuilder.String(),
				}, nil
			}
		}
	}
}
