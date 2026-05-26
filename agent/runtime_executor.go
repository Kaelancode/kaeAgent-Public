package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	"github.com/yourorg/agent-sdk/store"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

type runState struct {
	generation  uint64
	sessionID   string
	runID       string
	config      SessionConfig
	metadata    map[string]string
	budget      *streaming.Budget
	conv        *Conversation
	userID      string
	agentID     string
	activeAgent *Agent
	rootAgent   *Agent
	tools       *tools.Registry
}

type runExecutor struct {
	rt *Runtime
	rs *runState
}

func (r *Runtime) Run(ctx context.Context, userMessage string) (string, error) {
	if !r.HasProvider() {
		return "", fmt.Errorf("runtime: provider is nil")
	}

	exec := r.newRunExecutor()
	handler := Chain(exec.executeStep, r.middleware...)
	exec.rs.conv.appendOwned(llm.Message{Role: "user", Content: userMessage})

	var runSpan observability.Span
	var agentSpan observability.Span
	endSpans := func(err error) {
		if r.tracer == nil || runSpan == nil {
			return
		}
		if agentSpan != nil && agentSpan != runSpan {
			r.tracer.EndSpan(ctx, agentSpan, err)
		}
		r.tracer.EndSpan(ctx, runSpan, err)
	}
	if r.tracer != nil {
		agentName := ""
		if exec.rs.activeAgent != nil {
			agentName = exec.rs.activeAgent.Name()
		}
		spanAttrs := map[string]string{
			"gen_ai.operation.name":  "invoke_agent",
			"gen_ai.conversation.id": exec.rs.sessionID,
			"session.id":            exec.rs.sessionID,
		}
		if agentName != "" {
			spanAttrs["gen_ai.agent.name"] = agentName
		}
		if exec.rs.userID != "" {
			spanAttrs["user.id"] = exec.rs.userID
		}
		if exec.rs.agentID != "" {
			spanAttrs["gen_ai.agent.id"] = exec.rs.agentID
		}
		spanName := "invoke_agent"
		if agentName != "" {
			spanName = "invoke_agent " + agentName
		}
		ctx, runSpan = r.tracer.StartSpan(ctx, spanName, spanAttrs)
		agentSpan = runSpan
		inputMsg := map[string]any{
			"role": "user",
			"parts": []map[string]any{
				{"type": "text", "content": userMessage},
			},
		}
		inputJSON, _ := json.Marshal([]map[string]any{inputMsg})
		r.tracer.SetSpanAttributes(ctx, runSpan, map[string]any{
			"gen_ai.operation.name":    "invoke_agent",
			"gen_ai.input.messages":    string(inputJSON),
			"langfuse.observation.input":  string(inputJSON),
		})
		r.tracer.AddEvent(ctx, runSpan, "gen_ai.user.message", map[string]string{
			"role":                    "user",
			"content":                 userMessage,
			"gen_ai.input.messages":  string(inputJSON),
		})
	}

	for step := 0; step < r.maxSteps; step++ {
		select {
		case <-ctx.Done():
			endSpans(ctx.Err())
			return "", fmt.Errorf("runtime: context cancelled: %w", ctx.Err())
		default:
		}

		if err := exec.rs.budget.Check(); err != nil {
			endSpans(err)
			return "", fmt.Errorf("runtime: %w", err)
		}

stepInput := &Step{
		SessionID:    exec.rs.sessionID,
		RunID:        exec.rs.runID,
		StepIndex:    step,
		Messages:     exec.rs.conv.messagesOwned(),
		AvailTools:   exec.availableToolDefs(),
		ProviderName: r.provider.Name(),
		UserID:       exec.rs.userID,
		AgentID:      exec.rs.agentID,
	}
	if exec.rs.activeAgent != nil {
		stepInput.AgentName = exec.rs.activeAgent.Name()
	}

		result, err := handler(ctx, stepInput)
		if err != nil {
			endSpans(err)
			return "", fmt.Errorf("runtime: step %d: %w", step, err)
		}

		exec.rs.budget.Add(result.TokensUsed.InputTokens, result.TokensUsed.OutputTokens)

		if result.Transfer != nil {
			assistantMsg := llm.Message{Role: "assistant"}
			assistantMsg.ToolCalls = []llm.ToolCall{{
				ID:    result.Transfer.Call.ID,
				Name:  result.Transfer.Call.Name,
				Input: cloneMap(result.Transfer.Call.Input),
			}}
			if textContent := extractResponseText(result.Response); textContent != "" {
				assistantMsg.Content = textContent
			}
			exec.rs.conv.appendOwned(assistantMsg)
			if err := exec.checkpoint(ctx); err != nil {
				endSpans(err)
				return "", err
			}

			transferAck := fmt.Sprintf("Transferred control to %s.", result.Transfer.Request.AgentName)
			exec.rs.conv.appendOwned(llm.Message{
				Role:       "tool",
				Content:    transferAck,
				ToolCallID: result.Transfer.Call.ID,
				Name:       result.Transfer.Call.Name,
			})
			if err := exec.checkpoint(ctx); err != nil {
				endSpans(err)
				return "", err
			}

			if err := exec.applyTransfer(result.Transfer.Request); err != nil {
				endSpans(fmt.Errorf("runtime: transfer: %w", err))
				return "", fmt.Errorf("runtime: transfer: %w", err)
			}
			if r.tracer != nil && runSpan != nil {
				fromAgent := exec.rs.activeAgent.Name()
				r.tracer.AddEvent(ctx, runSpan, "gen_ai.agent.transfer", map[string]string{
					"gen_ai.handoff.from_agent": fromAgent,
					"gen_ai.handoff.to_agent":   result.Transfer.Request.AgentName,
					"content":                  result.Transfer.Request.Input,
				})
				if agentSpan != runSpan {
					r.tracer.EndSpan(ctx, agentSpan, nil)
				}
				targetName := result.Transfer.Request.AgentName
				agentSpanAttrs := map[string]string{
					"gen_ai.operation.name":  "invoke_agent",
					"gen_ai.conversation.id": exec.rs.sessionID,
					"gen_ai.agent.name":      targetName,
					"session.id":             exec.rs.sessionID,
				}
				agentCtx, newAgentSpan := r.tracer.StartSpan(ctx, "invoke_agent "+targetName, agentSpanAttrs)
				agentSpan = newAgentSpan
				ctx = agentCtx
				transferInput := result.Transfer.Request.Input
				if transferInput == "" {
					transferInput = extractResponseText(result.Response)
				}
				if transferInput == "" {
					transferInput = fromAgent + " requested transfer to " + targetName
				}
				transferMsg := map[string]any{
					"role": "user",
					"parts": []map[string]any{
						{"type": "text", "content": transferInput},
					},
				}
				transferJSON, _ := json.Marshal([]map[string]any{transferMsg})
				r.tracer.SetSpanAttributes(ctx, agentSpan, map[string]any{
					"gen_ai.operation.name":      "invoke_agent",
					"gen_ai.input.messages":      string(transferJSON),
					"langfuse.observation.input": string(transferJSON),
				})
			}
			continue
		}

		if len(result.ToolCalls) == 0 {
			text := extractResponseText(result.Response)
			exec.rs.conv.appendOwned(llm.Message{Role: "assistant", Content: text})
			if err := exec.compactConversation(ctx); err != nil {
				endSpans(err)
				return "", fmt.Errorf("runtime: compact conversation: %w", err)
			}
			if err := exec.checkpoint(ctx); err != nil {
				endSpans(err)
				return "", err
			}
			if err := exec.saveSessionData(ctx); err != nil {
				endSpans(err)
				return "", err
			}
			r.publishRunState(exec.rs)

			if r.tracer != nil && runSpan != nil {
				outputMsg := map[string]any{
					"role":          "assistant",
					"finish_reason": "",
					"parts": []map[string]any{
						{"type": "text", "content": text},
					},
				}
				outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
				r.tracer.AddEvent(ctx, runSpan, "gen_ai.assistant.message", map[string]string{
					"role":                     "assistant",
					"content":                 text,
					"gen_ai.output.messages":  string(outputJSON),
				})
				r.tracer.SetSpanAttributes(ctx, runSpan, map[string]any{
					"gen_ai.usage.input_tokens":    result.TokensUsed.InputTokens,
					"gen_ai.usage.output_tokens":   result.TokensUsed.OutputTokens,
					"gen_ai.output.messages":      string(outputJSON),
					"langfuse.observation.output":  text,
				})
				if agentSpan != runSpan {
					r.tracer.SetSpanAttributes(ctx, agentSpan, map[string]any{
						"gen_ai.output.messages":      string(outputJSON),
						"langfuse.observation.output": text,
					})
				}
				endSpans(nil)
			}
			return text, nil
		}

		assistantMsg := llm.Message{Role: "assistant"}
		llmToolCalls := make([]llm.ToolCall, len(result.ToolCalls))
		for i, tc := range result.ToolCalls {
			llmToolCalls[i] = llm.ToolCall{ID: tc.ID, Name: tc.Name, Input: cloneMap(tc.Input)}
		}
		assistantMsg.ToolCalls = llmToolCalls
		if textContent := extractResponseText(result.Response); textContent != "" {
			assistantMsg.Content = textContent
		}
		exec.rs.conv.appendOwned(assistantMsg)
		if err := exec.checkpoint(ctx); err != nil {
			endSpans(err)
			return "", err
		}

		toolResults := exec.dispatchWithTracing(ctx, result.ToolCalls)
		for _, tr := range toolResults {
			content := tools.ResultToString(tr)
			exec.rs.conv.appendOwned(llm.Message{
				Role:       "tool",
				Content:    content,
				ToolCallID: tr.CallID,
				Name:       tr.Name,
			})
		}
		if err := exec.checkpoint(ctx); err != nil {
			endSpans(err)
			return "", err
		}
	}

	err := fmt.Errorf("runtime: max steps (%d) exceeded", r.maxSteps)
	endSpans(err)
	return "", err
}

func (r *Runtime) Stream(ctx context.Context, userMessage string) (<-chan streaming.Event, error) {
	if !r.HasProvider() {
		return nil, fmt.Errorf("runtime: provider is nil")
	}

	exec := r.newRunExecutor()
	handler := ChainStreaming(exec.executeStreamingStep, r.streamMiddleware...)
	exec.rs.conv.appendOwned(llm.Message{Role: "user", Content: userMessage})

	out := make(chan streaming.Event, 128)
	go func() {
		defer close(out)

		var runSpan observability.Span
		var agentSpan observability.Span
		endSpans := func(err error) {
			if r.tracer == nil || runSpan == nil {
				return
			}
			if agentSpan != nil && agentSpan != runSpan {
				r.tracer.EndSpan(ctx, agentSpan, err)
			}
			r.tracer.EndSpan(ctx, runSpan, err)
		}
		if r.tracer != nil {
			agentName := ""
			if exec.rs.activeAgent != nil {
				agentName = exec.rs.activeAgent.Name()
			}
			spanAttrs := map[string]string{
				"gen_ai.operation.name":  "invoke_agent",
				"gen_ai.conversation.id": exec.rs.sessionID,
				"session.id":            exec.rs.sessionID,
			}
			if agentName != "" {
				spanAttrs["gen_ai.agent.name"] = agentName
			}
			if exec.rs.userID != "" {
				spanAttrs["user.id"] = exec.rs.userID
			}
			if exec.rs.agentID != "" {
				spanAttrs["gen_ai.agent.id"] = exec.rs.agentID
			}
			spanName := "invoke_agent"
			if agentName != "" {
				spanName = "invoke_agent " + agentName
			}
			runCtx, span := r.tracer.StartSpan(ctx, spanName, spanAttrs)
			ctx = runCtx
			runSpan = span
			agentSpan = runSpan
			inputMsg := map[string]any{
				"role": "user",
				"parts": []map[string]any{
					{"type": "text", "content": userMessage},
				},
			}
			inputJSON, _ := json.Marshal([]map[string]any{inputMsg})
			r.tracer.SetSpanAttributes(ctx, runSpan, map[string]any{
				"gen_ai.operation.name":      "invoke_agent",
				"gen_ai.input.messages":      string(inputJSON),
				"langfuse.observation.input": string(inputJSON),
			})
			r.tracer.AddEvent(ctx, runSpan, "gen_ai.user.message", map[string]string{
				"role":                    "user",
				"content":                 userMessage,
				"gen_ai.input.messages":  string(inputJSON),
			})
		}

		for step := 0; step < r.maxSteps; step++ {
			select {
			case <-ctx.Done():
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: fmt.Errorf("runtime: context cancelled: %w", ctx.Err())})
				endSpans(ctx.Err())
				return
			default:
			}

			if err := exec.rs.budget.Check(); err != nil {
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: fmt.Errorf("runtime: %w", err)})
				endSpans(err)
				return
			}

			stepInput := &StreamingStep{
				SessionID:    exec.rs.sessionID,
				RunID:        exec.rs.runID,
				StepIndex:    step,
				Messages:     exec.rs.conv.messagesOwned(),
				AvailTools:   exec.availableToolDefs(),
				ProviderName: r.provider.Name(),
				UserID:       exec.rs.userID,
				AgentID:      exec.rs.agentID,
			}
			if exec.rs.activeAgent != nil {
				stepInput.AgentName = exec.rs.activeAgent.Name()
			}

			result, err := handler(ctx, stepInput, out)
			if err != nil {
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: fmt.Errorf("runtime: step %d: %w", step, err)})
				endSpans(err)
				return
			}
			if result == nil {
				nilErr := fmt.Errorf("runtime: step %d: nil result", step)
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: nilErr})
				endSpans(nilErr)
				return
			}

			exec.rs.budget.Add(result.TokensUsed.InputTokens, result.TokensUsed.OutputTokens)

			if result.Transfer != nil {
				assistantMsg := llm.Message{Role: "assistant"}
				assistantMsg.ToolCalls = []llm.ToolCall{{
					ID:    result.Transfer.Call.ID,
					Name:  result.Transfer.Call.Name,
					Input: cloneMap(result.Transfer.Call.Input),
				}}
				textContent := result.StreamedText
				if textContent == "" && result.Response != nil {
					textContent = extractResponseText(result.Response)
				}
				if textContent != "" {
					assistantMsg.Content = textContent
				}
				exec.rs.conv.appendOwned(assistantMsg)
				if err := exec.checkpoint(ctx); err != nil {
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
					endSpans(err)
					return
				}

				transferAck := fmt.Sprintf("Transferred control to %s.", result.Transfer.Request.AgentName)
				if err := r.sendStreamingEvent(ctx, out, streaming.Event{
					Kind: streaming.EventToolResult,
					Result: &streaming.ToolResultDelta{
						CallID:  result.Transfer.Call.ID,
						Name:    result.Transfer.Call.Name,
						Content: transferAck,
					},
				}); err != nil {
					endSpans(err)
					return
				}
				exec.rs.conv.appendOwned(llm.Message{
					Role:       "tool",
					Content:    transferAck,
					ToolCallID: result.Transfer.Call.ID,
					Name:       result.Transfer.Call.Name,
				})
				if err := exec.checkpoint(ctx); err != nil {
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
					endSpans(err)
					return
				}
				if err := exec.applyTransfer(result.Transfer.Request); err != nil {
					err = fmt.Errorf("runtime: transfer: %w", err)
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
					endSpans(err)
					return
				}
				if r.tracer != nil && runSpan != nil {
					fromAgent := exec.rs.activeAgent.Name()
					r.tracer.AddEvent(ctx, runSpan, "gen_ai.agent.transfer", map[string]string{
						"gen_ai.handoff.from_agent": fromAgent,
						"gen_ai.handoff.to_agent":   result.Transfer.Request.AgentName,
						"content":                  result.Transfer.Request.Input,
					})
					if agentSpan != runSpan {
						r.tracer.EndSpan(ctx, agentSpan, nil)
					}
					targetName := result.Transfer.Request.AgentName
					agentSpanAttrs := map[string]string{
						"gen_ai.operation.name":  "invoke_agent",
						"gen_ai.conversation.id": exec.rs.sessionID,
						"gen_ai.agent.name":      targetName,
						"session.id":             exec.rs.sessionID,
					}
					agentCtx, newAgentSpan := r.tracer.StartSpan(ctx, "invoke_agent "+targetName, agentSpanAttrs)
					agentSpan = newAgentSpan
					ctx = agentCtx
					transferInput := result.Transfer.Request.Input
					if transferInput == "" {
						transferInput = result.StreamedText
						if transferInput == "" && result.Response != nil {
							transferInput = extractResponseText(result.Response)
						}
					}
					if transferInput == "" {
						transferInput = fromAgent + " requested transfer to " + targetName
					}
					transferMsg := map[string]any{
						"role": "user",
						"parts": []map[string]any{
							{"type": "text", "content": transferInput},
						},
					}
					transferJSON, _ := json.Marshal([]map[string]any{transferMsg})
					r.tracer.SetSpanAttributes(ctx, agentSpan, map[string]any{
						"gen_ai.operation.name":      "invoke_agent",
						"gen_ai.input.messages":      string(transferJSON),
						"langfuse.observation.input": string(transferJSON),
					})
				}
				continue
			}

			if len(result.ToolCalls) == 0 {
				text := result.StreamedText
				if text == "" && result.Response != nil {
					text = extractResponseText(result.Response)
				}
				exec.rs.conv.appendOwned(llm.Message{Role: "assistant", Content: text})
				if err := exec.compactConversation(ctx); err != nil {
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: fmt.Errorf("runtime: compact conversation: %w", err)})
					endSpans(err)
					return
				}
				if err := exec.checkpoint(ctx); err != nil {
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
					endSpans(err)
					return
				}
				if err := exec.saveSessionData(ctx); err != nil {
					_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
					endSpans(err)
					return
				}
				r.publishRunState(exec.rs)
				if err := r.sendStreamingEvent(ctx, out, streaming.Event{
					Kind:  streaming.EventFinalText,
					Final: &streaming.FinalTextDelta{Content: text},
				}); err != nil {
					endSpans(err)
					return
				}
				if r.tracer != nil && runSpan != nil {
					outputMsg := map[string]any{
						"role":          "assistant",
						"finish_reason": "",
						"parts": []map[string]any{
							{"type": "text", "content": text},
						},
					}
					outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
					r.tracer.AddEvent(ctx, runSpan, "gen_ai.assistant.message", map[string]string{
						"role":                     "assistant",
						"content":                 text,
						"gen_ai.output.messages":  string(outputJSON),
					})
					r.tracer.SetSpanAttributes(ctx, runSpan, map[string]any{
						"gen_ai.usage.input_tokens":    result.TokensUsed.InputTokens,
						"gen_ai.usage.output_tokens":   result.TokensUsed.OutputTokens,
						"gen_ai.output.messages":      string(outputJSON),
						"langfuse.observation.output":  text,
					})
					if agentSpan != runSpan {
						r.tracer.SetSpanAttributes(ctx, agentSpan, map[string]any{
							"gen_ai.output.messages":      string(outputJSON),
							"langfuse.observation.output": text,
						})
					}
					endSpans(nil)
				}
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventDone})
				return
			}

			assistantMsg := llm.Message{Role: "assistant"}
			llmToolCalls := make([]llm.ToolCall, len(result.ToolCalls))
			for i, tc := range result.ToolCalls {
				llmToolCalls[i] = llm.ToolCall{ID: tc.ID, Name: tc.Name, Input: cloneMap(tc.Input)}
			}
			assistantMsg.ToolCalls = llmToolCalls
			textContent := result.StreamedText
			if textContent == "" && result.Response != nil {
				textContent = extractResponseText(result.Response)
			}
			if textContent != "" {
				assistantMsg.Content = textContent
			}
			exec.rs.conv.appendOwned(assistantMsg)
			if err := exec.checkpoint(ctx); err != nil {
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
				endSpans(err)
				return
			}

			for _, tr := range exec.dispatchWithTracing(ctx, result.ToolCalls) {
				content := tools.ResultToString(tr)
				if err := r.sendStreamingEvent(ctx, out, streaming.Event{
					Kind: streaming.EventToolResult,
					Result: &streaming.ToolResultDelta{
						CallID:  tr.CallID,
						Name:    tr.Name,
						Content: content,
					},
				}); err != nil {
					endSpans(err)
					return
				}
				exec.rs.conv.appendOwned(llm.Message{
					Role:       "tool",
					Content:    content,
					ToolCallID: tr.CallID,
					Name:       tr.Name,
				})
			}
			if err := exec.checkpoint(ctx); err != nil {
				_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
				endSpans(err)
				return
			}
		}

		err := fmt.Errorf("runtime: max steps (%d) exceeded", r.maxSteps)
		_ = r.sendStreamingEvent(ctx, out, streaming.Event{Kind: streaming.EventError, Err: err})
		endSpans(err)
	}()

	return out, nil
}

func (r *Runtime) SessionSnapshot() SessionSnapshot {
	r.mu.RLock()
	session := r.session
	r.mu.RUnlock()
	if session == nil {
		return SessionSnapshot{Metadata: map[string]string{}}
	}
	return session.Snapshot()
}

func (r *Runtime) ConversationSnapshot() ConversationState {
	r.mu.RLock()
	conv := r.conv
	r.mu.RUnlock()
	if conv == nil {
		return ConversationState{}
	}
	return conv.Snapshot()
}

func (r *Runtime) ConversationMessages() []llm.Message {
	return r.ConversationSnapshot().Messages
}

func (r *Runtime) ConversationSlice(start, end int) []llm.Message {
	r.mu.RLock()
	conv := r.conv
	r.mu.RUnlock()
	return conv.Slice(start, end)
}

func (r *Runtime) AppendConversationMessage(msg llm.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.Append(msg)
}

func (r *Runtime) AppendConversationMessages(msgs []llm.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.AppendAll(msgs)
}

func (r *Runtime) ClearConversation() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.Clear()
}

func (r *Runtime) SetConversationSystem(content string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.SetSystem(content)
}

func (r *Runtime) SessionStoreData(userID, agentID string) *store.SessionData {
	snap := r.SessionSnapshot()
	configJSON, _ := json.Marshal(snap.Config)
	budgetJSON, _ := json.Marshal(snap.Budget)
	return &store.SessionData{
		ID:       snap.ID,
		UserID:   userID,
		AgentID:  agentID,
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: cloneStringMap(snap.Metadata),
	}
}

func (r *Runtime) SetSessionMetadata(key, value string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.session == nil {
		return
	}
	r.generation++
	r.session.SetMeta(key, value)
}

func (r *Runtime) LoadState(sessionSnap SessionSnapshot, convState ConversationState) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.generation++
	r.session = NewSessionFromSnapshot(sessionSnap)
	r.conv = NewConversationFromState(convState)
}

func (r *Runtime) newRunExecutor() *runExecutor {
	return &runExecutor{
		rt: r,
		rs: r.captureRunState(),
	}
}

func (r *Runtime) captureRunState() *runState {
	r.mu.RLock()
	generation := r.generation
	session := r.session
	conv := r.conv
	userID := r.userID
	agentID := r.agentID
	activeAgent := r.agent
	rootAgent := r.rootAgent
	toolRegistry := r.tools
	r.mu.RUnlock()

	sessionSnap := session.Snapshot()
	convState := conv.Snapshot()

	return &runState{
		generation:  generation,
		sessionID:   sessionSnap.ID,
		runID:       uuid.NewString(),
		config:      cloneSessionConfig(sessionSnap.Config),
		metadata:    cloneStringMap(sessionSnap.Metadata),
		budget:      streaming.NewBudgetFromSnapshot(sessionSnap.Budget),
		conv:        NewConversationFromState(convState),
		userID:      userID,
		agentID:     agentID,
		activeAgent: activeAgent,
		rootAgent:   rootAgent,
		tools:       toolRegistry,
	}
}

func (r *Runtime) publishRunState(rs *runState) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.generation != rs.generation {
		return
	}

	if r.session != nil {
		r.session.Restore(SessionSnapshot{
			ID:       rs.sessionID,
			Config:   cloneSessionConfig(rs.config),
			Budget:   rs.budget.Snapshot(),
			Metadata: cloneStringMap(rs.metadata),
		})
	}
	if r.conv != nil {
		r.conv.Restore(rs.conv.Snapshot())
	}
	r.agent = rs.activeAgent
	r.agentID = rs.agentID
	r.tools = rs.tools
	r.dispatcher = tools.NewDispatcher(r.tools)
}

func (e *runExecutor) buildRequest(stepMessages []llm.Message, stepTools []tools.ToolDef, stepIndex int) *llm.Request {
	return &llm.Request{
		Model:       e.rs.config.Model,
		Messages:    cloneMessages(stepMessages),
		Tools:       toLLMToolDefs(stepTools),
		MaxTokens:   e.rs.config.MaxTokens,
		Temperature: e.rs.config.Temperature,
		Execution: &llm.ExecutionContext{
			SessionID: e.rs.sessionID,
			UserID:    e.rs.userID,
			AgentID:   e.rs.agentID,
			RunID:     e.rs.runID,
			StepIndex: stepIndex,
			Metadata:  cloneStringMap(e.rs.metadata),
		},
	}
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
		inputMessages := messagesToOTelFormat(req.Messages)
		inputJSON, _ := json.Marshal(inputMessages)
		spanAttrs := map[string]string{
			"gen_ai.operation.name":    "chat",
			"gen_ai.provider.name":     otelProviderName(e.rt.provider.Name()),
			"gen_ai.request.model":     req.Model,
			"gen_ai.conversation.id":   step.SessionID,
			"session.id":              step.SessionID,
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
			"gen_ai.request.stream":     false,
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
			msgJSON, _ := json.Marshal(msg)
			content := extractContentFromOTelMsg(msg)
			eventAttrs := map[string]string{
				"role":                  role,
				"gen_ai.input.messages": string(msgJSON),
			}
			if content != "" {
				eventAttrs["content"] = content
			}
			e.rt.tracer.AddEvent(ctx, llmSpan, eventName, eventAttrs)
		}
	}

	resp, err := e.rt.provider.Complete(ctx, req)
	if err != nil {
		e.rt.logger.Error().Err(err).Msg("llm.error")
		if e.rt.tracer != nil && llmSpan != nil {
			e.rt.tracer.EndSpan(ctx, llmSpan, err)
		}
		return nil, fmt.Errorf("runtime: provider complete: %w", err)
	}

e.rt.logger.Debug().Interface("response", resp).Msg("llm.response")
	if e.rt.tracer != nil && llmSpan != nil {
		responseAttrs := map[string]any{
			"gen_ai.response.finish_reasons": []string{resp.FinishReason},
			"gen_ai.response.model":           req.Model,
			"gen_ai.usage.input_tokens":       resp.Usage.InputTokens,
			"gen_ai.usage.output_tokens":      resp.Usage.OutputTokens,
		}
		if text := extractResponseText(resp); text != "" {
			outputMsg := map[string]any{
				"role":          "assistant",
				"finish_reason": resp.FinishReason,
				"parts": []map[string]any{
					{"type": "text", "content": text},
				},
			}
			outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
			responseAttrs["gen_ai.output.messages"] = string(outputJSON)
			responseAttrs["langfuse.observation.output"] = text
			e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
				"role":                     "assistant",
				"content":                 text,
				"gen_ai.output.messages":  string(outputJSON),
			})
		} else {
			outputParts := []map[string]any{}
			for _, block := range resp.Content {
				if block.Type == "tool_call" && block.ToolCall != nil {
					outputParts = append(outputParts, map[string]any{
						"type": "tool_call",
						"tool_call": map[string]any{
							"id":   block.ToolCall.ID,
							"name": block.ToolCall.Name,
						},
					})
				}
			}
			if len(outputParts) == 0 {
				outputParts = append(outputParts, map[string]any{
					"type": "text", "content": "",
				})
			}
			outputMsg := map[string]any{
				"role":          "assistant",
				"finish_reason": resp.FinishReason,
				"parts":         outputParts,
			}
			outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
			responseAttrs["gen_ai.output.messages"] = string(outputJSON)
			var toolNames []string
			for _, block := range resp.Content {
				if block.Type == "tool_call" && block.ToolCall != nil {
					toolNames = append(toolNames, block.ToolCall.Name)
				}
			}
			outputSummary := "tool_calls: " + strings.Join(toolNames, ", ")
			responseAttrs["langfuse.observation.output"] = outputSummary
			e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
				"role":                     "assistant",
				"gen_ai.output.messages":  string(outputJSON),
				"content":                 outputSummary,
			})
		}
		e.rt.tracer.SetSpanAttributes(ctx, llmSpan, responseAttrs)
		e.rt.tracer.EndSpan(ctx, llmSpan, nil)
	}

	var toolCalls []tools.ToolCall
	for _, block := range resp.Content {
		if block.Type == "tool_call" && block.ToolCall != nil {
			toolCalls = append(toolCalls, tools.ToolCall{
				ID:    block.ToolCall.ID,
				Name:  block.ToolCall.Name,
				Input: cloneMap(block.ToolCall.Input),
			})
		}
	}

	transferStep, err := e.extractTransfer(toolCalls, extractResponseText(resp))
	if err != nil {
		return nil, err
	}
	if transferStep != nil {
		return &StepResult{
			Response:   resp,
			Transfer:   transferStep,
			TokensUsed: resp.Usage,
		}, nil
	}

	return &StepResult{
		Response:   resp,
		ToolCalls:  toolCalls,
		TokensUsed: resp.Usage,
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
		inputMessages := messagesToOTelFormat(req.Messages)
		inputJSON, _ := json.Marshal(inputMessages)
		spanAttrs := map[string]string{
			"gen_ai.operation.name":    "chat",
			"gen_ai.provider.name":     otelProviderName(e.rt.provider.Name()),
			"gen_ai.request.model":     req.Model,
			"gen_ai.conversation.id":   step.SessionID,
			"session.id":              step.SessionID,
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
			"gen_ai.request.stream":     true,
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
			msgJSON, _ := json.Marshal(msg)
			content := extractContentFromOTelMsg(msg)
			eventAttrs := map[string]string{
				"role":                  role,
				"gen_ai.input.messages": string(msgJSON),
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
	assembler := newToolCallAssembler()
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
				toolCalls, asmErr := assembler.assemble()
				if asmErr != nil {
					return nil, asmErr
				}
				transferStep, transferErr := e.extractTransfer(toolCalls, textBuilder.String())
				if transferErr != nil {
					return nil, transferErr
				}
				return &StreamingStepResult{
					Response: &llm.Response{
						Content:      nil,
						Usage:        usage,
						FinishReason: responseFinishReason,
					},
					ToolCalls:    toolCalls,
					Transfer:     transferStep,
					TokensUsed:   usage,
					StreamedText: textBuilder.String(),
				}, nil
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
					assembler.addFragment(event.Tool.Index, event.Tool)
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
				toolCalls, asmErr := assembler.assemble()
				if asmErr != nil {
					return nil, asmErr
				}
				if e.rt.tracer != nil && llmSpan != nil {
					responseAttrs := map[string]any{
						"gen_ai.response.finish_reasons": []string{responseFinishReason},
						"gen_ai.response.model":           req.Model,
						"gen_ai.usage.input_tokens":       usage.InputTokens,
						"gen_ai.usage.output_tokens":      usage.OutputTokens,
					}
					if text := textBuilder.String(); text != "" {
						outputMsg := map[string]any{
							"role":          "assistant",
							"finish_reason": responseFinishReason,
							"parts": []map[string]any{
								{"type": "text", "content": text},
							},
						}
						outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
						responseAttrs["gen_ai.output.messages"] = string(outputJSON)
						responseAttrs["langfuse.observation.output"] = text
						e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
							"role":                     "assistant",
							"content":                 text,
							"gen_ai.output.messages":  string(outputJSON),
						})
					} else if len(toolCalls) > 0 {
						outputParts := []map[string]any{}
						for _, tc := range toolCalls {
							outputParts = append(outputParts, map[string]any{
								"type": "tool_call",
								"tool_call": map[string]any{
									"id":   tc.ID,
									"name": tc.Name,
								},
							})
						}
						outputMsg := map[string]any{
							"role":          "assistant",
							"finish_reason": responseFinishReason,
							"parts":          outputParts,
						}
						outputJSON, _ := json.Marshal([]map[string]any{outputMsg})
						responseAttrs["gen_ai.output.messages"] = string(outputJSON)
						var toolNames []string
						for _, tc := range toolCalls {
							toolNames = append(toolNames, tc.Name)
						}
						outputSummary := "tool_calls: " + strings.Join(toolNames, ", ")
						responseAttrs["langfuse.observation.output"] = outputSummary
						e.rt.tracer.AddEvent(ctx, llmSpan, "gen_ai.assistant.message", map[string]string{
							"role":                     "assistant",
							"gen_ai.output.messages":  string(outputJSON),
							"content":                 outputSummary,
						})
					}
					e.rt.tracer.SetSpanAttributes(ctx, llmSpan, responseAttrs)
					e.rt.tracer.EndSpan(ctx, llmSpan, nil)
				}

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

func (e *runExecutor) compactConversation(ctx context.Context) error {
	if e.rt.compactor == nil {
		return nil
	}
	result, err := e.rt.compactor.Compact(ctx, compaction.Input{
		SessionID: e.rs.sessionID,
		Messages:  e.rs.conv.messagesOwned(),
		Tools:     toLLMToolDefs(e.availableToolDefs()),
	})
	if err != nil {
		return err
	}
	if result.Compacted {
		e.rs.conv.replaceMessagesOwned(result.Messages)
	}
	return nil
}

func (e *runExecutor) prepareMessagesForRequest(ctx context.Context, sessionID string, messages []llm.Message, stepTools []tools.ToolDef) ([]llm.Message, error) {
	current := cloneMessages(messages)
	if !e.shouldForcePreflightCompaction(current, stepTools) {
		return current, nil
	}
	if e.rt.compactor == nil {
		return nil, fmt.Errorf("request exceeds model context limit and no compactor is configured")
	}

	result, err := e.rt.compactor.ForceCompact(ctx, compaction.Input{
		SessionID: sessionID,
		Messages:  current,
		Tools:     toLLMToolDefs(stepTools),
	})
	if err != nil {
		return nil, err
	}
	if result.Compacted {
		e.rs.conv.replaceMessagesOwned(result.Messages)
		current = cloneMessages(result.Messages)
	}
	if e.shouldForcePreflightCompaction(current, stepTools) {
		return nil, fmt.Errorf("request still exceeds model context limit after compaction")
	}
	return current, nil
}

func (e *runExecutor) shouldForcePreflightCompaction(messages []llm.Message, stepTools []tools.ToolDef) bool {
	if e.rt.modelContextLimit <= 0 {
		return false
	}
	return compaction.EstimatePromptTokens(messages, toLLMToolDefs(stepTools), nil)+e.rt.outputReserve > e.rt.modelContextLimit
}

func (e *runExecutor) dispatchWithTracing(ctx context.Context, calls []tools.ToolCall) []tools.ToolResult {
	return tools.NewDispatcher(e.rs.tools).DispatchAllWith(ctx, calls, e.rt.maxToolConcurrency, e.dispatchOneWithTracing)
}

func (e *runExecutor) dispatchOneWithTracing(ctx context.Context, call tools.ToolCall) tools.ToolResult {
	var toolSpan observability.Span
	if e.rt.tracer != nil {
		spanAttrs := map[string]string{
			"gen_ai.operation.name":    "execute_tool",
			"gen_ai.provider.name":     otelProviderName(e.rt.provider.Name()),
			"gen_ai.tool.name":         call.Name,
			"gen_ai.tool.call_id":     call.ID,
			"gen_ai.conversation.id":  e.rs.sessionID,
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
		inputJSON, _ := json.Marshal(call.Input)
		e.rt.tracer.SetSpanAttributes(ctx, toolSpan, map[string]any{
			"gen_ai.operation.name":    "execute_tool",
			"gen_ai.tool.input":       string(inputJSON),
			"langfuse.observation.input": string(inputJSON),
		})
		e.rt.tracer.AddEvent(ctx, toolSpan, "gen_ai.tool.message", map[string]string{
			"role":              "tool",
			"content":           string(inputJSON),
			"gen_ai.tool.input": string(inputJSON),
		})
	}

	result := e.dispatchToolCall(ctx, call)

	if e.rt.tracer != nil && toolSpan != nil {
		if result.Err != nil {
			e.rt.tracer.AddEvent(ctx, toolSpan, "gen_ai.tool.message", map[string]string{
				"role":               "tool",
				"content":            result.Err.Error(),
				"gen_ai.tool.output": result.Err.Error(),
			})
			e.rt.tracer.SetSpanAttributes(ctx, toolSpan, map[string]any{
				"gen_ai.tool.output":        result.Err.Error(),
				"langfuse.observation.output": result.Err.Error(),
				"error.type":                "tool_error",
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
				"gen_ai.tool.output":        outputContent,
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

func (e *runExecutor) availableToolDefs() []tools.ToolDef {
	var defs []tools.ToolDef
	if e.rs.tools != nil {
		defs = append(defs, e.rs.tools.All()...)
	}
	defs = append(defs, e.consultToolDefs()...)
	defs = append(defs, e.transferToolDefs()...)
	return defs
}

func (e *runExecutor) consultToolDefs() []tools.ToolDef {
	if e.rt.subagentResolver == nil || e.rs.activeAgent == nil {
		return nil
	}
	subagents := e.rs.activeAgent.Subagents()
	if len(subagents) == 0 {
		return nil
	}

	defs := make([]tools.ToolDef, 0, len(subagents))
	for _, name := range subagents {
		if _, ok := e.rt.subagentResolver.Get(name); !ok {
			continue
		}
		defs = append(defs, tools.ToolDef{
			Name:        consultToolName(name),
			Description: fmt.Sprintf("Consult subagent %q and return its result to the current agent.", name),
			Schema:      consultToolSchema,
		})
	}
	return defs
}

func (e *runExecutor) prepareConsultRuntime(req ConsultRequest) (*Runtime, error) {
	if err := ensureSubagentAllowedFor(e.rs.activeAgent, req.AgentName); err != nil {
		return nil, err
	}
	childAgent, err := resolveSubagent(e.rt.subagentResolver, req.AgentName)
	if err != nil {
		return nil, err
	}

	child := NewRuntime(e.rt.inheritedSubagentRuntimeConfig(childAgent))
	childSnap := child.SessionSnapshot()
	childSnap.ID = e.rs.sessionID
	childSnap.Metadata = cloneStringMap(req.Metadata)
	child.LoadState(childSnap, ConversationState{
		Messages: withAgentSystemPrompt(req.Context, childAgent.SessionConfig().SystemPrompt),
	})
	return child, nil
}

func (e *runExecutor) consultTargetForTool(toolName string) (string, bool) {
	if !strings.HasPrefix(toolName, consultToolPrefix) || e.rs.activeAgent == nil {
		return "", false
	}
	target := strings.TrimPrefix(toolName, consultToolPrefix)
	if target == "" || !e.rs.activeAgent.HasSubagent(target) {
		return "", false
	}
	if e.rt.subagentResolver == nil {
		return "", false
	}
	if _, ok := e.rt.subagentResolver.Get(target); !ok {
		return "", false
	}
	return target, true
}

func (e *runExecutor) transferToolDefs() []tools.ToolDef {
	if e.rs.activeAgent == nil {
		return nil
	}
	names := e.transferTargetNames()
	defs := make([]tools.ToolDef, 0, len(names))
	for _, name := range names {
		defs = append(defs, tools.ToolDef{
			Name:        transferToolName(name),
			Description: fmt.Sprintf("Transfer control to subagent %q.", name),
			Schema:      transferToolSchema,
		})
	}
	return defs
}

func (e *runExecutor) extractTransfer(calls []tools.ToolCall, fallbackInput string) (*TransferStep, error) {
	if len(calls) == 0 {
		return nil, nil
	}

	var transferCall *tools.ToolCall
	for i := range calls {
		if !strings.HasPrefix(calls[i].Name, transferToolPrefix) {
			continue
		}
		target, ok := e.transferTargetForTool(calls[i].Name)
		if !ok {
			agentName := e.rs.agentID
			if e.rs.activeAgent != nil && e.rs.activeAgent.Name() != "" {
				agentName = e.rs.activeAgent.Name()
			}
			return nil, fmt.Errorf("runtime: transfer target %q is not available to agent %q", strings.TrimPrefix(calls[i].Name, transferToolPrefix), agentName)
		}
		if len(calls) > 1 {
			return nil, fmt.Errorf("runtime: transfer tool %q cannot be combined with other tool calls", calls[i].Name)
		}
		req, err := parseTransferRequest(target, calls[i].Input, fallbackInput)
		if err != nil {
			return nil, err
		}
		call := calls[i]
		transferCall = &call
		return &TransferStep{Call: call, Request: req}, nil
	}

	if transferCall != nil {
		return &TransferStep{Call: *transferCall}, nil
	}
	return nil, nil
}

func (e *runExecutor) transferTargetForTool(toolName string) (string, bool) {
	if !strings.HasPrefix(toolName, transferToolPrefix) || e.rs.activeAgent == nil {
		return "", false
	}
	target := strings.TrimPrefix(toolName, transferToolPrefix)
	if target == "" || !isTransferAllowedFor(e.rs.activeAgent, e.rs.rootAgent, target) {
		return "", false
	}
	if _, err := resolveTransferAgent(e.rt.subagentResolver, e.rs.rootAgent, target); err != nil {
		return "", false
	}
	return target, true
}

func (e *runExecutor) transferTargetNames() []string {
	seen := map[string]struct{}{}
	var names []string
	add := func(name string) {
		if name == "" {
			return
		}
		if _, err := resolveTransferAgent(e.rt.subagentResolver, e.rs.rootAgent, name); err != nil {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	for _, name := range e.rs.activeAgent.Subagents() {
		add(name)
	}
	if rootName := transferRootNameFor(e.rs.activeAgent, e.rs.rootAgent); rootName != "" {
		add(rootName)
		for _, name := range e.rs.rootAgent.Subagents() {
			if name == e.rs.activeAgent.Name() {
				continue
			}
			add(name)
		}
	}
	return names
}

func (e *runExecutor) applyTransfer(req TransferRequest) error {
	targetAgent, sessionSnap, convState, toolRegistry, err := resolveTransferStateFrom(
		e.rs.activeAgent,
		SessionSnapshot{
			ID:       e.rs.sessionID,
			Config:   cloneSessionConfig(e.rs.config),
			Budget:   e.rs.budget.Snapshot(),
			Metadata: cloneStringMap(e.rs.metadata),
		},
		ConversationState{Messages: e.rs.conv.messagesOwned()},
		req,
		e.rt.subagentResolver,
		e.rs.rootAgent,
		e.rt.resolveTransferInputFilter(req.Filter),
	)
	if err != nil {
		return err
	}

	e.rs.config = cloneSessionConfig(sessionSnap.Config)
	e.rs.metadata = cloneStringMap(sessionSnap.Metadata)
	e.rs.budget = streaming.NewBudgetFromSnapshot(sessionSnap.Budget)
	e.rs.conv = NewConversationFromState(convState)
	if strings.TrimSpace(req.Input) != "" {
		e.rs.conv.appendOwned(llm.Message{Role: "user", Content: req.Input})
	}
	e.rs.activeAgent = targetAgent
	e.rs.agentID = targetAgent.Name()
	e.rs.tools = toolRegistry
	return nil
}

func (e *runExecutor) checkpoint(ctx context.Context) error {
	if e.rt.conversationStore == nil {
		return nil
	}
	if err := e.rt.conversationStore.Save(ctx, e.rs.sessionID, e.rs.conv.messagesOwned()); err != nil {
		e.rt.logger.Error().Err(err).Str("session_id", e.rs.sessionID).Msg("conversation checkpoint failed")
		return fmt.Errorf("runtime: checkpoint conversation: %w", err)
	}
	return nil
}

func (e *runExecutor) saveSessionData(ctx context.Context) error {
	if e.rt.sessionStore == nil {
		return nil
	}
	data := e.sessionStoreData()
	if err := e.rt.sessionStore.SaveSession(ctx, data); err != nil {
		e.rt.logger.Error().Err(err).Str("session_id", e.rs.sessionID).Msg("session save failed")
		return fmt.Errorf("runtime: save session: %w", err)
	}
	return nil
}

func (e *runExecutor) sessionStoreData() *store.SessionData {
	configJSON, _ := json.Marshal(cloneSessionConfig(e.rs.config))
	budgetJSON, _ := json.Marshal(e.rs.budget.Snapshot())
	return &store.SessionData{
		ID:       e.rs.sessionID,
		UserID:   e.rs.userID,
		AgentID:  e.rs.agentID,
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: cloneStringMap(e.rs.metadata),
	}
}
