package agent

import (
	"context"
	"fmt"
	"strings"

	agentsubagent "github.com/Kaelancode/kaeAgent-Public/agent/internal/subagent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

const ActiveAgentMetadataKey = "active_agent"
const TransferReasonMetadataKey = agentsubagent.TransferReasonMetadataKey
const ConsultReasonMetadataKey = agentsubagent.ConsultReasonMetadataKey

type SubagentResolver interface {
	Get(name string) (*Agent, bool)
}

type ConsultRequest struct {
	AgentName string
	Input     string
	Context   []llm.Message
	Metadata  map[string]string
}

type TransferRequest struct {
	AgentName string
	Input     string
	Context   []llm.Message
	Metadata  map[string]string
	Filter    TransferInputFilter
}

type TransferStep struct {
	Call    tools.ToolCall
	Request TransferRequest
}

const consultToolPrefix = agentsubagent.ConsultToolPrefix
const transferToolPrefix = agentsubagent.TransferToolPrefix

var consultToolSchema = agentsubagent.ConsultToolSchema
var transferToolSchema = agentsubagent.TransferToolSchema

func (r *Runtime) Consult(ctx context.Context, resolver SubagentResolver, req ConsultRequest) (string, error) {
	child, err := r.prepareConsultRuntime(resolver, req)
	if err != nil {
		return "", err
	}

	result, err := child.Run(ctx, req.Input)
	if err != nil {
		return "", fmt.Errorf("runtime: consult %q: %w", req.AgentName, err)
	}
	return result, nil
}

func (r *Runtime) ConsultStream(ctx context.Context, resolver SubagentResolver, req ConsultRequest) (<-chan streaming.Event, error) {
	child, err := r.prepareConsultRuntime(resolver, req)
	if err != nil {
		return nil, err
	}

	ch, err := child.Stream(ctx, req.Input)
	if err != nil {
		return nil, fmt.Errorf("runtime: consult stream %q: %w", req.AgentName, err)
	}
	return ch, nil
}

func (r *Runtime) Transfer(ctx context.Context, resolver SubagentResolver, req TransferRequest) (string, error) {
	if err := r.ApplyTransfer(resolver, req); err != nil {
		return "", err
	}

	result, err := r.Run(ctx, req.Input)
	if err != nil {
		return "", fmt.Errorf("runtime: transfer %q: %w", req.AgentName, err)
	}

	return result, nil
}

func (r *Runtime) TransferStream(ctx context.Context, resolver SubagentResolver, req TransferRequest) (<-chan streaming.Event, error) {
	child, generation, err := r.prepareTransferStreamRuntime(resolver, req)
	if err != nil {
		return nil, err
	}

	ch, err := child.Stream(ctx, req.Input)
	if err != nil {
		return nil, fmt.Errorf("runtime: transfer stream %q: %w", req.AgentName, err)
	}

	return r.adoptTransferStreamOnDone(ch, child, generation), nil
}

func (r *Runtime) ActiveAgent(resolver SubagentResolver, rootAgent string) string {
	snap := r.SessionSnapshot()
	name := snap.Metadata[ActiveAgentMetadataKey]
	if name == "" {
		return rootAgent
	}
	if resolver == nil {
		return rootAgent
	}
	if _, ok := resolver.Get(name); !ok {
		return rootAgent
	}
	return name
}

func (r *Runtime) prepareConsultRuntime(resolver SubagentResolver, req ConsultRequest) (*Runtime, error) {
	if err := r.ensureSubagentAllowed(req.AgentName); err != nil {
		return nil, err
	}
	childAgent, err := resolveSubagent(resolver, req.AgentName)
	if err != nil {
		return nil, err
	}

	child := NewRuntime(r.inheritedSubagentRuntimeConfig(childAgent))
	childSnap := child.SessionSnapshot()
	parentSnap := r.SessionSnapshot()
	childSnap.ID = parentSnap.ID
	childSnap.Metadata = cloneStringMap(req.Metadata)
	child.LoadState(childSnap, ConversationState{
		Messages: withAgentSystemPrompt(req.Context, childAgent.SessionConfig().SystemPrompt),
	})
	return child, nil
}

func (r *Runtime) ApplyTransfer(resolver SubagentResolver, req TransferRequest) error {
	targetAgent, sessionSnap, convState, toolRegistry, err := r.resolveTransferState(resolver, req)
	if err != nil {
		return err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.generation++
	r.agent = targetAgent
	r.agentID = targetAgent.Name()
	if r.session != nil {
		r.session.Restore(sessionSnap)
	} else {
		r.session = NewSessionFromSnapshot(sessionSnap)
	}
	if r.conv != nil {
		r.conv.Restore(convState)
	} else {
		r.conv = NewConversationFromState(convState)
	}
	r.tools = toolRegistry
	r.dispatcher = tools.NewDispatcher(r.tools)
	return nil
}

func (r *Runtime) resolveTransferInputFilter(override TransferInputFilter) TransferInputFilter {
	if override != nil {
		return override
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.transferInputFilter
}

func (r *Runtime) prepareTransferStreamRuntime(resolver SubagentResolver, req TransferRequest) (*Runtime, uint64, error) {
	r.mu.RLock()
	generation := r.generation
	provider := r.provider
	rootAgent := r.rootAgent
	subagentResolver := r.subagentResolver
	maxToolConcurrency := r.maxToolConcurrency
	middleware := append([]Middleware(nil), r.middleware...)
	streamMiddleware := append([]StreamingMiddleware(nil), r.streamMiddleware...)
	maxSteps := r.maxSteps
	transferInputFilter := r.transferInputFilter
	tracer := r.tracer
	conversationStore := r.conversationStore
	sessionStore := r.sessionStore
	compactor := r.compactor
	modelContextLimit := r.modelContextLimit
	outputReserve := r.outputReserve
	userID := r.userID
	logger := r.logger
	r.mu.RUnlock()

	targetAgent, sessionSnap, convState, _, err := r.resolveTransferState(resolver, req)
	if err != nil {
		return nil, 0, err
	}

	child := NewRuntime(RuntimeConfig{
		Provider:            provider,
		Agent:               targetAgent,
		RootAgent:           rootAgent,
		SubagentResolver:    subagentResolver,
		Session:             NewSessionFromSnapshot(sessionSnap),
		Conversation:        NewConversationFromState(convState),
		MaxToolConcurrency:  maxToolConcurrency,
		Middleware:          middleware,
		StreamingMiddleware: streamMiddleware,
		MaxSteps:            maxSteps,
		TransferInputFilter: transferInputFilter,
		Tracer:              tracer,
		ConversationStore:   conversationStore,
		SessionStore:        sessionStore,
		Compactor:           compactor,
		ModelContextLimit:   modelContextLimit,
		OutputTokenReserve:  outputReserve,
		UserID:              userID,
		AgentID:             targetAgent.Name(),
		Logger:              logger,
	})
	return child, generation, nil
}

func (r *Runtime) adoptTransferStreamOnDone(ch <-chan streaming.Event, child *Runtime, generation uint64) <-chan streaming.Event {
	out := make(chan streaming.Event)
	go func() {
		defer close(out)
		for event := range ch {
			if event.Kind == streaming.EventDone {
				r.adoptRuntimeState(child, generation)
			}
			out <- event
		}
	}()
	return out
}

func (r *Runtime) adoptRuntimeState(child *Runtime, generation uint64) {
	if child == nil {
		return
	}

	child.mu.RLock()
	childAgent := child.agent
	childAgentID := child.agentID
	childTools := child.tools
	child.mu.RUnlock()

	sessionSnap := child.SessionSnapshot()
	convState := child.ConversationSnapshot()

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.generation != generation {
		return
	}
	if r.session != nil {
		r.session.Restore(sessionSnap)
	} else {
		r.session = NewSessionFromSnapshot(sessionSnap)
	}
	if r.conv != nil {
		r.conv.Restore(convState)
	} else {
		r.conv = NewConversationFromState(convState)
	}
	r.agent = childAgent
	r.agentID = childAgentID
	r.tools = childTools
	r.dispatcher = tools.NewDispatcher(r.tools)
}

func (r *Runtime) resolveTransferState(resolver SubagentResolver, req TransferRequest) (*Agent, SessionSnapshot, ConversationState, *tools.Registry, error) {
	r.mu.RLock()
	caller := r.agent
	rootAgent := r.rootAgent
	r.mu.RUnlock()
	return resolveTransferStateFrom(
		caller,
		r.SessionSnapshot(),
		r.ConversationSnapshot(),
		req,
		resolver,
		rootAgent,
		r.resolveTransferInputFilter(req.Filter),
	)
}

func (r *Runtime) inheritedSubagentRuntimeConfig(agentDef *Agent) RuntimeConfig {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return RuntimeConfig{
		Provider:            r.provider,
		Agent:               agentDef,
		RootAgent:           r.rootAgent,
		SubagentResolver:    r.subagentResolver,
		MaxToolConcurrency:  r.maxToolConcurrency,
		Middleware:          append([]Middleware(nil), r.middleware...),
		StreamingMiddleware: append([]StreamingMiddleware(nil), r.streamMiddleware...),
		MaxSteps:            r.maxSteps,
		TransferInputFilter: r.transferInputFilter,
		Tracer:              r.tracer,
		Compactor:           r.compactor,
		ModelContextLimit:   r.modelContextLimit,
		OutputTokenReserve:  r.outputReserve,
		UserID:              r.userID,
		AgentID:             agentDef.Name(),
		Logger:              r.logger,
	}
}

func resolveSubagent(resolver SubagentResolver, name string) (*Agent, error) {
	if resolver == nil {
		return nil, fmt.Errorf("runtime: subagent resolver is nil")
	}
	if name == "" {
		return nil, fmt.Errorf("runtime: subagent name is required")
	}
	agentDef, ok := resolver.Get(name)
	if !ok || agentDef == nil {
		return nil, fmt.Errorf("runtime: subagent %q not found", name)
	}
	return agentDef, nil
}

func (r *Runtime) ensureSubagentAllowed(name string) error {
	r.mu.RLock()
	caller := r.agent
	r.mu.RUnlock()
	return ensureSubagentAllowedFor(caller, name)
}

func ensureSubagentAllowedFor(caller *Agent, name string) error {
	if name == "" {
		return fmt.Errorf("runtime: subagent name is required")
	}

	if caller == nil {
		return nil
	}
	if !caller.HasSubagent(name) {
		return fmt.Errorf("runtime: agent %q is not allowed to call subagent %q", caller.Name(), name)
	}
	return nil
}

func resolveTransferStateFrom(caller *Agent, sessionSnap SessionSnapshot, convState ConversationState, req TransferRequest, resolver SubagentResolver, rootAgent *Agent, filter TransferInputFilter) (*Agent, SessionSnapshot, ConversationState, *tools.Registry, error) {
	if err := ensureTransferAllowedFor(caller, rootAgent, req.AgentName); err != nil {
		return nil, SessionSnapshot{}, ConversationState{}, nil, err
	}

	childAgent, err := resolveTransferAgent(resolver, rootAgent, req.AgentName)
	if err != nil {
		return nil, SessionSnapshot{}, ConversationState{}, nil, err
	}

	sessionSnap = cloneSessionSnapshot(sessionSnap)
	sessionSnap.Config = childAgent.SessionConfig()
	sessionSnap.Budget = rebaseBudgetSnapshot(sessionSnap.Budget, sessionSnap.Config.BudgetConfig)
	sessionSnap.Metadata = mergeStringMaps(sessionSnap.Metadata, req.Metadata)
	if sessionSnap.Metadata == nil {
		sessionSnap.Metadata = make(map[string]string)
	}
	sessionSnap.Metadata[ActiveAgentMetadataKey] = req.AgentName

	convState = ConversationState{Messages: cloneMessages(convState.Messages)}
	if req.Context != nil {
		convState.Messages = cloneMessages(req.Context)
	} else if filter != nil {
		filtered, err := filter(TransferInputData{
			Session:  cloneSessionSnapshot(sessionSnap),
			Messages: cloneMessages(convState.Messages),
			Input:    req.Input,
			Metadata: cloneStringMap(sessionSnap.Metadata),
		})
		if err != nil {
			return nil, SessionSnapshot{}, ConversationState{}, nil, fmt.Errorf("runtime: transfer input filter: %w", err)
		}
		convState.Messages = cloneMessages(filtered.Messages)
		sessionSnap.Metadata = mergeStringMaps(sessionSnap.Metadata, filtered.Metadata)
		if sessionSnap.Metadata == nil {
			sessionSnap.Metadata = make(map[string]string)
		}
		sessionSnap.Metadata[ActiveAgentMetadataKey] = req.AgentName
	}
	convState.Messages = withAgentSystemPrompt(convState.Messages, childAgent.SessionConfig().SystemPrompt)

	return childAgent, sessionSnap, convState, childAgent.ToolRegistry(), nil
}

func rebaseBudgetSnapshot(snap streaming.BudgetSnapshot, cfg *streaming.BudgetConfig) streaming.BudgetSnapshot {
	out := snap
	if cfg == nil {
		out.MaxTokens = 0
		out.MaxCostUSD = 0
		out.CostPerInputToken = 0
		out.CostPerOutputToken = 0
		return out
	}
	out.MaxTokens = cfg.MaxTokens
	out.MaxCostUSD = cfg.MaxCostUSD
	out.CostPerInputToken = cfg.CostPerInputToken
	out.CostPerOutputToken = cfg.CostPerOutputToken
	return out
}

func ensureTransferAllowedFor(caller, rootAgent *Agent, name string) error {
	if name == "" {
		return fmt.Errorf("runtime: subagent name is required")
	}
	if caller == nil {
		return nil
	}
	if isTransferAllowedFor(caller, rootAgent, name) {
		return nil
	}
	return fmt.Errorf("runtime: agent %q is not allowed to transfer to agent %q", caller.Name(), name)
}

func isTransferAllowedFor(caller, rootAgent *Agent, name string) bool {
	if caller == nil {
		return true
	}
	if caller.HasSubagent(name) {
		return true
	}
	if rootAgent != nil && caller.Name() != rootAgent.Name() && rootAgent.HasSubagent(name) {
		return true
	}
	return transferRootNameFor(caller, rootAgent) == name
}

func transferRootNameFor(caller, rootAgent *Agent) string {
	if caller == nil || rootAgent == nil {
		return ""
	}
	rootName := rootAgent.Name()
	if rootName == "" || caller.Name() == rootName {
		return ""
	}
	return rootName
}

func resolveTransferAgent(resolver SubagentResolver, rootAgent *Agent, name string) (*Agent, error) {
	if rootAgent != nil && name == rootAgent.Name() {
		return rootAgent, nil
	}
	return resolveSubagent(resolver, name)
}

func withAgentSystemPrompt(messages []llm.Message, systemPrompt string) []llm.Message {
	out := make([]llm.Message, 0, len(messages)+1)
	if strings.TrimSpace(systemPrompt) != "" {
		out = append(out, llm.Message{Role: "system", Content: systemPrompt})
	}
	for _, msg := range messages {
		if msg.Role == "system" {
			continue
		}
		out = append(out, cloneMessage(msg))
	}
	return out
}

func transferToolName(agentName string) string {
	return agentsubagent.TransferToolName(agentName)
}

func consultToolName(agentName string) string {
	return agentsubagent.ConsultToolName(agentName)
}

func parseConsultRequest(target string, input map[string]any) (ConsultRequest, error) {
	payload, err := agentsubagent.ParseConsultPayload(target, input)
	if err != nil {
		return ConsultRequest{}, err
	}
	return ConsultRequest{
		AgentName: payload.AgentName,
		Input:     payload.Input,
		Metadata:  payload.Metadata,
	}, nil
}

func parseTransferRequest(target string, input map[string]any, fallbackInput string) (TransferRequest, error) {
	payload, err := agentsubagent.ParseTransferPayload(target, input, fallbackInput)
	if err != nil {
		return TransferRequest{}, err
	}
	return TransferRequest{
		AgentName: payload.AgentName,
		Input:     payload.Input,
		Metadata:  payload.Metadata,
	}, nil
}

func mergeStringMaps(base, override map[string]string) map[string]string {
	if base == nil && override == nil {
		return nil
	}
	out := cloneStringMap(base)
	if out == nil {
		out = make(map[string]string)
	}
	for k, v := range override {
		out[k] = v
	}
	return out
}
