package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/compaction/strategy/tokenlimit"
	"github.com/yourorg/agent-sdk/compaction/strategy/turnwindow"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	"github.com/yourorg/agent-sdk/store"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

const defaultMaxSteps = 25

var streamSendTimeout = 250 * time.Millisecond

type RuntimeConfig struct {
	Provider            llm.Provider
	Agent               *Agent
	RootAgent           *Agent
	SubagentResolver    SubagentResolver
	Session             *Session
	Tools               *tools.Registry
	Dispatcher          *tools.Dispatcher
	MaxToolConcurrency  int
	Middleware          []Middleware
	StreamingMiddleware []StreamingMiddleware
	MaxSteps            int
	TransferInputFilter TransferInputFilter
	Tracer              observability.Tracer
	ConversationStore   store.ConversationStore
	Conversation        *Conversation
	SessionStore        store.SessionStore
	Compactor           compaction.Compactor
	ModelContextLimit   int
	OutputTokenReserve  int
	UserID              string
	AgentID             string
	Logger              zerolog.Logger
}

type Runtime struct {
	mu                  sync.RWMutex
	generation          uint64
	provider            llm.Provider
	agent               *Agent
	rootAgent           *Agent
	subagentResolver    SubagentResolver
	session             *Session
	conv                *Conversation
	tools               *tools.Registry
	dispatcher          *tools.Dispatcher
	maxToolConcurrency  int
	middleware          []Middleware
	streamMiddleware    []StreamingMiddleware
	maxSteps            int
	transferInputFilter TransferInputFilter
	tracer              observability.Tracer
	conversationStore   store.ConversationStore
	sessionStore        store.SessionStore
	compactor           compaction.Compactor
	modelContextLimit   int
	outputReserve       int
	userID              string
	agentID             string
	logger              zerolog.Logger
}

func NewRuntime(cfg RuntimeConfig) *Runtime {
	rootAgent := cfg.RootAgent
	if rootAgent == nil {
		rootAgent = cfg.Agent
	}
	if cfg.Agent != nil {
		if cfg.Session == nil {
			cfg.Session = NewSession(cfg.Agent.SessionConfig())
		} else {
			resolvedAgent := ResolveSessionAgent(cfg.Agent, cfg.Session.Snapshot(), cfg.SubagentResolver)
			if resolvedAgent != cfg.Agent {
				cfg.Agent = resolvedAgent
				bindSessionToAgent(cfg.Session, cfg.Agent)
			} else {
				applyAgentDefaultsToSession(cfg.Session, cfg.Agent)
			}
		}
		cfg.Tools = mergeToolRegistries(cfg.Agent.ToolRegistry(), cfg.Tools)
	}
	if cfg.Session == nil {
		cfg.Session = NewSession(SessionConfig{})
	}
	if cfg.Tools == nil {
		cfg.Tools = tools.NewRegistry()
	}
	if cfg.Dispatcher == nil {
		cfg.Dispatcher = tools.NewDispatcher(cfg.Tools)
	}

	var conv *Conversation
	if cfg.Conversation != nil {
		conv = cfg.Conversation
	} else {
		conv = NewConversation(
			cfg.Session.Config.TrimStrategy,
			cfg.Session.Config.MaxHistory,
			cfg.Session.Config.TokenBudget,
		)
		if cfg.Session.Config.SystemPrompt != "" {
			conv.SetSystem(cfg.Session.Config.SystemPrompt)
		}
	}

	maxSteps := cfg.MaxSteps
	if maxSteps <= 0 {
		maxSteps = defaultMaxSteps
	}

	rt := &Runtime{
		provider:            cfg.Provider,
		agent:               cfg.Agent,
		rootAgent:           rootAgent,
		subagentResolver:    cfg.SubagentResolver,
		session:             cfg.Session,
		conv:                conv,
		tools:               cfg.Tools,
		dispatcher:          cfg.Dispatcher,
		maxToolConcurrency:  cfg.MaxToolConcurrency,
		maxSteps:            maxSteps,
		transferInputFilter: cfg.TransferInputFilter,
		tracer:              cfg.Tracer,
		conversationStore:   cfg.ConversationStore,
		sessionStore:        cfg.SessionStore,
		compactor:           cfg.Compactor,
		modelContextLimit:   cfg.ModelContextLimit,
		outputReserve:       cfg.OutputTokenReserve,
		userID:              cfg.UserID,
		agentID:             cfg.AgentID,
		logger:              cfg.Logger,
	}
	if rt.compactor == nil && cfg.Session != nil {
		rt.compactor = defaultCompactor(cfg.Session)
	}
	if rt.modelContextLimit <= 0 && cfg.Session != nil {
		rt.modelContextLimit = cfg.Session.Config.TokenBudget
	}
	if rt.outputReserve <= 0 {
		rt.outputReserve = 20000
	}
	rt.middleware = append([]Middleware(nil), cfg.Middleware...)
	rt.streamMiddleware = append([]StreamingMiddleware(nil), cfg.StreamingMiddleware...)

	return rt
}

func applyAgentDefaultsToSession(session *Session, agent *Agent) {
	if session == nil || agent == nil {
		return
	}

	snap := session.Snapshot()
	agentCfg := agent.SessionConfig()

	if snap.Config.Model == "" {
		snap.Config.Model = agentCfg.Model
	}
	if snap.Config.SystemPrompt == "" {
		snap.Config.SystemPrompt = agentCfg.SystemPrompt
	}
	if snap.Config.MaxTokens <= 0 {
		snap.Config.MaxTokens = agentCfg.MaxTokens
	}
	if snap.Config.Temperature == nil {
		snap.Config.Temperature = cloneFloat32Ptr(agentCfg.Temperature)
	}
	if snap.Config.TrimStrategy == "" {
		snap.Config.TrimStrategy = agentCfg.TrimStrategy
	}
	if snap.Config.MaxHistory <= 0 {
		snap.Config.MaxHistory = agentCfg.MaxHistory
	}
	if snap.Config.TokenBudget <= 0 {
		snap.Config.TokenBudget = agentCfg.TokenBudget
	}
	if snap.Config.BudgetConfig == nil {
		snap.Config.BudgetConfig = cloneBudgetConfig(agentCfg.BudgetConfig)
	}

	session.Restore(snap)
}

func mergeToolRegistries(agentTools, cfgTools *tools.Registry) *tools.Registry {
	if agentTools == nil && cfgTools == nil {
		return nil
	}
	if agentTools == nil {
		return cfgTools
	}
	if cfgTools == nil {
		return agentTools
	}

	merged := tools.NewRegistry()
	for _, t := range agentTools.All() {
		merged.Register(t)
	}
	for _, t := range cfgTools.All() {
		merged.Register(t)
	}
	return merged
}

func (r *Runtime) HasProvider() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.provider != nil
}

func (r *Runtime) sendStreamingEvent(ctx context.Context, out chan<- streaming.Event, event streaming.Event) error {
	select {
	case out <- event:
		return nil
	default:
	}

	timer := time.NewTimer(streamSendTimeout)
	defer timer.Stop()

	select {
	case out <- event:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("runtime: stream output cancelled: %w", ctx.Err())
	case <-timer.C:
		return fmt.Errorf("runtime: stream consumer is not draining events")
	}
}

func defaultCompactor(session *Session) compaction.Compactor {
	switch session.Config.TrimStrategy {
	case TrimSlidingWindow:
		return compaction.NewEngine(
			compaction.MaxTurnsTrigger{MaxTurns: session.Config.MaxHistory},
			turnwindow.New(session.Config.MaxHistory),
			nil,
		)
	case TrimTokenCount:
		softLimit := session.Config.TokenBudget * 80 / 100
		if softLimit <= 0 {
			softLimit = session.Config.TokenBudget
		}
		return compaction.NewEngine(
			compaction.MaxTokensTrigger{MaxTokens: softLimit},
			tokenlimit.New(softLimit, nil),
			nil,
		)
	default:
		return nil
	}
}

func toLLMToolDefs(stepTools []tools.ToolDef) []llm.ToolDef {
	llmTools := make([]llm.ToolDef, len(stepTools))
	for i, t := range stepTools {
		var params map[string]any
		if t.Schema != nil {
			params = t.Schema.ToMap()
		}
		llmTools[i] = llm.ToolDef{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		}
	}
	return llmTools
}

func messagesSummary(msgs []llm.Message) []map[string]string {
	summary := make([]map[string]string, len(msgs))
	for i, m := range msgs {
		s := map[string]string{"role": m.Role}
		if m.Content != "" {
			if len(m.Content) > 2000 {
				s["content"] = m.Content[:1997] + "..."
			} else {
				s["content"] = m.Content
			}
		}
		if m.ToolCallID != "" {
			s["tool_call_id"] = m.ToolCallID
		}
		summary[i] = s
	}
	return summary
}

func otelProviderName(providerName string) string {
	switch providerName {
	case "claude":
		return "anthropic"
	case "gemini":
		return "gcp.gemini"
	default:
		return providerName
	}
}

func messagesToOTelFormat(msgs []llm.Message) []map[string]any {
	result := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		entry := map[string]any{"role": m.Role}
		parts := make([]map[string]any, 0)
		if m.Content != "" {
			content := m.Content
			if len(content) > 2000 {
				content = content[:1997] + "..."
			}
			parts = append(parts, map[string]any{"type": "text", "content": content})
		}
		for _, tc := range m.ToolCalls {
			parts = append(parts, map[string]any{
				"type":      "tool_call",
				"id":        tc.ID,
				"name":      tc.Name,
				"arguments": tc.Input,
			})
		}
		if m.ToolCallID != "" {
			content := m.Content
			if len(content) > 2000 {
				content = content[:1997] + "..."
			}
			parts = append(parts, map[string]any{
				"type":            "tool_call_response",
				"id":              m.ToolCallID,
				"result":          content,
			})
		}
		if len(parts) > 0 {
			entry["parts"] = parts
		}
		result = append(result, entry)
	}
	return result
}

func extractContentFromOTelMsg(msg map[string]any) string {
	parts, ok := msg["parts"].([]map[string]any)
	if !ok {
		if content, ok := msg["content"].(string); ok {
			return content
		}
		return ""
	}
	for _, p := range parts {
		if p["type"] == "text" {
			if content, ok := p["content"].(string); ok {
				return content
			}
		}
	}
	return ""
}

func extractResponseText(resp *llm.Response) string {
	if resp == nil {
		return ""
	}
	for _, block := range resp.Content {
		if block.Type == "text" {
			return block.Text
		}
	}
	return ""
}

func LoadConversationFromStore(ctx context.Context, store store.ConversationStore, convID string) (*Conversation, error) {
	messages, err := store.Load(ctx, convID)
	if err != nil {
		return nil, fmt.Errorf("loading conversation %s: %w", convID, err)
	}
	if messages == nil {
		return nil, fmt.Errorf("conversation %s not found", convID)
	}
	state := ConversationState{
		Messages: messages,
		Strategy: TrimSlidingWindow,
		MaxMsgs:  50,
		MaxChars: 128000 * 4,
	}
	return NewConversationFromState(state), nil
}
