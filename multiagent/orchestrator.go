// multiagent/orchestrator.go
package multiagent

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
	"github.com/Kaelancode/kaeAgent-Public/workflow"
)

// Orchestrator provides thin consult/transfer helpers over the core
// agent.Agent + Session + Runtime model.
type Orchestrator struct {
	provider llm.Provider
	router   *Router
	registry *tools.Registry
}

const ActiveAgentMetadataKey = agent.ActiveAgentMetadataKey

type ConsultRequest struct {
	Runtime   *agent.Runtime
	SessionID string
	AgentName string
	Input     string
	Context   []llm.Message
	Metadata  map[string]string
}

type TransferRequest struct {
	Runtime   *agent.Runtime
	AgentName string
	Input     string
	Context   []llm.Message
	Metadata  map[string]string
}

type JoinResult = workflow.JoinResult

// NewOrchestrator creates an orchestrator with a provider and router.
func NewOrchestrator(provider llm.Provider, router *Router) *Orchestrator {
	return &Orchestrator{
		provider: provider,
		router:   router,
		registry: tools.NewRegistry(),
	}
}

// ToolRegistry returns the orchestrator's compatibility tool registry so callers
// can add deterministic workflow tools alongside workflow.WorkflowAgentTool values.
func (o *Orchestrator) ToolRegistry() *tools.Registry {
	return o.registry
}

// RegisterWorkflowAgentTools creates a workflow-invoked tool for each
// registered agent in the router and adds them to the orchestrator's tool
// registry.
func (o *Orchestrator) RegisterWorkflowAgentTools() {
	for _, name := range o.router.List() {
		cfg, ok := o.router.Get(name)
		if !ok {
			continue
		}
		tool := WorkflowAgentTool(cfg, o.provider)
		o.registry.Register(tool)
	}
}

// RegisterAgentTools is kept for compatibility. Prefer
// RegisterWorkflowAgentTools for deterministic workflow composition.
//
// Deprecated: use RegisterWorkflowAgentTools.
func (o *Orchestrator) RegisterAgentTools() {
	o.RegisterWorkflowAgentTools()
}

func (o *Orchestrator) Consult(ctx context.Context, req ConsultRequest) (string, error) {
	parent := o.compatibilityParentRuntime(req.Runtime, req.SessionID, req.Metadata)

	result, err := parent.Consult(ctx, routerResolver{router: o.router}, agent.ConsultRequest{
		AgentName: req.AgentName,
		Input:     req.Input,
		Context:   req.Context,
		Metadata:  req.Metadata,
	})
	if err != nil {
		return "", fmt.Errorf("orchestrator: consult %q: %w", req.AgentName, err)
	}
	return result, nil
}

func (o *Orchestrator) ConsultStream(ctx context.Context, req ConsultRequest) (<-chan streaming.Event, error) {
	parent := o.compatibilityParentRuntime(req.Runtime, req.SessionID, req.Metadata)

	ch, err := parent.ConsultStream(ctx, routerResolver{router: o.router}, agent.ConsultRequest{
		AgentName: req.AgentName,
		Input:     req.Input,
		Context:   req.Context,
		Metadata:  req.Metadata,
	})
	if err != nil {
		return nil, fmt.Errorf("orchestrator: consult stream %q: %w", req.AgentName, err)
	}
	return ch, nil
}

func (o *Orchestrator) ActiveAgent(rt *agent.Runtime, rootAgent string) string {
	if rt == nil {
		return rootAgent
	}
	return rt.ActiveAgent(routerResolver{router: o.router}, rootAgent)
}

func (o *Orchestrator) Transfer(ctx context.Context, req TransferRequest) (string, error) {
	if req.Runtime == nil {
		return "", fmt.Errorf("orchestrator: transfer %q requires a parent runtime", req.AgentName)
	}

	parent := o.compatibilityParentRuntime(req.Runtime, "", nil)
	result, err := parent.Transfer(ctx, routerResolver{router: o.router}, agent.TransferRequest{
		AgentName: req.AgentName,
		Input:     req.Input,
		Context:   req.Context,
		Metadata:  req.Metadata,
	})
	if err != nil {
		return "", fmt.Errorf("orchestrator: transfer %q: %w", req.AgentName, err)
	}
	if parent != req.Runtime {
		req.Runtime.SetSessionMetadata(ActiveAgentMetadataKey, req.AgentName)
	}
	return result, nil
}

func (o *Orchestrator) TransferStream(ctx context.Context, req TransferRequest) (<-chan streaming.Event, error) {
	if req.Runtime == nil {
		return nil, fmt.Errorf("orchestrator: transfer stream %q requires a parent runtime", req.AgentName)
	}

	parent := o.compatibilityParentRuntime(req.Runtime, "", nil)
	ch, err := parent.TransferStream(ctx, routerResolver{router: o.router}, agent.TransferRequest{
		AgentName: req.AgentName,
		Input:     req.Input,
		Context:   req.Context,
		Metadata:  req.Metadata,
	})
	if err != nil {
		return nil, fmt.Errorf("orchestrator: transfer stream %q: %w", req.AgentName, err)
	}
	if parent != req.Runtime {
		ch = setActiveAgentMetadataOnDone(ch, req.Runtime, req.AgentName)
	}
	return ch, nil
}

// RunAgent performs an isolated consult-style sub-agent invocation and returns
// the child agent's final text result to the caller.
//
// Deprecated: use Consult with an explicit ConsultRequest for new code.
func (o *Orchestrator) RunAgent(ctx context.Context, name string, input string) (string, error) {
	return o.Consult(ctx, ConsultRequest{
		AgentName: name,
		Input:     input,
	})
}

// RunByTag is a convenience helper that consults the first router match for a
// tag. It is not the preferred orchestration model; caller-chosen agent names
// via Consult/Transfer are the primary path.
//
// Deprecated: prefer explicit agent selection with Consult or Transfer.
func (o *Orchestrator) RunByTag(ctx context.Context, tag string, input string) (string, error) {
	cfg, err := o.router.Route(tag)
	if err != nil {
		return "", fmt.Errorf("orchestrator: %w", err)
	}
	return o.Consult(ctx, ConsultRequest{
		AgentName: cfg.Name,
		Input:     input,
	})
}

// WorkflowAgentTool wraps an agent as a callable workflow step.
//
// Deprecated: use workflow.WorkflowAgentTool.
func WorkflowAgentTool(cfg AgentConfig, provider llm.Provider) tools.ToolDef {
	return workflow.WorkflowAgentTool(workflow.AgentConfig{
		Agent:       cfg.Definition(),
		Name:        cfg.Name,
		Description: cfg.Description,
		Tags:        cfg.Tags,
		MaxSteps:    cfg.MaxSteps,
	}, provider)
}

// AgentTool is kept for compatibility with older router-based code.
//
// Deprecated: use workflow.WorkflowAgentTool.
func AgentTool(cfg AgentConfig, provider llm.Provider) tools.ToolDef {
	return WorkflowAgentTool(cfg, provider)
}

// JoinAll waits for multiple child agent runs to complete concurrently.
func JoinAll(ctx context.Context, tasks map[string]func(ctx context.Context) (string, error)) (map[string]string, error) {
	return workflow.JoinAll(ctx, tasks)
}

// JoinAllDetailed waits for multiple child agent runs, returns partial results,
// and cancels queued/in-flight siblings after the first error.
func JoinAllDetailed(ctx context.Context, tasks map[string]func(ctx context.Context) (string, error)) (map[string]JoinResult, error) {
	return workflow.JoinAllDetailed(ctx, tasks)
}

type routerResolver struct {
	router *Router
}

var _ agent.SubagentResolver = (routerResolver{})

func (r routerResolver) Get(name string) (*agent.Agent, bool) {
	cfg, ok := r.router.Get(name)
	if !ok {
		return nil, false
	}
	return cfg.Definition(), true
}

func (o *Orchestrator) newConsultParentRuntime(req ConsultRequest) *agent.Runtime {
	rt := agent.NewRuntime(agent.RuntimeConfig{Provider: o.provider})
	if req.SessionID == "" && len(req.Context) == 0 && len(req.Metadata) == 0 {
		return rt
	}

	sessionSnap := rt.SessionSnapshot()
	if req.SessionID != "" {
		sessionSnap.ID = req.SessionID
	}
	if len(req.Metadata) > 0 {
		sessionSnap.Metadata = cloneMetadata(req.Metadata)
	}
	rt.LoadState(sessionSnap, agent.ConversationState{})
	return rt
}

func (o *Orchestrator) compatibilityParentRuntime(parent *agent.Runtime, sessionID string, metadata map[string]string) *agent.Runtime {
	if parent == nil {
		return o.newConsultParentRuntime(ConsultRequest{
			SessionID: sessionID,
			Metadata:  metadata,
		})
	}
	if parent.HasProvider() {
		return parent
	}

	compat := agent.NewRuntime(agent.RuntimeConfig{
		Provider: o.provider,
	})
	sessionSnap := parent.SessionSnapshot()
	if sessionID != "" {
		sessionSnap.ID = sessionID
	}
	if len(metadata) > 0 {
		sessionSnap.Metadata = cloneMetadata(metadata)
	}
	compat.LoadState(sessionSnap, parent.ConversationSnapshot())
	return compat
}

func cloneMetadata(meta map[string]string) map[string]string {
	if meta == nil {
		return nil
	}
	out := make(map[string]string, len(meta))
	for k, v := range meta {
		out[k] = v
	}
	return out
}

func setActiveAgentMetadataOnDone(ch <-chan streaming.Event, rt *agent.Runtime, agentName string) <-chan streaming.Event {
	out := make(chan streaming.Event)
	go func() {
		defer close(out)
		for event := range ch {
			if event.Kind == streaming.EventDone {
				rt.SetSessionMetadata(ActiveAgentMetadataKey, agentName)
			}
			out <- event
		}
	}()
	return out
}
