package agent

import (
	"context"
	"fmt"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func (e *runExecutor) engineHooks() agentengine.Hooks {
	return agentengine.Hooks{
		Complete:        e.engineComplete,
		Stream:          e.engineStream,
		ExecuteTools:    e.engineExecuteTools,
		ResolveSubagent: e.engineResolveSubagent,
		FilterTransfer:  e.engineFilterTransfer,
		Compact:         e.engineCompact,
		Checkpoint:      e.engineCheckpoint,
		SaveSession:     e.engineSaveSession,
	}
}

func (e *runExecutor) engineComplete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if e == nil || e.rt == nil || e.rt.provider == nil {
		return nil, fmt.Errorf("runtime: provider is nil")
	}
	return e.rt.provider.Complete(ctx, req)
}

func (e *runExecutor) engineStream(ctx context.Context, req *llm.Request) (<-chan llm.Event, error) {
	if e == nil || e.rt == nil || e.rt.provider == nil {
		return nil, fmt.Errorf("runtime: provider is nil")
	}
	return e.rt.provider.Stream(ctx, req)
}

func (e *runExecutor) engineExecuteTools(ctx context.Context, step agentengine.ToolStep) ([]tools.ToolResult, error) {
	if e == nil || e.rs == nil {
		return nil, fmt.Errorf("runtime: run state is nil")
	}
	maxConcurrency := step.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = e.rt.maxToolConcurrency
	}
	groupID := fmt.Sprintf("%s/step/%d/tools", e.rs.runID, step.StepIndex)
	mode := "sequential"
	if maxConcurrency > 1 && len(step.Calls) > 1 {
		mode = "parallel"
	}
	toolIndexByCall := make(map[string]int, len(step.Calls))
	for i, call := range step.Calls {
		toolIndexByCall[call.ID+"\x00"+call.Name] = i
	}
	results := tools.NewDispatcher(e.rs.tools).DispatchAllWith(ctx, step.Calls, maxConcurrency, func(ctx context.Context, call tools.ToolCall) tools.ToolResult {
		toolIndex, ok := toolIndexByCall[call.ID+"\x00"+call.Name]
		if !ok {
			toolIndex = 0
		}
		return e.dispatchOneWithTracing(ctx, nil, call, step.StepIndex, toolIndex, groupID, mode)
	})
	return results, nil
}

func (e *runExecutor) engineResolveSubagent(_ context.Context, name string) (agentengine.AgentView, bool) {
	if e == nil || e.rt == nil {
		return agentengine.AgentView{}, false
	}
	agentDef, err := resolveTransferAgent(e.rt.subagentResolver, e.rs.rootAgent, name)
	if err != nil {
		return agentengine.AgentView{}, false
	}
	return engineAgentViewFromAgent(agentDef, agentDef.ToolRegistry()), true
}

func (e *runExecutor) engineFilterTransfer(_ context.Context, input agentengine.TransferInputData) (agentengine.TransferInputResult, error) {
	if e == nil || e.rs == nil {
		return agentengine.TransferInputResult{}, fmt.Errorf("runtime: run state is nil")
	}

	filter := e.rt.resolveTransferInputFilter(nil)
	if filter == nil {
		return agentengine.TransferInputResult{
			Input:    input.Input,
			Messages: cloneMessages(input.Messages),
			Metadata: cloneStringMap(input.Metadata),
		}, nil
	}

	sessionSnap := SessionSnapshot{
		ID:       e.rs.sessionID,
		Config:   cloneSessionConfig(e.rs.config),
		Budget:   e.rs.budget.Snapshot(),
		Metadata: cloneStringMap(e.rs.metadata),
	}
	filtered, err := filter(TransferInputData{
		Session:  sessionSnap,
		Messages: cloneMessages(input.Messages),
		Input:    input.Input,
		Metadata: cloneStringMap(input.Metadata),
	})
	if err != nil {
		return agentengine.TransferInputResult{}, fmt.Errorf("runtime: transfer input filter: %w", err)
	}
	return agentengine.TransferInputResult{
		Input:    filtered.Input,
		Messages: cloneMessages(filtered.Messages),
		Metadata: cloneStringMap(filtered.Metadata),
	}, nil
}

func (e *runExecutor) engineCompact(ctx context.Context, input compaction.Input, force bool) (compaction.Output, error) {
	if e == nil || e.rt == nil || e.rt.compactor == nil {
		return compaction.Output{Messages: compaction.CloneMessages(input.Messages)}, nil
	}
	if force {
		return e.rt.compactor.ForceCompact(ctx, input)
	}
	return e.rt.compactor.Compact(ctx, input)
}

func (e *runExecutor) engineCheckpoint(ctx context.Context, _ agentengine.State) error {
	if e == nil {
		return fmt.Errorf("runtime: run executor is nil")
	}
	return e.checkpoint(ctx)
}

func (e *runExecutor) engineSaveSession(ctx context.Context, _ agentengine.State) error {
	if e == nil {
		return fmt.Errorf("runtime: run executor is nil")
	}
	return e.saveSessionData(ctx)
}
