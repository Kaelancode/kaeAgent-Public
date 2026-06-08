package agent

import (
	"fmt"
	"strings"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

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
