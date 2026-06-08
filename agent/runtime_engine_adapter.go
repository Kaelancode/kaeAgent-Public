package agent

import (
	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func (e *runExecutor) engineTurnInput(userMessage string) agentengine.TurnInput {
	return agentengine.TurnInput{
		UserMessage: userMessage,
		State:       engineStateFromRunState(e.rs),
		Config:      e.engineConfig(),
		Hooks:       e.engineHooks(),
	}
}

func engineStateFromRunState(rs *runState) agentengine.State {
	if rs == nil {
		return agentengine.State{}
	}
	var messages []llm.Message
	if rs.conv != nil {
		messages = rs.conv.messagesOwned()
	}
	var budget agentengine.BudgetSnapshot
	if rs.budget != nil {
		budget = engineBudgetSnapshot(rs.budget.Snapshot())
	}

	return agentengine.State{
		Generation:  rs.generation,
		SessionID:   rs.sessionID,
		RunID:       rs.runID,
		UserID:      rs.userID,
		AgentID:     rs.agentID,
		ActiveAgent: engineAgentViewFromRunState(rs),
		RootAgent:   engineAgentViewFromAgent(rs.rootAgent, nil),
		Messages:    messages,
		Metadata:    cloneStringMap(rs.metadata),
		Budget:      budget,
	}
}

func (e *runExecutor) engineConfig() agentengine.Config {
	if e == nil || e.rs == nil {
		return agentengine.Config{}
	}

	return agentengine.Config{
		Model:              e.rs.config.Model,
		MaxTokens:          e.rs.config.MaxTokens,
		Temperature:        cloneFloat32Ptr(e.rs.config.Temperature),
		MaxSteps:           e.rt.maxSteps,
		MaxToolConcurrency: e.rt.maxToolConcurrency,
		ModelContextLimit:  e.rt.modelContextLimit,
		OutputTokenReserve: e.rt.outputReserve,
	}
}

func engineAgentViewFromRunState(rs *runState) agentengine.AgentView {
	if rs == nil {
		return agentengine.AgentView{}
	}
	return engineAgentViewFromAgent(rs.activeAgent, rs.tools)
}

func engineAgentViewFromAgent(agentDef *Agent, toolRegistry *tools.Registry) agentengine.AgentView {
	if agentDef == nil {
		return agentengine.AgentView{}
	}

	cfg := agentDef.Snapshot()
	view := agentengine.AgentView{
		Name:         cfg.Name,
		Model:        cfg.Model,
		SystemPrompt: cfg.SystemPrompt,
		MaxSteps:     cfg.MaxSteps,
		Subagents:    append([]string(nil), cfg.Subagents...),
	}
	if toolRegistry != nil {
		view.Tools = cloneToolDefs(toolRegistry.All())
	} else {
		view.Tools = cloneToolDefs(agentDef.Tools())
	}
	return view
}

func engineBudgetSnapshot(s streaming.BudgetSnapshot) agentengine.BudgetSnapshot {
	return agentengine.BudgetSnapshot{
		MaxTokens:          s.MaxTokens,
		MaxCostUSD:         s.MaxCostUSD,
		TotalInput:         s.TotalInput,
		TotalOutput:        s.TotalOutput,
		TotalCostUSD:       s.TotalCostUSD,
		CostPerInputToken:  s.CostPerInputToken,
		CostPerOutputToken: s.CostPerOutputToken,
	}
}

func stepFromEngineStepInput(input agentengine.StepInput) *Step {
	return &Step{
		SessionID:    input.SessionID,
		RunID:        input.RunID,
		StepIndex:    input.StepIndex,
		Messages:     cloneMessages(input.Messages),
		AvailTools:   cloneToolDefs(input.AvailableTools),
		ProviderName: input.ProviderName,
		UserID:       input.UserID,
		AgentID:      input.AgentID,
		AgentName:    input.AgentName,
	}
}

func streamingStepFromEngineStepInput(input agentengine.StepInput) *StreamingStep {
	return &StreamingStep{
		SessionID:    input.SessionID,
		RunID:        input.RunID,
		StepIndex:    input.StepIndex,
		Messages:     cloneMessages(input.Messages),
		AvailTools:   cloneToolDefs(input.AvailableTools),
		ProviderName: input.ProviderName,
		UserID:       input.UserID,
		AgentID:      input.AgentID,
		AgentName:    input.AgentName,
	}
}

func cloneToolDefs(defs []tools.ToolDef) []tools.ToolDef {
	if defs == nil {
		return nil
	}

	out := make([]tools.ToolDef, len(defs))
	for i, def := range defs {
		out[i] = def
		out[i].Tags = append([]string(nil), def.Tags...)
	}
	return out
}
