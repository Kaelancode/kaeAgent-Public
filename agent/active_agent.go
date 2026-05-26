package agent

func ResolveSessionAgent(root *Agent, snap SessionSnapshot, resolver SubagentResolver) *Agent {
	if root == nil {
		return nil
	}

	name := snap.Metadata[ActiveAgentMetadataKey]
	if name == "" || resolver == nil {
		return root
	}

	agentDef, ok := resolver.Get(name)
	if !ok || agentDef == nil {
		return root
	}
	return agentDef
}

func bindSessionToAgent(session *Session, agentDef *Agent) {
	if session == nil || agentDef == nil {
		return
	}

	snap := session.Snapshot()
	agentCfg := agentDef.SessionConfig()
	snap.Config.Model = agentCfg.Model
	snap.Config.SystemPrompt = agentCfg.SystemPrompt
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
