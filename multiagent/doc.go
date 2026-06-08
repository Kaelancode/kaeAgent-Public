// Package multiagent provides compatibility and router helpers around the core
// agent runtime's multi-agent model.
//
// The primary model-driven subagent mechanisms live in package agent. Declared
// subagents may be exposed to the model as synthetic consult_<subagent> and
// transfer_to_<subagent> tools during Runtime.Run or Runtime.Stream.
//
// This package is intentionally thin and should not be confused with the core
// model-driven subagent path:
//
//   - Orchestrator.Consult and ConsultStream call a selected subagent and return
//     its result to the caller. These are API-driven helper calls: application
//     code has already chosen the target agent.
//   - Orchestrator.Transfer and TransferStream switch reply ownership to a
//     selected subagent through the session active-agent state. These helpers
//     also require application code to choose the target agent.
//   - WorkflowAgentTool and RegisterWorkflowAgentTools are deprecated
//     compatibility wrappers. Prefer package workflow for deterministic
//     application-owned workflow composition.
//   - Router.Register registers agents for this compatibility layer and tag
//     discovery.
//     It is not required for normal Runtime.Run model-driven consult/transfer.
//
// Do not use router/tag lookup as the core orchestration intelligence. The
// caller agent or application should choose the subagent explicitly; Router is
// only a registry/discovery helper.
package multiagent
