# Multi-Agent Implementation Plan

## Current Status

The core plan is implemented:

- `agent.Agent` owns tools and declared subagents
- model-driven consult and transfer run through `Runtime.Run` and `Runtime.Stream`
- consult remains agent-as-tool and transfer changes reply ownership
- active-agent metadata is session-scoped
- transfer can continue in the same engine-driven turn
- deterministic agent steps live in `workflow`
- `multiagent` remains a compatibility/router/discovery layer
- consult, transfer, streaming, persistence, tracing, and examples are test-covered

Swarm orchestration remains a future, separate coordinator/task layer.

This document describes the incremental implementation plan for moving the SDK's multi-agent layer to the intended architecture.

## Target Outcome

The final model should provide:

- session-isolated multi-agent orchestration
- explicit `consult` semantics for agent-as-tool delegation
- explicit `transfer` semantics for reply ownership change
- a clear design path for future swarm orchestration when many agents need to work in
  parallel under one session
- no implicit global child-runtime reuse keyed only by agent name
- agent definitions that own their own tools and subagents
- execution that binds agent + session at run time
- caller-chosen subagent invocation as the primary orchestration path

## Swarm Design Consideration

Swarm should be treated as a future layer above the core consult/transfer model.

It is needed for cases where:

- many agents may work in parallel on one complex task
- some tasks may later require:
  - user feedback
  - user input
  - approvals
  - external signals

This should not be forced into `transfer`.

The intended distinction is:

- `consult` = isolated delegated call
- `transfer` = exclusive reply-ownership switch
- `swarm` = coordinator-managed parallel task orchestration

If implemented later, swarm should likely introduce:

- a session-level coordinator
- task state for spawned agent work
- explicit waiting states such as:
  - waiting for user input
  - waiting for approval
  - running
  - done
  - failed
- a way to resume a specific blocked task when the user or system provides the needed input

This should be designed as a higher-level orchestration concern, not as an overload of
the basic transfer API.

## Phase 1: Introduce a First-Class Agent Definition in `agent`

### Goal

Make the core `agent` package own the primary `Agent + Session + Runtime` model.

### Changes

Files:

- `agent/agent.go` (new)
- `agent/runtime.go`
- `agent/runtime_engine_turn.go`
- `agent/runtime_subagent.go`
- related examples/tests

Tasks:

- Add a first-class `Agent` definition to `agent`.
- Move agent-owned setup into that definition:
  - name
  - model
  - system prompt
  - tools
  - subagents
  - execution limits/policies
- Make runtime execution conceptually bind:
  - an `Agent`
  - a `Session`
- Reduce reliance on loose runtime-only tool wiring as the long-term model.

### Expected Result

The primary SDK model becomes:

- `Agent` owns capabilities
- `Session` owns state
- `Runtime` executes `Agent + Session`

`multiagent` should build on this model rather than define a separate one.

## Phase 2: Normalize Current `multiagent` Behavior as Consult-Only

### Goal

Make the existing `multiagent` code reflect what it actually does today:

- create fresh sub-agent runtimes
- run them ephemerally
- return results to the caller

### Changes

Files:

- `multiagent/orchestrator.go`

Tasks:

- Remove misleading reuse language from comments.
- Reframe current sub-agent execution as consult-style behavior.
- Remove or de-emphasize any child-runtime tracking that implies persistent global ownership.
- Introduce a single helper for constructing a fresh runtime from `AgentConfig`.
- Stop creating child tool registries that are detached from agent definitions.

### Expected Result

`RunAgent(...)` clearly means:

- resolve agent definition
- create isolated runtime
- run once
- return result
- discard runtime

## Phase 3: Move Agent Setup Into Agent Definitions

### Goal

Make agent definitions own their tools and subagents directly.

### Changes

Files:

- `multiagent/router.go`
- `multiagent/orchestrator.go`
- possibly new `multiagent/agent.go`

Tasks:

- Extend the agent definition shape so each agent can carry:
  - tools
  - subagents
  - execution limits/policies
- Stop relying on the orchestrator to invent empty child tool registries.
- Ensure child runtimes are built from the target agent definition itself.

### Expected Result

Agent construction becomes closer to:

- define agents and their capabilities up front
- create or restore a session separately
- bind agent + session into a run

## Selection Rule

The multi-agent framework should treat subagent selection like a function/tool call:

- caller agent chooses target subagent explicitly
- router is discovery support only
- exact name lookup is primary
- `Route(tag)`-style first-match behavior is convenience only and should not be treated as the core orchestration decision path

## Phase 4: Add Explicit Consult API

### Goal

Introduce a first-class consult path instead of relying on a generic "run agent" abstraction.

### Changes

Files:

- `multiagent/orchestrator.go`

Add:

```go
type ConsultRequest struct {
    SessionID string
    AgentName string
    Input     string
    Context   []llm.Message
    Metadata  map[string]string
}
```

Add:

```go
func (o *Orchestrator) Consult(ctx context.Context, req ConsultRequest) (string, error)
```

### Rules

- Only explicitly passed context is visible to the consult callee.
- Consult does not change the active replying agent.
- Consult returns a result to the caller.

### Compatibility

`RunAgent(...)` may temporarily call `Consult(...)` internally for backward compatibility.

## Phase 5: Workflow Agent Tools

### Goal

Keep deterministic workflow-invoked agent steps separate from model-driven consult.

### Changes

Files:

- `workflow/workflow.go`
- `multiagent/orchestrator.go`

Tasks:

- Expose `workflow.WorkflowAgentTool(...)` for application/workflow-owned agent invocation.
- Keep `multiagent.AgentTool(...)` only as a deprecated compatibility wrapper.
- Ensure workflow-wrapped agent execution remains ephemeral and isolated.
- Keep model-driven agent delegation on the core runtime's `consult_<subagent>` synthetic tools.
- Document that `multiagent.Router.Register(...)` is for tag/compatibility registration, not the normal model-driven `Runtime.Run` subagent setup.
- Document that `Orchestrator.Consult(...)` and `Orchestrator.Transfer(...)` are API-driven helpers where application code has already selected the target agent.

### Expected Result

Workflow-owned agent steps are named as workflow primitives, while model-driven subagent selection remains consult semantics.

## Runtime Inheritance Policy

### Decision

For simplicity, child agent runs should inherit the caller/orchestrator runtime capabilities by default.

This includes:

- tracer
- middleware
- conversation/session stores where applicable
- compactor
- model context limit
- output token reserve
- logger
- `MaxToolConcurrency`
- user/session metadata that is part of runtime execution context

### Requirement

Child runs must be traced.

The simplest rule is:

- inherit runtime execution capabilities from the caller/orchestrator
- but do not implicitly inherit parent conversation state unless explicitly passed

This keeps observability and guardrails consistent while preserving context isolation.

## Child Tool Policy

### Decision

Sub-agents should get their own tools.

That means:

- child tool access is defined by the child agent definition
- child runs do not automatically inherit the parent's full tool registry
- global/orchestrator tools should not be injected into every child by default

### Reason

This keeps delegated agents explicit and reduces accidental capability bleed across agents.

## Phase 6: Add Session-Scoped Transfer State

### Goal

Introduce explicit reply ownership transfer without reusing global child runtimes.

### Changes

Files:

- `multiagent/orchestrator.go`

Add a session-level state structure, for example:

```go
type sessionAgentState struct {
    ActiveAgent string
}
```

Add storage keyed by session ID.

### Rules

- Active replying agent is tracked per session.
- Different sessions do not share active-agent state.
- Transfer state is not keyed by agent name alone.

## Phase 7: Add Explicit Transfer API

### Goal

Provide a first-class transfer path distinct from consult.

### Changes

Files:

- `multiagent/orchestrator.go`

Add:

```go
type TransferRequest struct {
    SessionID string
    AgentName string
    Input     string
    Context   []llm.Message
    Metadata  map[string]string
}
```

Add:

```go
func (o *Orchestrator) Transfer(ctx context.Context, req TransferRequest) (string, error)
```

### Rules

- The callee's reply becomes the user-facing reply.
- The session's active agent is updated.
- Transfer is a reply-ownership operation, not just a delegated subtask.

## Transfer Context Policy

### Decision

A transferred sub-agent should receive only the history relevant to why the transfer is being made.

It should not automatically receive the full parent conversation transcript.

The transferred agent keeps its own system message.

### Implications

- transfer payloads should be selective
- transfer should not duplicate or blindly copy parent system prompts
- the transfer path should eventually support preparing a relevant conversation slice or summary
- the child agent's system prompt remains the authoritative instructions for that child

## Future Swarm Context Policy

If swarm is added later, parallel agent tasks should not all receive the same raw
session transcript by default.

Instead:

- each task should receive an explicit context projection
- interactive tasks may receive different context slices or summaries depending on role
- the coordinator should remain responsible for deciding what user-facing prompt or
  approval request is surfaced next
- multiple parallel tasks may exist, but only the coordinator should own the final
  session-level interaction flow

## Phase 8: Add Active-Agent Helpers

### Goal

Make session-scoped reply ownership explicit and accessible.

### Changes

Files:

- `multiagent/orchestrator.go`

Add helpers such as:

```go
func (o *Orchestrator) ActiveAgent(sessionID string) (string, bool)
func (o *Orchestrator) SetActiveAgent(sessionID, agentName string)
func (o *Orchestrator) ClearActiveAgent(sessionID string)
```

### Expected Result

The application/server can reason about which agent currently owns the session reply path.

## Phase 9: Decide Transfer Persistence

### Goal

Decide whether active-agent state should survive session reloads.

### Options

#### In-memory only

- simpler
- no persistence changes
- active agent resets after process restart or runtime reconstruction

#### Persist in session metadata

- active agent survives resume
- small, practical first step
- can be stored under a metadata key such as `active_agent`

### Recommendation

Start with in-memory only unless the product explicitly needs resumed transfer state.
If persistence is needed later, session metadata is the simplest first implementation.

## Future Phase: Add Swarm Coordinator

### Goal

Support parallel multi-agent work under one session without abusing `transfer`.

### Changes

Likely files:

- `agent/` core task/coordinator abstractions
- thin `multiagent/` helpers or discovery adapters if still needed

### Expected Model

- one coordinator per session
- many agent tasks can run in parallel
- each task has:
  - ID
  - assigned agent
  - local context
  - status
  - optional wait condition
- the coordinator decides:
  - which task can continue
  - what to ask the user next
  - how user replies or approvals resume blocked tasks

### Important Rule

Do not model swarm by allowing multiple transferred agents to own the same
session reply path at once.

Parallelism belongs to task orchestration, while reply ownership remains coordinated
at the session boundary.

## Phase 10: Add Tests

### Consult tests

Files:

- `multiagent/orchestrator_test.go` or new equivalent

Cases:

- consult returns callee result to caller
- consult does not share child state across calls
- consult does not share child state across sessions
- consult only sees explicitly passed context

### Transfer tests

Cases:

- transfer returns callee direct reply
- transfer updates active agent for the session
- different sessions can transfer to the same agent name independently
- transfer does not create global cross-session child-runtime bleed
- transfer only passes relevant history, not implicit full parent history
- transferred agent keeps its own system message

### WorkflowAgentTool tests

Cases:

- workflow-agent tool path executes an isolated child runtime
- workflow-wrapped agent execution remains ephemeral
- deprecated `AgentTool(...)` wrapper preserves compatibility

### Streaming tests

Cases:

- child consult path can surface streaming child results
- transfer path can surface streaming child results when the callee owns the reply
- parent/orchestrator correctly handles streamed child output and terminal events

### Requirement

Multi-agent orchestration must be test-covered. Tests are not optional for this refactor.

### Current Status

The implementation now includes explicit streaming APIs:

- `ConsultStream(...)`
- `TransferStream(...)`

These reuse the core `Runtime.Stream()` event contract.

## Phase 11: Packaging Decision

### Goal

Decide whether the final API should live directly in `agent` or remain behind a thin `multiagent` layer.

### Recommended Direction

The ownership model should match the core agent package:

- `Agent` owns tools and subagents
- `Session` owns state
- execution binds `Agent + Session`

That means `multiagent` should not define a different state model from `agent`.

Two acceptable end states:

1. move the primary API into `agent` and keep `multiagent` as a small compatibility/helper layer
2. keep `multiagent` as a package, but make it a thin orchestration facade over the same `Agent + Session` model

### Anti-goal

Do not keep a separate long-term design where:

- `agent` owns one state model
- `multiagent` owns a different child-runtime/state model

## Phase 12: Documentation Cleanup

### Files

- `AGENTS.md`
- `docs/multiagent-architecture.md`
- example docs or package comments as needed

### Tasks

- document consult vs transfer explicitly
- document session-scoped ownership
- document that global reuse by agent name is unsafe
- document any persisted active-agent behavior if added

## Suggested Execution Order

1. Introduce a first-class `Agent` definition in `agent`.
2. Normalize current `multiagent` behavior as consult-only.
3. Move tool/transfer ownership into agent definitions.
4. Add `Consult(...)`.
5. Rename deterministic wrapper semantics to `WorkflowAgentTool(...)` while keeping deprecated `AgentTool(...)` compatibility.
6. Add tests for consult isolation.
7. Add session-scoped transfer state.
8. Add `Transfer(...)`.
9. Add tests for transfer ownership.
10. Decide packaging: `agent` core vs thin `multiagent` facade.
11. Treat swarm as a later coordinator/task phase, not part of the initial transfer rollout.
12. Finalize documentation and compatibility notes.

## Immediate Next Step

The safest immediate implementation step is:

- add the first-class `Agent` type to `agent`

That establishes the correct ownership model before simplifying `multiagent` into a thin consult/transfer orchestration layer.

## Follow-Up Topics

These remain open for later design discussion:

- router policy beyond first-match tags
- duplicate/stale tag registration handling
- `JoinAll` partial-result and cancellation semantics
- structured transfer payload format and summarization strategy
- swarm coordinator/task model and how blocked agent tasks request user input or approval
