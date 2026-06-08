# Multi-Agent Architecture

This document defines the intended multi-agent model for the SDK.

## Current Runtime Integration

Both `Runtime.Run` and `Runtime.Stream` use the internal
`agent/internal/engine.Engine.ExecuteTurn` progression loop. Model-driven consult
and transfer are still agent-owned semantics:

- synthetic subagent tools are exposed by `agent`
- consult executes as tool-like isolated child work and returns to the caller
- transfer commands are interpreted by `agent`, which changes active-agent state,
  rebinds tools/config, and rotates trace ownership
- the engine does not resolve or mutate public `agent` objects directly

## Goals

- Keep session state isolated from other sessions.
- Allow multiple registered agents to participate in one session.
- Support both agent-as-tool delegation and direct-reply transfer.
- Leave room for agent-swarm style orchestration when complex tasks require many agents
  working in parallel, including cases that may later need user input, approvals, or
  intermediate feedback.
- Avoid implicit shared child-runtime state across sessions.

## Core Principles

### 0. Agent Definition and Session State Are Separate

Agent setup should live on the agent definition itself.

An agent definition should own:

- name
- model
- system prompt
- tools
- subagents
- execution limits and policies

Session state should remain separate.

A session should own:

- session ID
- conversation/history
- metadata
- active replying agent

Execution then binds:

- an agent definition
- a session
- an orchestration mode (`normal`, `consult`, or `transfer`)

This is preferable to having orchestration fabricate child tool registries at call time.

### 0.5. The Caller Agent Chooses the Subagent

Subagent invocation is closer to a function or tool call than a framework-side routing decision.

That means:

- the current agent chooses which subagent to invoke
- the router is only a registry/discovery helper
- exact lookup by name is the primary orchestration path
- tag-based lookup is secondary and should be treated as candidate discovery or convenience

### 1. Agent Definitions Are Global, Runtime State Is Not

Registered agent definitions may be global:

- name
- model
- system prompt
- tags
- tool set
- subagents
- execution limits

But execution state is session-scoped:

- conversation state
- active replying agent
- any persistent child-agent state, if introduced later

Different sessions must not share long-lived sub-agent runtime state implicitly.

### 2. Two Orchestration Modes: Consult and Transfer

The orchestration layer must distinguish between two different agent interactions.

#### Consult

Consult is agent-as-tool delegation.

- Agent A invokes Agent B for help on a subtask.
- Agent B returns a result to Agent A.
- Agent A still owns the final reply to the user.
- Agent B does not become the active speaking agent for the session.

Consult runs should be:

- explicit
- isolated
- ephemeral

The parent agent passes only the context it chooses to share.
No implicit parent conversation sharing should occur.

#### Workflow Agent Tools

`workflow.WorkflowAgentTool(...)` is not the primary model-driven consult mechanism.
It exists for application-owned or workflow-owned orchestration where the caller has already decided that a specific agent should run as a deterministic step.

Use:

- `consult_<subagent>` when the model chooses to consult a declared subagent during an agent run.
- `workflow.WorkflowAgentTool(...)` when application code or a workflow engine invokes a specific agent step directly.

The older `multiagent.AgentTool(...)` name is a compatibility wrapper and should not be used for new workflow-facing code.

##### Model-Driven Consult Tool Contract

When an agent has declared subagents, the runtime may expose synthetic consult tools:

- `consult_<subagent>`

The structured payload currently supports:

- `input`
  - string
  - required task, question, or context for the consulted subagent
- `reason`
  - string
  - short explanation for why consult is needed
- `metadata`
  - object with string values
  - additional consult metadata for execution context

Behavior:

- the consulted agent runs in an isolated child run
- the child agent's final output is returned to the caller as a normal tool result
- consult does not update `active_agent`
- invalid target or invalid payload types return a tool error to the caller

#### Transfer

Transfer is direct reply-ownership transfer.

- Agent A hands control to Agent B.
- Agent B may reply directly to the user.
- Agent B may become the active agent for subsequent turns in that session.
- Active ownership is persisted in session metadata under `active_agent`.
- If transfer is selected during a running turn, the runtime may continue the same run under Agent B after applying transfer-input shaping and rebinding the active agent in run-local state.
- A transferred subagent may transfer back to the root/default/coordinator agent through `transfer_to_<root>` and may transfer to the root agent's declared specialists when the user changes topics.
- Transfer fails explicitly if the selected target is not declared and resolvable, if the structured transfer payload uses invalid field types, or if a transfer tool call is mixed with normal tool calls in the same step.
- Missing or unknown `active_agent` values fall back to the root/default/coordinator agent.

Transfer changes reply ownership at the session boundary.

##### Model-Driven Transfer Tool Contract

When an agent has declared subagents, the runtime may expose synthetic transfer tools:

- `transfer_to_<subagent>`

The structured payload currently supports:

- `input`
  - string
  - explicit task or continuation message for the target agent
- `reason`
  - string
  - short explanation for why transfer is happening
- `metadata`
  - object with string values
  - additional transfer metadata persisted into execution/session metadata

Behavior:

- if `input` is omitted, the runtime falls back to the assistant text from the transfer step
- the chosen transfer input becomes a new user message for the target agent in the continued run
- invalid target, invalid payload types, or mixed transfer + normal tool calls fail the run explicitly

### 2.5. Swarm Is a Separate Orchestration Layer

Swarm-style orchestration should be treated as a separate design consideration, not
as a synonym for consult or transfer.

Swarm means:

- multiple agents may work on one broader task in parallel
- some agent tasks may block waiting for:
  - user feedback
  - user input
  - approvals
  - external events
- one coordinator still owns the session-level orchestration state
- one coordinator decides what to surface to the user and which blocked task a reply resumes

This is different from both core modes:

- `consult` is a one-shot delegated call
- `transfer` is exclusive reply ownership for one active agent
- `swarm` is many concurrent task-level agent executions under one coordinating session

The current SDK should not overload `transfer` to mean swarm. If swarm is introduced,
it should be modeled as a higher-level coordination layer built on top of the core
agent/session model.

## API Boundaries

### Core Runtime Path

Model-driven consult and transfer belong to the core `agent` package.

The normal setup is:

1. Create `agent.Agent` definitions.
2. Declare subagents on the calling agent.
3. Register definitions in a `SubagentResolver`, usually `agent.Registry`.
4. Create `agent.Runtime` with `RuntimeConfig.Agent` and `RuntimeConfig.SubagentResolver`.
5. Let the model choose synthetic tools such as `consult_<subagent>` or `transfer_to_<subagent>`.

`multiagent.Router.Register(...)` is not required for this path.

### Package Paths

The `multiagent` package is a compatibility and router/discovery layer.

- `Router.Register(...)` registers agents for tag lookup and compatibility orchestrator calls.
- `Orchestrator.Consult(...)` and `Orchestrator.Transfer(...)` are API-driven helpers where application code has already selected the target agent.
- `workflow.WorkflowAgentTool(...)` is for deterministic application/workflow-owned delegation.

Do not treat router registration as the mechanism that exposes model-driven
subagent tools during `Runtime.Run` or `Runtime.Stream`.

## Ownership Model

### Session Scope

Session scope is the correct ownership boundary for long-lived state.

- A session may involve many registered agents.
- One session must not leak state into another.
- If persistent child state is ever introduced, it must be keyed by session, not just agent name.

### Agent Scope

Agent names identify registered capabilities, not globally shared runtime state.

Reusing a child runtime keyed only by agent name is unsafe in a session-isolated server because:

- two sessions may invoke the same agent name
- history can bleed across sessions
- reply ownership becomes ambiguous

## Runtime Model

### Consult Runtime Model

Consult-style sub-agents should use a fresh runtime per invocation.

The expected lifecycle is:

1. Resolve agent definition by name.
2. Create a fresh runtime for that invocation.
3. Pass only explicit input/context.
4. Run the delegated task.
5. Return the result to the caller.
6. Discard the runtime after completion.

This keeps consult semantics close to tool invocation.

The child runtime should be built from the target agent definition, not from an empty registry that is wired ad hoc by the orchestrator.

### Transfer Runtime Model

Transfer should not be modeled as a generic reusable child-runtime map.

Instead:

- the session tracks which agent currently owns the user-facing reply
- transfer changes that session-level active agent
- the callee's reply becomes the outward reply for that turn

If persistent state is needed for transfer in the future, it must be scoped by:

- session ID
- agent name

not by agent name alone.

### Swarm Runtime Model

If swarm support is added later, it should use a coordinator/task model rather than
pretending that multiple transferred agents all own the same user-facing reply path
at once.

The expected shape is:

- one session-level coordinator
- many agent tasks that may run in parallel
- each task has:
  - assigned agent
  - local working context
  - lifecycle state
  - optional wait conditions such as input or approval
- at any moment, the coordinator still controls what is asked or shown to the user

That means:

- `transfer` should remain exclusive at the session reply-ownership level
- parallelism belongs to consult-style or task-style swarm work
- interactive parallel agents should be represented as tracked tasks, not as many
  simultaneous active transferred owners

## Context Passing

Context passed to a sub-agent should always be explicit.

Examples:

- the current task string
- a summary prepared by the caller
- selected prior messages
- selected metadata

The framework should not assume that a consult-style sub-agent automatically sees the full parent session transcript.

For future swarm support, the same principle applies:

- each task should receive an explicit context projection
- not every agent task should automatically inherit the same full transcript view
- interactive tasks may need different context slices depending on their role

## Current Codebase Implications

The current `multiagent` implementation should be interpreted as consult-style behavior by default.

If it creates fresh runtimes per invocation, that is acceptable for consult semantics.

What should be avoided is:

- implying runtime reuse when there is none
- storing child runtimes in a way that suggests global long-lived ownership by agent name

Longer term, the core model should look like:

- define agents with their own tools and subagents
- create or restore a session separately
- bind an agent and a session into a run

That makes multi-agent orchestration an extension of the core agent model rather than a separate ownership model.

## Recommended API Shape

The multi-agent layer should eventually expose distinct operations:

- `Consult(...)`
- `ConsultStream(...)`
- `Transfer(...)`
- `TransferStream(...)`

They should not be collapsed into one vague "run sub-agent" operation because they have different ownership and reply semantics.

Selection model:

- prefer explicit subagent choice by name
- use router/tag lookup only to discover candidates when needed
- do not treat the router as the owner of orchestration intelligence

At the type level, the natural direction is:

- `Agent` owns tools and subagents
- `Session` owns state
- runtime execution binds `Agent + Session`

Whether that ultimately lives in `agent` directly or behind a thin `multiagent` layer is an implementation packaging decision; the ownership model should be the same either way.

## Summary

The intended model is:

- registered agents are global capabilities
- sessions are isolated execution/state boundaries
- consult is ephemeral delegation
- transfer is session-level reply ownership change
- swarm is a separate coordinator/task orchestration layer for parallel agent work
- no implicit global child-runtime reuse by agent name
