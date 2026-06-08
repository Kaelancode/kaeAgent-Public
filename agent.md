# Agent Guide

This repository is a Go SDK for building LLM agents, tools, multi-agent workflows,
conversation memory, compaction, streaming, storage, and observability integrations.

The module path is:

```text
github.com/Kaelancode/kaeAgent-Public
```

## Project Map

- `agent/` - core agent definitions, sessions, runtime execution, middleware,
  streaming accumulation, subagent resolution, consult/transfer behavior, and
  conversation state.
- `llm/` - provider interface plus OpenAI, Anthropic Claude, Gemini, and Qwen
  implementations and provider wrappers for retry, rate limiting, and concurrency.
- `tools/` - tool definitions, registry, dispatcher, HTTP tool, and tool execution
  tests.
- `multiagent/` - workflow-oriented multi-agent orchestration and compatibility
  layer. Core model-driven consult and transfer behavior lives in `agent/`.
- `streaming/` - streaming runner, events, and token/cost budget accounting.
- `compaction/` - prompt/context compaction interfaces, triggers, estimators, and
  strategies.
- `store/` - in-memory, file, and SQL-backed conversation/session persistence.
- `schema/` - JSON schema helpers for structured tool/input definitions.
- `observability/` - tracing abstraction and OpenTelemetry bridge.
- `examples/` - runnable examples for basic agents, tracing, workflow tools, and
  multi-agent consult/transfer scenarios.
- `docs/` - architectural notes for multi-agent behavior, compaction, and memory
  sequencing.

## Development Commands

Use these from the repository root.

```bash
go test ./...
go test ./agent ./multiagent ./tools
go test ./compaction/...
go test ./store/...
go test ./observability/...
go run ./examples/basic "Give me a UTC clock summary."
```

Before sending changes, run `go test ./...` unless the change is documentation-only.

## Environment

Examples load `.env` with `github.com/joho/godotenv` when available, then fall
back to process environment variables.

Useful variables include:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `DASHSCOPE_API_KEY`
- `TRACE_BACKEND`
- `OTEL_EXPORTER_OTLP_ENDPOINT`
- `OTEL_EXPORTER_OTLP_INSECURE`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_BASE_URL`
- `MLFLOW_EXPERIMENT_ID`
- `OPIK_API_KEY`
- `SESSION_ID`

Provider selection in examples usually prefers OpenAI, then Claude, then Gemini,
then Qwen, depending on which API key is present.

## Core Architecture Notes

Agent definitions and session state are separate.

An `agent.Agent` owns durable definition-level configuration such as name, model,
system prompt, tools, subagents, execution limits, and policy settings. A session
owns runtime state such as session ID, conversation history, metadata, and active
replying agent.

The runtime binds:

- an agent definition or session configuration
- a conversation
- stores
- tools and dispatcher
- LLM provider
- optional middleware
- optional tracer
- optional subagent resolver

Conversation memory is incremental in RAM, but each provider call receives the
full currently retained conversation buffer. Runtime appends user, assistant, and
tool messages as the run progresses, checkpoints to configured stores, and may
compact retained messages according to the configured compactor/trim strategy.

## Multi-Agent Rules

There are two core model-driven interaction modes.

Consult is agent-as-tool delegation:

- the current agent invokes a declared subagent for a task
- the subagent returns a result to the caller
- the caller still owns the final response
- consult runs are isolated and do not change `active_agent`

Transfer is direct reply ownership transfer:

- the current agent hands control to a declared target agent
- the target agent may reply directly to the user
- the target can become the active agent for the session
- active ownership is persisted in session metadata under `active_agent`

When an agent declares subagents, the runtime can expose synthetic tools named:

- `consult_<subagent>`
- `transfer_to_<subagent>`

Use `agent.Registry` as the usual `SubagentResolver` for core runtime behavior.
The `multiagent` package is for workflow and compatibility scenarios.

For workflow-owned orchestration, prefer `WorkflowAgentTool(...)`. The older
`AgentTool(...)` name is compatibility-oriented and should not be used for new
workflow-facing code.

## Compaction Model

Compaction is separate from persistence.

- persistence stores retained conversation/session state
- compaction builds or replaces a smaller retained prompt view
- runtime may compact after a turn or force compact before provider calls near a
  model context limit

Core compaction interfaces are:

- `compaction.Compactor`
- `compaction.Trigger`
- `compaction.Strategy`

Built-in trigger types include max turns, max messages, max estimated tokens, and
an any-trigger composition. Built-in strategies include sliding window, turn
window, token limit, and summary.

Prefer turn-aware strategies when tool calls are involved so user turns,
assistant tool calls, tool results, and final assistant replies stay together.

## Testing Guidance

Follow the existing package-local test style. The repo already has focused tests
for runtime behavior, transfer input handling, subagents, provider request bodies,
tool dispatch, storage, compaction strategies, streaming budgets, and tracing.

When changing behavior:

- add or update focused tests in the affected package
- include multi-agent tests for consult/transfer routing changes
- include persistence tests when changing stores or session metadata
- include streaming tests when changing event ordering, budget handling, or stream
  accumulation
- include provider tests when changing request/response translation

Avoid relying on live provider calls in unit tests. Prefer fake providers and
deterministic tool handlers, matching the existing test style.

## Coding Conventions

- Keep package boundaries intact; do not move core consult/transfer behavior into
  `multiagent`.
- Prefer explicit errors for invalid tool payloads, unknown subagents, mixed
  transfer and normal tool calls, and over-limit prompt requests.
- Keep session-scoped state out of global agent definitions.
- Preserve isolation between sessions and child consult runs.
- Use existing registries and dispatcher types instead of ad hoc lookup logic.
- Keep exported API changes small and covered by tests.
- Use `gofmt` on changed Go files.
- Do not commit `.env`, local trace credentials, or generated local data.

## Useful Entry Points

- Basic runtime setup: `examples/basic/main.go`
- Agent definition: `agent/agent.go`
- Runtime execution: `agent/runtime.go`
- Subagent registry and resolver: `agent/registry.go`
- Consult/transfer docs: `docs/multiagent-architecture.md`
- Memory flow: `docs/memory-sequence.md`
- Compaction docs: `docs/compaction.md`
- Observability docs: `observability/README.md`
