# kaeAgent Public

`kaeAgent Public` is a Go SDK for building LLM-powered agents. It includes core
agent runtime primitives, tool calling, streaming responses, conversation memory,
context compaction, persistence, multi-agent consult/transfer workflows, and
observability integrations.

The module path is:

```text
github.com/yourorg/agent-sdk
```

## What This Repo Contains

- Core agent runtime with sessions, conversations, tools, middleware, and budget
  guards.
- LLM provider adapters for OpenAI, Anthropic Claude, Google Gemini, and Qwen.
- Tool registry and dispatcher support, including an HTTP tool.
- Multi-agent patterns for consulting subagents and transferring reply ownership.
- Streaming response support with event handling and token/cost tracking.
- Conversation storage implementations for in-memory, file, SQLite, and SQL use.
- Context compaction strategies for keeping long conversations within model
  limits.
- Observability support through a local tracer abstraction and OpenTelemetry.
- Runnable examples that demonstrate basic agents, chat, streaming, persistence,
  compaction, tracing, and multi-agent workflows.

## Requirements

- Go `1.25` or newer, matching `go.mod`.
- At least one supported provider API key if you want to run examples that call an
  actual model.

Supported provider environment variables:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
DASHSCOPE_API_KEY=...
```

Examples select a provider based on the available API keys. Most examples prefer
OpenAI first, then Claude, Gemini, or Qwen depending on the specific example.

## Setup

Clone the repo and enter the project directory:

```bash
git clone <repo-url>
cd kaeAgent-Public
```

Install Go dependencies:

```bash
go mod download
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` and add at least one provider API key.

## Run Tests

Run the full test suite:

```bash
go test ./...
```

Run focused package tests:

```bash
go test ./agent ./multiagent ./tools
go test ./compaction/...
go test ./store/...
go test ./observability/...
```

## Run The Basic Example

The simplest example creates an agent with time/date tools and asks it to call
those tools.

```bash
go run ./examples/basic
```

You can also pass a prompt:

```bash
go run ./examples/basic "Give me a UTC clock summary by calling current_time, current_date, and unix_timestamp."
```

## Run Chat Examples

Interactive chat:

```bash
go run ./examples/chat
```

Streaming interactive chat:

```bash
go run ./examples/chat-stream
```

Chat with file-backed memory:

```bash
go run ./examples/chat-memory
```

Chat with database-backed storage:

```bash
go run ./examples/chat-db
```

Compaction examples:

```bash
go run ./examples/chat-compaction
go run ./examples/chat-summary-compaction
```

Inside the interactive examples, common commands include:

```text
/usage
/history
/clear
/trace
/quit
```

Some examples also support persistence commands such as:

```text
/save
/sessions
```

## Resume A Saved Session

Some examples read `SESSION_ID` from the environment. To resume a stored
conversation, set it before running the example:

```bash
SESSION_ID=sess_1234567890 go run ./examples/chat-stream
```

File-backed sessions are usually written under `./data/...`.

## Run Multi-Agent Examples

Consult-style multi-agent chat:

```bash
go run ./examples/multiagent/consult-chat
```

Transfer-style multi-agent chat:

```bash
go run ./examples/multiagent/transfer-chat
```

Workflow-owned agent tool example:

```bash
go run ./examples/workflow-agent-tool
```

Additional multi-agent observability examples:

```bash
go run ./examples/multiagent/langfuse-consult-chat
go run ./examples/multiagent/ecommerce-mlflow
go run ./examples/multiagent/ecommerce-transfer-mlflow
go run ./examples/multiagent/ecommerce-transfer-langfuse
go run ./examples/multiagent/travel-opik
```

## Run Tracing Examples

The repo supports tracing through `observability` and `observability/otel`.

Stdout tracing:

```bash
TRACE_BACKEND=stdout go run ./examples/traced
```

Jaeger or another OTLP endpoint:

```bash
TRACE_BACKEND=jaeger OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 go run ./examples/traced
```

Langfuse, MLflow, and Opik examples need their related environment variables.
See `.env.example` and `observability/README.md` for the expected values.

## Key Concepts

### Agent Runtime

The `agent` package owns the main runtime. A runtime combines:

- an LLM provider
- a session
- conversation memory
- tools and dispatcher
- middleware
- optional stores
- optional tracing
- optional subagent resolver

### Tools

Tools are registered through `tools.Registry` and executed through
`tools.Dispatcher`. Tool handlers receive structured input as `map[string]any`
and return any JSON-serializable result.

### Multi-Agent Consult And Transfer

Consult means one agent calls a subagent for help, receives a result, and still
owns the final user response.

Transfer means one agent hands reply ownership to another agent. The new active
agent can continue the conversation in later turns.

### Memory And Compaction

Conversation memory is retained in `agent.Conversation` and can be persisted
through stores. Compaction is separate from persistence and is used to reduce the
prompt footprint before or after model calls.

Built-in compaction strategies include sliding window, turn window, token limit,
and summary.

## Useful Docs

- `agent.md` - contributor and coding-agent guide for this repo.
- `docs/multiagent-architecture.md` - consult and transfer architecture.
- `docs/memory-sequence.md` - runtime memory and resume flow.
- `docs/compaction.md` - compaction model and built-in strategies.
- `observability/README.md` - tracing setup and span structure.

## Repository Layout

```text
agent/          core agent runtime
llm/            LLM provider adapters
tools/          tool definitions, registry, dispatcher
multiagent/     workflow-oriented multi-agent layer
streaming/      streaming events and budget tracking
compaction/     context compaction triggers and strategies
store/          persistence backends
schema/         JSON schema helpers
observability/  tracing abstraction and OTel bridge
examples/       runnable examples
docs/           architecture notes
```
