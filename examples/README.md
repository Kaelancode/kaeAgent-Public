# Examples

These examples demonstrate common SDK usage patterns: basic model calls, chat loops, persistence, tracing, compaction, multi-agent consult, multi-agent transfer, and deterministic workflow composition.

## Environment

Copy the template and fill in only the values needed for the example you are running:

```bash
cp examples/.env.example .env
```

The examples read environment variables directly. If your shell does not auto-load `.env`, export the variables manually or source it with your preferred tooling.

At least one provider key is required for examples that call a model:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DASHSCOPE_API_KEY=...
```

Provider base URL variables are optional and are mainly for gateways or compatible endpoints:

```bash
export OPENAI_BASE_URL=...
export GEMINI_BASE_URL=...
```

## Running

Run an example from the repository root:

```bash
go run ./examples/chat
go run ./examples/chat-stream
go run ./examples/multiagent/consult-chat
go run ./examples/multiagent/transfer-chat
go run ./examples/workflow-agent-tool
```

Some examples prompt for a provider at startup. Others choose based on available API keys.

All example packages are continuously compile-checked with:

```bash
go build ./examples/...
go test ./examples/... -count=1
```

Interactive and external tracing examples still require valid provider credentials and, where applicable, reachable Langfuse, MLflow, Opik, Jaeger, or database services.

## Tracing

Examples that use `examples/internal/exampleutil` support:

```bash
export TRACE_BACKEND=stdout
export TRACE_BACKEND=jaeger
export TRACE_BACKEND=langfuse
export TRACE_BACKEND=mlflow
export TRACE_BACKEND=opik
```

Backend-specific variables are listed in [`.env.example`](./.env.example). Langfuse requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. MLflow, Jaeger, and Opik use OTLP-style endpoint settings.

## Persistence

`examples/traced` supports optional persistence:

```bash
export STORE_BACKEND=file
export STORE_DIR=./data
```

## Model Names

Examples may use different valid model names intentionally. Keep example models valid for their provider, but they do not need to be identical across examples.
