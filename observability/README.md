# Observability

The `observability` package provides a tracing abstraction for distributed tracing backends. The `otel` sub-package bridges to OpenTelemetry.

## Tracer Interface

```go
type Tracer interface {
    StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, Span)
    EndSpan(ctx context.Context, span Span, err error)
    AddEvent(ctx context.Context, span Span, name string, attrs map[string]string)
    SetSpanAttributes(ctx context.Context, span Span, attrs map[string]any)
}
```

- `StartSpan` creates a child span of whatever span is in `ctx`. Returns updated context and opaque `Span`.
- `EndSpan` finishes the span. Pass a non-nil `err` to mark the span as failed.
- `AddEvent` records a timestamped event with string attributes.
- `SetSpanAttributes` sets typed attributes (string, int, int64, float64, bool) on an existing span.

##Built-in Implementations

| Implementation | Package | Use Case |
|---|---|---|
| `NoopTracer` | `observability` | Default; disables all tracing |
| `StdoutTracer` | `observability` | Debug; prints spans and events to any `io.Writer` |
| `otelTracer` | `observability/otel` | Production; bridges to any OTel-compatible backend |

## Setting Up Tracing

### OpenTelemetry (Jaeger, OTLP gRPC)

```go
import oteltracer "github.com/yourorg/agent-sdk/observability/otel"

tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
    Endpoint:     "localhost:4317",
    ServiceName:  "my-service",
    ExporterType: "grpc",
    Insecure:     true,
})
defer shutdown()
tracer := oteltracer.NewTracer(tp, "my-service")
```

### MLflow

```go
tracer, shutdown, err := exampleutil.SetupMLflowTracer(ctx)
defer shutdown()
```

Requires `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` and `MLFLOW_EXPERIMENT_ID` env vars.

### Langfuse

```go
tracer, shutdown, err := exampleutil.SetupLangfuseTracer(ctx)
defer shutdown()
```

Requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` env vars.

### Opik

```go
tracer, shutdown, err := exampleutil.SetupOpikTracer(ctx)
defer shutdown()
```

Requires `OPIK_ENDPOINT` env var.

## Wiring Tracing into Runtime

```go
rt := agent.NewRuntime(agent.RuntimeConfig{
    Provider: provider,
    Agent:    myAgent,
    Tracer:   tracer,
    // ...
})
```

When `Tracer` is `nil`, the runtime skips all tracing calls with zero overhead.

## Span Structure

Each `Run` / `Stream` call produces a trace with this span hierarchy:

```
invoke_agent {agent_name}                     ← root span
├── gen_ai.user.message event                 ← user input
├── langfuse.observation.input attribute      ← structured input (OTel message JSON)
│
├── chat {model}                              ← one per LLM call
│   ├── per-message events                    ← gen_ai.user.message, gen_ai.system.message, etc.
│   ├── gen_ai.input.messages attribute       ← full input in OTel structured format
│   ├── langfuse.observation.input attribute  ← same structured input
│   ├── [if tool calls]: execute_tool {name}  ← one per tool dispatch
│   │   ├── gen_ai.tool.input attribute
│   │   ├── gen_ai.tool.output attribute
│   │   ├── langfuse.observation.input attribute
│   │   └── langfuse.observation.output attribute
│   ├── gen_ai.output.messages attribute      ← response in OTel structured format
│   ├── langfuse.observation.output attribute ← response text or tool call summary
│   └── gen_ai.usage.input_tokens / output_tokens
│
├── [if transfer]: gen_ai.agent.transfer event
├── [if transfer]: invoke_agent {target_name} ← child agent span
│   ├── langfuse.observation.input attribute   ← transfer input
│   ├── chat {model} ...                      ← subsequent LLM calls under child span
│   ├── gen_ai.output.messages + langfuse.observation.output
│   └── EndSpan (child agent span)
│
├── gen_ai.output.messages attribute            ← final root output
├── langfuse.observation.output attribute       ← final root output text
└── EndSpan (root span)
```

### Span Types and Attributes

#### `invoke_agent` span (root or child)

Created at the start of each `Run`/`Stream`. For transfers, a new child `invoke_agent` span is created under the same root span.

| Attribute | Type | Set When |
|---|---|---|
| `gen_ai.operation.name` | string | Span creation |
| `gen_ai.conversation.id` | string | Span creation |
| `gen_ai.agent.name` | string | Span creation (if agent has a name) |
| `gen_ai.agent.id` | string | Span creation (if agent ID set) |
| `session.id` | string | Span creation |
| `user.id` | string | Span creation (if user ID set) |
| `gen_ai.input.messages` | string (JSON) | Span creation |
| `langfuse.observation.input` | string (JSON) | Span creation |
| `gen_ai.usage.input_tokens` | int | Final response |
| `gen_ai.usage.output_tokens` | int | Final response |
| `gen_ai.output.messages` | string (JSON) | Final response |
| `langfuse.observation.output` | string | Final response |

Events:
- `gen_ai.user.message` — at start with `role`, `content`, `gen_ai.input.messages`
- `gen_ai.assistant.message` — at end with `role`, `content`, `gen_ai.output.messages`
- `gen_ai.agent.transfer` — on transfer with `gen_ai.handoff.from_agent`, `gen_ai.handoff.to_agent`, `content`

#### `chat` span

Created for each LLM provider call.

| Attribute | Type | Set When |
|---|---|---|
| `gen_ai.operation.name` | string | Span creation |
| `gen_ai.provider.name` | string | Span creation |
| `gen_ai.request.model` | string | Span creation |
| `gen_ai.conversation.id` | string | Span creation |
| `gen_ai.agent.name` | string | Span creation (if agent has a name) |
| `gen_ai.agent.id` | string | Span creation (if agent ID set) |
| `gen_ai.request.max_tokens` | int | After creation |
| `gen_ai.request.temperature` | float64 | After creation (if set) |
| `gen_ai.request.stream` | bool | After creation |
| `gen_ai.tool.definitions` | string (JSON) | After creation (if tools present) |
| `gen_ai.input.messages` | string (JSON) | After creation |
| `langfuse.observation.input` | string (JSON) | After creation |
| `gen_ai.response.finish_reasons` | []string | After provider response |
| `gen_ai.response.model` | string | After provider response |
| `gen_ai.usage.input_tokens` | int | After provider response |
| `gen_ai.usage.output_tokens` | int | After provider response |
| `gen_ai.output.messages` | string (JSON) | After provider response |
| `langfuse.observation.output` | string | After provider response |

For text responses, `langfuse.observation.output` is the response text.

For tool-call/transfer responses (no text), `langfuse.observation.output` is a summary like `"tool_calls: lookup_product, transfer_to_product_specialist"` and `gen_ai.output.messages` contains structured tool call parts.

Events:
- `gen_ai.user.message`, `gen_ai.system.message`, `gen_ai.assistant.message`, `gen_ai.tool.message` — one per input message

#### `execute_tool` span

Created for each tool dispatch.

| Attribute | Type | Set When |
|---|---|---|
| `gen_ai.operation.name` | string | Span creation |
| `gen_ai.provider.name` | string | Span creation |
| `gen_ai.tool.name` | string | Span creation |
| `gen_ai.tool.call_id` | string | Span creation |
| `gen_ai.conversation.id` | string | Span creation |
| `gen_ai.agent.name` | string | Span creation (if agent has a name) |
| `gen_ai.agent.id` | string | Span creation (if agent ID set) |
| `gen_ai.tool.input` | string (JSON) | After creation |
| `langfuse.observation.input` | string (JSON) | After creation |
| `gen_ai.tool.output` | string | After tool execution |
| `langfuse.observation.output` | string | After tool execution |
| `error.type` | string | On tool error |

Events:
- `gen_ai.tool.message` — at start (input) and end (output)

## Transfer Tracing

When an agent transfers control to another agent during a run:

1. A `gen_ai.agent.transfer` event is recorded on the **root** `invoke_agent` span with `gen_ai.handoff.from_agent` and `gen_ai.handoff.to_agent`.
2. The current agent span (if different from root) is ended.
3. A new child `invoke_agent {target_name}` span is created under the root span.
4. Subsequent LLM calls, tool dispatches, and further transfers happen under this new child span.
5. At run completion, the child agent span is ended before the root span.

This produces a trace like:

```
invoke_agent shop_assistant          ← root span
├── chat gpt-4o                      ← shop_assistant's LLM call (may produce a transfer tool call)
├── execute_tool transfer_to_product_specialist
├── gen_ai.agent.transfer event      ← from=shop_assistant, to=product_specialist
└── invoke_agent product_specialist   ← child span (steps after transfer run here)
    ├── chat gpt-4o                   ← product_specialist's LLM calls
    ├── execute_tool lookup_product
    ├── langfuse.observation.output
    └── EndSpan
├── langfuse.observation.output on root span
└── EndSpan
```

## Backend Compatibility

The SDK emits both OTel GenAI semantic conventions and vendor-specific compatibility attributes:

| Backend | Input Source | Output Source |
|---|---|---|
| Jaeger / generic OTel | `gen_ai.input.messages` | `gen_ai.output.messages` |
| Langfuse | `langfuse.observation.input` | `langfuse.observation.output` |
| MLflow | `gen_ai.input.messages` (via OTLP) | `gen_ai.output.messages` (via OTLP) |
| Opik | `gen_ai.input.messages` (via OTLP) | `gen_ai.output.messages` (via OTLP) |

All backends receive all attributes. The `langfuse.observation.*` attributes are set on every span alongside the OTel GenAI attributes so that Langfuse can populate its `input`/`output` observation fields correctly. Without these, Langfuse shows `output: null`.

## Provider Name Mapping

`llm.Provider.Name()` returns OTel well-known values:

| SDK Class | `Name()` return |
|---|---|
| `OpenAIProvider` | `"openai"` |
| `ClaudeProvider` | `"anthropic"` |
| `GeminiProvider` | `"gcp.gemini"` |
| `QwenProvider` | `"qwen"` |

Custom providers that don't use OTel-standard names are mapped by `otelProviderName()` in `agent/runtime.go`.

## Adding a Custom Tracer

Implement the `observability.Tracer` interface:

```go
type myTracer struct{}

var _ observability.Tracer = (*myTracer)(nil)

func (t *myTracer) StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, observability.Span) {
    // Create span, store in context
}

func (t *myTracer) EndSpan(ctx context.Context, span observability.Span, err error) {
    // Finish span, record error if non-nil
}

func (t *myTracer) AddEvent(ctx context.Context, span observability.Span, name string, attrs map[string]string) {
    // Add timestamped event
}

func (t *myTracer) SetSpanAttributes(ctx context.Context, span observability.Span, attrs map[string]any) {
    // Set typed attributes on existing span
}
```

Pass an instance to `RuntimeConfig.Tracer`. When `Tracer` is `nil`, the runtime skips all tracing calls with zero overhead (the `NoopTracer` is never instantiated — the runtime checks `if r.tracer != nil` before every call).