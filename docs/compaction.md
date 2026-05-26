# Compaction

This SDK keeps the full retained conversation in `agent.Conversation` and uses the `compaction` package to reduce prompt size when needed.

Compaction is separate from persistence:
- conversation storage keeps the retained transcript
- compaction builds a smaller prompt view
- runtime can replace the retained in-memory message set after compaction

Relevant code:
- [compaction/compaction.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/compaction.go)
- [compaction/trigger.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/trigger.go)
- [compaction/estimator.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/estimator.go)
- [compaction/strategy/slidingwindow/slidingwindow.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/slidingwindow/slidingwindow.go)
- [compaction/strategy/summary/summary.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/summary/summary.go)
- [compaction/strategy/turnwindow/turnwindow.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/turnwindow/turnwindow.go)
- [compaction/strategy/tokenlimit/tokenlimit.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/tokenlimit/tokenlimit.go)
- [agent/runtime.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/agent/runtime.go)

## Mental Model

Compaction has three parts:

1. `Trigger`
Decides when compaction should run.

2. `Strategy`
Decides how messages should be reduced.

3. `Engine`
Combines a trigger and a strategy into one `Compactor`.

The core types are:

```go
type Compactor interface {
    Compact(ctx context.Context, input Input) (Output, error)
    ForceCompact(ctx context.Context, input Input) (Output, error)
}

type Trigger interface {
    Name() string
    ShouldCompact(ctx context.Context, input Input) (bool, string, error)
}

type Strategy interface {
    Name() string
    Compact(ctx context.Context, input Input) (Output, error)
}
```

`Compact(...)` uses the trigger.

`ForceCompact(...)` skips trigger checks and runs the strategy immediately. Runtime uses this for the preflight safety guard near the model context limit.

## How Runtime Uses Compaction

Runtime uses compaction in two places.

### 1. Post-turn compaction

After a final assistant response is appended for a user turn, runtime may compact the retained conversation:
- append assistant message
- run `Compactor.Compact(...)`
- if `Output.Compacted == true`, replace retained messages
- checkpoint to the configured `ConversationStore`

This happens in [agent/runtime.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/agent/runtime.go:742).

### 2. Preflight safety guard

Before a provider call, runtime estimates request size. If:

```text
estimated_input_tokens + output_reserve > model_context_limit
```

runtime force-compacts before sending the request.

This happens in [agent/runtime.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/agent/runtime.go:779).

If compaction still cannot get the request under the hard limit, runtime returns an error instead of sending an unsafe request.

## Token Estimation

The default estimator is `ApproxTokenEstimator` in [compaction/estimator.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/estimator.go).

It estimates prompt tokens by:
- summing message character lengths
- dividing by `DefaultCharsPerToken`, which is `4`

This is approximate, not provider-accurate.

Runtime's preflight request-size estimate is a little broader than the default message estimator. It also includes:
- tool definitions
- tool call serialization
- message wrapper overhead

That logic is in [agent/runtime.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/agent/runtime.go:812).

## Built-in Triggers

Built-in triggers are defined in [compaction/trigger.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/trigger.go).

### `MaxTurnsTrigger`

Compacts when the number of user turns exceeds a limit.

Turn definition:
- each `role == "user"` message counts as one turn
- assistant and tool messages belong to that user turn

This is the best trigger when you want stable conversational boundaries.

### `MaxMessagesTrigger`

Compacts when the raw number of retained messages exceeds a limit.

This is lower-level than turn counting. It counts every message:
- system
- user
- assistant
- tool

### `MaxTokensTrigger`

Compacts when estimated prompt tokens exceed a limit.

This is useful when message lengths vary a lot and you care about prompt footprint more than message count.

### `AnyTrigger`

Compacts if any trigger in the list fires.

This is useful for policies like:
- compact after too many turns
- or compact after prompt estimate gets too large

## Built-in Strategies

Built-in strategies live under `compaction/strategy/...`.

### `slidingwindow`

Implementation:
- [compaction/strategy/slidingwindow/slidingwindow.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/slidingwindow/slidingwindow.go)

How it works:
- preserves all `system` messages
- keeps only the most recent `N` non-system messages
- drops older non-system messages

If you configure `slidingwindow.New(12)`:
- system messages stay
- only the latest 12 non-system messages remain

This is simple and predictable. It is best when recency matters most.

This strategy is still message-based. It is useful when you explicitly want a raw message window rather than a turn window.

Example:

```text
Before:
system
user1
assistant1
user2
assistant2
user3
assistant3

Window = 4

After:
system
user2
assistant2
user3
assistant3
```

### `turnwindow`

Implementation:
- [compaction/strategy/turnwindow/turnwindow.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/turnwindow/turnwindow.go)

How it works:
- preserves all `system` messages
- groups the conversation into turns
- keeps only the latest `N` turns
- drops older complete turns

A turn means:
- one user message
- everything after it until the next user message

So a turn that includes:
- a user query
- assistant tool call
- tool result
- final assistant reply

stays intact or is dropped intact.

If you configure `turnwindow.New(2)`:
- system messages stay
- only the latest 2 turns remain

This is the default runtime strategy used for `TrimSlidingWindow`.

### `summary`

Implementation:
- [compaction/strategy/summary/summary.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/summary/summary.go)

How it works:
- preserves all `system` messages
- keeps the latest `N` complete turns intact
- summarizes older turns into a synthetic summary message
- inserts that summary message before the recent retained turns

Like `turnwindow`, its grouping unit is a complete turn:
- one user message
- everything after it until the next user message

The strategy is provider-agnostic. You can inject your own summarizer function:

```go
type SummarizerFunc func(ctx context.Context, turns [][]llm.Message) (string, error)
```

If you do not provide one, it uses a deterministic default summarizer that converts older turns into a compact textual summary.

Example:

```text
Before:
system
user: "Search for the latest rate"
assistant: <tool call>
tool: "rate = 4.1"
assistant: "The latest rate is 4.1"
user: "Now summarize that"
assistant: "Summary..."
user: "What changed since yesterday?"
assistant: "..."

RecentTurns = 2

After:
system
system: "Conversation summary:
Turn 1: User: Search for the latest rate Tool: rate = 4.1 Assistant: The latest rate is 4.1"
user: "Now summarize that"
assistant: "Summary..."
user: "What changed since yesterday?"
assistant: "..."
```

### `tokenlimit`

Implementation:
- [compaction/strategy/tokenlimit/tokenlimit.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/compaction/strategy/tokenlimit/tokenlimit.go)

How it works:
- preserves system messages
- estimates token usage of the retained messages
- if over budget, removes the oldest complete turn
- repeats until the estimate fits under the configured max

Important detail:
- it is turn-aware
- it does not cut a message mid-sentence
- it does not cut a tool result mid-body
- it does not split a turn across a user query boundary

A turn means:
- one user message
- everything after it until the next user message

So if an old turn contains:
- a user query
- assistant tool call
- tool result
- final assistant reply

that whole turn is dropped together.

This is better than message-by-message dropping because it avoids orphaned tool results or assistant replies without the user query that caused them.

Example:

```text
Before:
system
user: "Search for the latest rate"
assistant: <tool call>
tool: "rate = 4.1"
assistant: "The latest rate is 4.1"
user: "Now summarize that"
assistant: "Summary..."

Budget too large

After:
system
user: "Now summarize that"
assistant: "Summary..."
```

## Strategy Differences

The main difference is what each strategy optimizes for.

### `slidingwindow`

- reduction unit: message
- preserved strongly: recent messages
- predictable size by message count
- simple behavior

Use it when:
- your messages are roughly similar in size
- you care mostly about recency
- you want easy-to-explain behavior

### `turnwindow`

- reduction unit: complete turn
- preserved strongly: recent turns
- stable for tool-heavy conversations
- best semantic match for chat history

Use it when:
- you want a sliding window without splitting turns
- tool calls and tool results should stay attached to the user query that caused them
- you want `MaxHistory` to behave like a turn window instead of a raw message window

### `summary`

- reduction unit: complete turn
- preserved strongly: recent turns plus compressed older context
- best option when you want to keep long-range continuity without keeping raw old turns

Use it when:
- you want recent turns verbatim
- older turns should remain available in compressed form
- a plain deletion strategy loses too much context
- you want to plug in an LLM-based summarizer later without changing runtime architecture

### `tokenlimit`

- reduction unit: complete turn
- preserved strongly: prompt budget
- adapts better when some messages are very large
- avoids splitting user/tool/assistant turn groupings

Use it when:
- message sizes vary a lot
- tool outputs can be large
- you want compaction to respect a token budget

In short:
- `slidingwindow` answers: "How many recent messages should I keep?"
- `summary` answers: "How do I keep recent turns intact while compressing older turns?"
- `tokenlimit` answers: "How much prompt space can I afford?"

## Default Runtime Behavior

If you do not supply a custom `Compactor`, runtime builds a default one from `SessionConfig` in [agent/runtime.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/agent/runtime.go:755).

### `TrimSlidingWindow`

Runtime builds:

```go
compaction.NewEngine(
    compaction.MaxTurnsTrigger{MaxTurns: session.Config.MaxHistory},
    turnwindow.New(session.Config.MaxHistory),
    nil,
)
```

Meaning:
- trigger when user turns exceed `MaxHistory`
- keep the latest `MaxHistory` complete turns

### `TrimTokenCount`

Runtime builds:

```go
softLimit := session.Config.TokenBudget * 80 / 100

compaction.NewEngine(
    compaction.MaxTokensTrigger{MaxTokens: softLimit},
    tokenlimit.New(softLimit, nil),
    nil,
)
```

Meaning:
- trigger when estimated prompt tokens exceed about 80% of the configured token budget
- drop oldest complete turns until the retained context fits the soft limit

## Using Compaction Directly

You can use the package without runtime if you want to compact a message list yourself.

Example:

```go
estimator := compaction.NewApproxTokenEstimator(compaction.DefaultCharsPerToken)

engine := compaction.NewEngine(
    compaction.AnyTrigger{
        compaction.MaxTurnsTrigger{MaxTurns: 6},
        compaction.MaxTokensTrigger{MaxTokens: 3200, Estimator: estimator},
    },
    slidingwindow.New(12),
    estimator,
)

out, err := engine.Compact(ctx, compaction.Input{
    SessionID: "sess_123",
    Messages:  messages,
})
if err != nil {
    return err
}

if out.Compacted {
    messages = out.Messages
}
```

## Using The Summary Strategy

You can use the built-in default summarizer:

```go
strategy := summary.New(2, nil)
```

Or provide your own summarizer:

```go
strategy := summary.New(2, func(ctx context.Context, turns [][]llm.Message) (string, error) {
    return "Older discussion covered provider selection, DB schema, and retry behavior.", nil
})
```

Then wire it into a compactor:

```go
compactor := compaction.NewEngine(
    compaction.MaxTurnsTrigger{MaxTurns: 6},
    summary.New(2, nil),
    nil,
)
```

That means:
- once turns exceed the trigger threshold
- the latest 2 turns remain verbatim
- older turns are replaced by one summary message

## Using Compaction in Runtime

Pass a custom `Compactor` through `agent.RuntimeConfig`.

Example:

```go
estimator := compaction.NewApproxTokenEstimator(compaction.DefaultCharsPerToken)

compactor := compaction.NewEngine(
    compaction.AnyTrigger{
        compaction.MaxTurnsTrigger{MaxTurns: 6},
        compaction.MaxTokensTrigger{MaxTokens: 3200, Estimator: estimator},
    },
    slidingwindow.New(12),
    estimator,
)

rt := agent.NewRuntime(agent.RuntimeConfig{
    Provider:           provider,
    Session:            session,
    Tools:              registry,
    Dispatcher:         dispatcher,
    Compactor:          compactor,
    ModelContextLimit:  4000,
    OutputTokenReserve: 800,
})
```

This gives you:
- post-turn compaction using your trigger and strategy
- preflight force-compaction if the request is near the model context limit

See the runnable example:
- [examples/chat-compaction/main.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/examples/chat-compaction/main.go)
- [examples/chat-summary-compaction/main.go](/home/aiops/Projects/agent_workshop/kaeAgent/agent-sdk-open/examples/chat-summary-compaction/main.go)

## Example Policy

Here is a practical policy:

- post-turn trigger at 80% of prompt budget
- or after too many turns
- preflight hard guard near model context limit

Example:

```go
estimator := compaction.NewApproxTokenEstimator(compaction.DefaultCharsPerToken)

compactor := compaction.NewEngine(
    compaction.AnyTrigger{
        compaction.MaxTurnsTrigger{MaxTurns: 6},
        compaction.MaxTokensTrigger{MaxTokens: 160000, Estimator: estimator},
    },
    tokenlimit.New(160000, estimator),
    estimator,
)

rt := agent.NewRuntime(agent.RuntimeConfig{
    Provider:           provider,
    Session:            session,
    Compactor:          compactor,
    ModelContextLimit:  200000,
    OutputTokenReserve: 20000,
})
```

Behavior:
- after a turn completes, runtime compacts if turns or token threshold exceed the policy
- before a provider call, runtime force-compacts if estimated input plus `20000` output headroom would exceed `200000`

## Current Limitations

Current built-in strategies only drop context. They do not summarize it.

So today compaction can:
- preserve system messages
- preserve recent messages or recent complete turns
- reduce prompt size

But it cannot yet:
- summarize old turns into a condensed memory
- extract durable facts into structured memory
- use a model-specific tokenizer for exact provider token counts

That means:
- `slidingwindow` is recency-preserving deletion
- `tokenlimit` is budget-preserving deletion
- future summarization strategies would be the right place to keep long-range meaning with less prompt cost
