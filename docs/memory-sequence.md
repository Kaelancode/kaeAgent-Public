# Agent Memory Sequence

This file documents how conversational memory works in this SDK during `Run()` and how resume works.

## Runtime Memory Flow for a Single `Run()` Call

The runtime keeps memory in an in-memory `agent.Conversation`. When `Run()` starts, it appends the new user message to that conversation. For each step, it reads the full current retained message list from the conversation and sends that list to the LLM, so the model sees the entire current conversation buffer rather than only the latest turn.

If the model returns a final assistant response, the runtime appends that assistant message to the conversation, checkpoints the full retained conversation to the configured `ConversationStore`, saves session metadata to `SessionStore` if one exists, and returns the final text. If the model returns tool calls, the runtime appends the assistant tool-call message, checkpoints the conversation, dispatches the tools, appends each tool result as a `role="tool"` message, checkpoints again, and then starts the next step with that updated history.

This means memory is incremental in RAM but full-context on each LLM call: new messages are appended one by one, while each model request receives the full current retained history. The retained history may still be trimmed by the conversation strategy, so what gets sent is the entire current buffer after trimming, not necessarily the entire lifetime history.

## Run Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant R as Runtime
    participant C as Conversation
    participant P as LLM Provider
    participant D as Tool Dispatcher
    participant S as ConversationStore
    participant SS as SessionStore

    U->>R: Run(ctx, userMessage)
    R->>C: Append(user message)
    C-->>R: Updated message history

    loop Each step until final response or maxSteps
        R->>C: Messages()
        C-->>R: Full retained history
        R->>P: Complete(request with full history + tools)

        alt Provider returns final assistant text
            P-->>R: Response(text, usage)
            R->>C: Append(assistant text)
            R->>S: Save(convID, C.Messages())
            opt SessionStore configured
                R->>SS: SaveSession(session snapshot)
            end
            R-->>U: Return final text

        else Provider returns tool calls
            P-->>R: Response(tool calls + optional text)
            R->>C: Append(assistant tool-call message)
            R->>S: Save(convID, C.Messages())

            loop For each tool call
                R->>D: Dispatch(tool call)
                D-->>R: Tool result
                R->>C: Append(tool result as role="tool")
            end

            R->>S: Save(convID, C.Messages())
            Note over R,C: Next step uses updated history,\nincluding tool results
        end
    end
```

## Resume Sequence

```mermaid
sequenceDiagram
    autonumber
    participant App as App or Example
    participant SS as SessionStore
    participant S as ConversationStore
    participant R as Runtime
    participant C as Conversation

    App->>SS: LoadSession(sessionID)
    SS-->>App: Session metadata

    App->>S: Load(sessionID)
    S-->>App: Stored messages

    App->>C: NewConversationFromState(loaded messages)
    App->>R: NewRuntime(Session, Conversation, Stores)

    Note over R,C: Future Run() or Stream() calls continue\nfrom restored message history
```
