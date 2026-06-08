package engine

import (
	"context"

	"github.com/Kaelancode/kaeAgent-Public/compaction"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

type TurnInput struct {
	UserMessage string
	State       State
	Config      Config
	Hooks       Hooks
}

type TurnOutput struct {
	FinalText string
	State     State
	Commands  []Command
	Events    []Event
}

type TurnStepOutput struct {
	StepOutput
	Transfer *TransferPlan
}

type TurnStep struct {
	Index    int
	Input    StepInput
	Output   TurnStepOutput
	Action   StepAction
	Commands []Command
}

type StepAction string

const (
	StepActionFinal    StepAction = "final"
	StepActionTools    StepAction = "tools"
	StepActionTransfer StepAction = "transfer"
)

type StepOutput struct {
	Request   *llm.Request
	Response  *llm.Response
	Text      string
	ToolCalls []tools.ToolCall
	Usage     llm.Usage
}

type TransferPlan struct {
	Call           tools.ToolCall
	TargetAgent    string
	Input          string
	Reason         string
	Metadata       map[string]string
	Acknowledgment string
}

type State struct {
	Generation  uint64
	SessionID   string
	RunID       string
	UserID      string
	AgentID     string
	ActiveAgent AgentView
	RootAgent   AgentView
	Messages    []llm.Message
	Metadata    map[string]string
	Budget      BudgetSnapshot
}

type AgentView struct {
	Name         string
	Model        string
	SystemPrompt string
	MaxSteps     int
	Tools        []tools.ToolDef
	Subagents    []string
}

type Config struct {
	Model              string
	MaxTokens          int
	Temperature        *float32
	MaxSteps           int
	MaxToolConcurrency int
	ModelContextLimit  int
	OutputTokenReserve int
}

type BudgetSnapshot struct {
	MaxTokens          int
	MaxCostUSD         float64
	TotalInput         int
	TotalOutput        int
	TotalCostUSD       float64
	CostPerInputToken  float64
	CostPerOutputToken float64
}

type Hooks struct {
	Complete        CompleteFunc
	Stream          StreamFunc
	ExecuteTools    ExecuteToolsFunc
	ResolveSubagent ResolveSubagentFunc
	FilterTransfer  FilterTransferFunc
	Compact         CompactFunc
	Checkpoint      CheckpointFunc
	SaveSession     SaveSessionFunc
	CheckBudget     CheckBudgetFunc
	AddUsage        AddUsageFunc
	CurrentStep     CurrentStepFunc
	ExecuteTurnStep ExecuteTurnStepFunc
	ApplyTurnStep   ApplyTurnStepFunc
	SnapshotState   SnapshotStateFunc
	StepStarted     StepStartedFunc
	StepCompleted   StepCompletedFunc
}

type CompleteFunc func(ctx context.Context, req *llm.Request) (*llm.Response, error)

type StreamFunc func(ctx context.Context, req *llm.Request) (<-chan llm.Event, error)

type ExecuteToolsFunc func(ctx context.Context, step ToolStep) ([]tools.ToolResult, error)

type ResolveSubagentFunc func(ctx context.Context, name string) (AgentView, bool)

type FilterTransferFunc func(ctx context.Context, input TransferInputData) (TransferInputResult, error)

type CompactFunc func(ctx context.Context, input compaction.Input, force bool) (compaction.Output, error)

type CheckpointFunc func(ctx context.Context, state State) error

type SaveSessionFunc func(ctx context.Context, state State) error

type CheckBudgetFunc func() error

type AddUsageFunc func(usage llm.Usage)

type CurrentStepFunc func(stepIndex int) StepInput

type ExecuteTurnStepFunc func(ctx context.Context, step StepInput) (TurnStepOutput, error)

type ApplyTurnStepFunc func(ctx context.Context, step TurnStep) error

type SnapshotStateFunc func() State

type StepStartedFunc func(step StepInput)

type StepCompletedFunc func(stepIndex int, output *TurnStepOutput, err error)

type ToolStep struct {
	StepIndex      int
	Calls          []tools.ToolCall
	MaxConcurrency int
}

type StepInput struct {
	SessionID      string
	RunID          string
	StepIndex      int
	Messages       []llm.Message
	AvailableTools []tools.ToolDef
	ProviderName   string
	UserID         string
	AgentID        string
	AgentName      string
	Metadata       map[string]string
}

type TransferInputData struct {
	FromAgent string
	ToAgent   string
	Input     string
	Reason    string
	Metadata  map[string]string
	Messages  []llm.Message
}

type TransferInputResult struct {
	Input    string
	Messages []llm.Message
	Metadata map[string]string
}

type CommandKind string

const (
	CommandAppendUserMessage      CommandKind = "append_user_message"
	CommandAppendAssistantMessage CommandKind = "append_assistant_message"
	CommandAppendToolResults      CommandKind = "append_tool_results"
	CommandExecuteTools           CommandKind = "execute_tools"
	CommandApplyTransfer          CommandKind = "apply_transfer"
	CommandCheckpoint             CommandKind = "checkpoint"
	CommandSaveSession            CommandKind = "save_session"
	CommandEmitOutput             CommandKind = "emit_output"
	CommandTraceEvent             CommandKind = "trace_event"
)

type Command struct {
	Kind CommandKind
	Data map[string]any
}

type EventKind string

const (
	EventStepStarted   EventKind = "agent.step.started"
	EventStepCompleted EventKind = "agent.step.completed"
	EventToolStarted   EventKind = "agent.tool.started"
	EventToolCompleted EventKind = "agent.tool.completed"
	EventTransfer      EventKind = "gen_ai.agent.transfer"
)

type Event struct {
	Kind EventKind
	Data map[string]any
}
