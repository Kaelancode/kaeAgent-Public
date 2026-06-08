package engine

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

func TestExecuteTurnFinalResponse(t *testing.T) {
	var applied []TurnStep
	var started []int
	var completed []int
	var usage llm.Usage

	out, err := ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "hello",
		State:       State{SessionID: "session"},
		Config:      Config{MaxSteps: 3},
		Hooks: Hooks{
			CheckBudget: func() error { return nil },
			AddUsage: func(got llm.Usage) {
				usage = got
			},
			CurrentStep: func(stepIndex int) StepInput {
				return StepInput{StepIndex: stepIndex, SessionID: "session"}
			},
			ExecuteTurnStep: func(_ context.Context, step StepInput) (TurnStepOutput, error) {
				return TurnStepOutput{
					StepOutput: StepOutput{
						Text:  "done",
						Usage: llm.Usage{InputTokens: 2, OutputTokens: 3},
					},
				}, nil
			},
			ApplyTurnStep: func(_ context.Context, step TurnStep) error {
				applied = append(applied, step)
				return nil
			},
			SnapshotState: func() State {
				return State{SessionID: "updated"}
			},
			StepStarted: func(step StepInput) {
				started = append(started, step.StepIndex)
			},
			StepCompleted: func(stepIndex int, _ *TurnStepOutput, _ error) {
				completed = append(completed, stepIndex)
			},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteTurn: %v", err)
	}
	if out.FinalText != "done" || out.State.SessionID != "updated" {
		t.Fatalf("unexpected turn output: %+v", out)
	}
	if len(out.Commands) != 7 || out.Commands[0].Kind != CommandAppendUserMessage {
		t.Fatalf("expected user plus final commands, got %+v", out.Commands)
	}
	if len(applied) != 2 || applied[0].Index != -1 || applied[1].Action != StepActionFinal {
		t.Fatalf("unexpected applied steps: %+v", applied)
	}
	if len(started) != 1 || started[0] != 0 || len(completed) != 1 || completed[0] != 0 {
		t.Fatalf("unexpected step callbacks: started=%v completed=%v", started, completed)
	}
	if usage.InputTokens != 2 || usage.OutputTokens != 3 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestEngineExecuteTurnMethod(t *testing.T) {
	out, err := New().ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "hello",
		Config:      Config{MaxSteps: 1},
		Hooks: Hooks{
			CheckBudget: func() error { return nil },
			AddUsage:    func(llm.Usage) {},
			CurrentStep: func(stepIndex int) StepInput {
				return StepInput{StepIndex: stepIndex}
			},
			ExecuteTurnStep: func(context.Context, StepInput) (TurnStepOutput, error) {
				return TurnStepOutput{StepOutput: StepOutput{Text: "done"}}, nil
			},
			ApplyTurnStep: func(context.Context, TurnStep) error { return nil },
		},
	})
	if err != nil {
		t.Fatalf("ExecuteTurn: %v", err)
	}
	if out.FinalText != "done" {
		t.Fatalf("unexpected final text: %q", out.FinalText)
	}
}

func TestNilEngineExecuteTurnFails(t *testing.T) {
	var engine *Engine
	_, err := engine.ExecuteTurn(context.Background(), TurnInput{})
	if !IsTurnFailure(err, TurnFailureInvalidRun) || !strings.Contains(err.Error(), "engine is nil") {
		t.Fatalf("expected nil engine failure, got %v", err)
	}
}

func TestExecuteTurnContinuesAfterToolsAndTransfer(t *testing.T) {
	stepOutputs := []TurnStepOutput{
		{
			StepOutput: StepOutput{
				ToolCalls: []tools.ToolCall{{ID: "call_tool", Name: "lookup"}},
			},
		},
		{
			Transfer: &TransferPlan{
				Call:        tools.ToolCall{ID: "call_transfer", Name: "transfer_to_billing"},
				TargetAgent: "billing",
				Input:       "billing issue",
			},
		},
		{StepOutput: StepOutput{Text: "resolved"}},
	}
	var actions []StepAction

	out, err := ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "help",
		Config:      Config{MaxSteps: 4},
		Hooks: Hooks{
			CheckBudget: func() error { return nil },
			AddUsage:    func(llm.Usage) {},
			CurrentStep: func(stepIndex int) StepInput {
				return StepInput{StepIndex: stepIndex}
			},
			ExecuteTurnStep: func(_ context.Context, step StepInput) (TurnStepOutput, error) {
				return stepOutputs[step.StepIndex], nil
			},
			ApplyTurnStep: func(_ context.Context, step TurnStep) error {
				if step.Index >= 0 {
					actions = append(actions, step.Action)
				}
				return nil
			},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteTurn: %v", err)
	}
	if out.FinalText != "resolved" {
		t.Fatalf("unexpected final text: %q", out.FinalText)
	}
	want := []StepAction{StepActionTools, StepActionTransfer, StepActionFinal}
	if len(actions) != len(want) {
		t.Fatalf("unexpected actions: %v", actions)
	}
	for i := range want {
		if actions[i] != want[i] {
			t.Fatalf("action %d: expected %q, got %q", i, want[i], actions[i])
		}
	}
}

func TestExecuteTurnBudgetFailureDoesNotAppendUser(t *testing.T) {
	want := errors.New("token limit")
	applied := false

	_, err := ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "rejected",
		Config:      Config{MaxSteps: 1},
		Hooks: Hooks{
			CheckBudget:     func() error { return want },
			AddUsage:        func(llm.Usage) {},
			CurrentStep:     func(int) StepInput { return StepInput{} },
			ExecuteTurnStep: func(context.Context, StepInput) (TurnStepOutput, error) { return TurnStepOutput{}, nil },
			ApplyTurnStep: func(context.Context, TurnStep) error {
				applied = true
				return nil
			},
		},
	})
	if !IsTurnFailure(err, TurnFailureBudget) || !errors.Is(err, want) {
		t.Fatalf("expected budget failure, got %v", err)
	}
	if applied {
		t.Fatal("budget failure must not append the user message")
	}
}

func TestExecuteTurnStepFailureReportsCompletion(t *testing.T) {
	want := errors.New("provider down")
	var completedErr error

	_, err := ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "hello",
		Config:      Config{MaxSteps: 1},
		Hooks: Hooks{
			CheckBudget: func() error { return nil },
			AddUsage:    func(llm.Usage) {},
			CurrentStep: func(stepIndex int) StepInput { return StepInput{StepIndex: stepIndex} },
			ExecuteTurnStep: func(context.Context, StepInput) (TurnStepOutput, error) {
				return TurnStepOutput{}, want
			},
			ApplyTurnStep: func(context.Context, TurnStep) error { return nil },
			StepCompleted: func(_ int, _ *TurnStepOutput, err error) {
				completedErr = err
			},
		},
	})
	if !IsTurnFailure(err, TurnFailureStep) || !errors.Is(err, want) {
		t.Fatalf("expected step failure, got %v", err)
	}
	if !errors.Is(completedErr, want) {
		t.Fatalf("expected completion callback error, got %v", completedErr)
	}
}

func TestExecuteTurnMaxStepsFailure(t *testing.T) {
	_, err := ExecuteTurn(context.Background(), TurnInput{
		UserMessage: "loop",
		Config:      Config{MaxSteps: 2},
		Hooks: Hooks{
			CheckBudget: func() error { return nil },
			AddUsage:    func(llm.Usage) {},
			CurrentStep: func(stepIndex int) StepInput { return StepInput{StepIndex: stepIndex} },
			ExecuteTurnStep: func(context.Context, StepInput) (TurnStepOutput, error) {
				return TurnStepOutput{
					StepOutput: StepOutput{
						ToolCalls: []tools.ToolCall{{ID: "call", Name: "loop"}},
					},
				}, nil
			},
			ApplyTurnStep: func(context.Context, TurnStep) error { return nil },
		},
	})
	if !IsTurnFailure(err, TurnFailureMaxSteps) || !strings.Contains(err.Error(), "max steps (2)") {
		t.Fatalf("expected max steps failure, got %v", err)
	}
}

func TestExecuteTurnRejectsIncompleteHooks(t *testing.T) {
	_, err := ExecuteTurn(context.Background(), TurnInput{Config: Config{MaxSteps: 1}})
	if !IsTurnFailure(err, TurnFailureInvalidRun) || !strings.Contains(err.Error(), "check budget hook is nil") {
		t.Fatalf("expected invalid turn failure, got %v", err)
	}
}
