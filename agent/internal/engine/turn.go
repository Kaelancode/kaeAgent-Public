package engine

import (
	"context"
	"errors"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

type TurnFailureKind string

const (
	TurnFailureBudget     TurnFailureKind = "budget"
	TurnFailureContext    TurnFailureKind = "context"
	TurnFailureStep       TurnFailureKind = "step"
	TurnFailureApply      TurnFailureKind = "apply"
	TurnFailureMaxSteps   TurnFailureKind = "max_steps"
	TurnFailureInvalidRun TurnFailureKind = "invalid_run"
)

type TurnFailure struct {
	Kind     TurnFailureKind
	Step     int
	MaxSteps int
	Err      error
}

type Engine struct{}

func New() *Engine {
	return &Engine{}
}

func (e *TurnFailure) Error() string {
	switch e.Kind {
	case TurnFailureBudget:
		return fmt.Sprintf("engine: budget: %v", e.Err)
	case TurnFailureContext:
		return fmt.Sprintf("engine: context cancelled: %v", e.Err)
	case TurnFailureStep:
		return fmt.Sprintf("engine: step %d: %v", e.Step, e.Err)
	case TurnFailureApply:
		return fmt.Sprintf("engine: apply step %d: %v", e.Step, e.Err)
	case TurnFailureMaxSteps:
		return fmt.Sprintf("engine: max steps (%d) exceeded", e.MaxSteps)
	default:
		return fmt.Sprintf("engine: invalid turn: %v", e.Err)
	}
}

func (e *TurnFailure) Unwrap() error {
	return e.Err
}

func ExecuteTurn(ctx context.Context, input TurnInput) (TurnOutput, error) {
	return New().ExecuteTurn(ctx, input)
}

func (e *Engine) ExecuteTurn(ctx context.Context, input TurnInput) (TurnOutput, error) {
	if e == nil {
		return TurnOutput{}, &TurnFailure{Kind: TurnFailureInvalidRun, Step: -1, Err: fmt.Errorf("engine is nil")}
	}
	if err := validateTurnInput(input); err != nil {
		return TurnOutput{}, &TurnFailure{Kind: TurnFailureInvalidRun, Step: -1, Err: err}
	}
	if err := input.Hooks.CheckBudget(); err != nil {
		return TurnOutput{}, &TurnFailure{Kind: TurnFailureBudget, Step: -1, Err: err}
	}

	appendUser := Command{
		Kind: CommandAppendUserMessage,
		Data: map[string]any{"message": llm.Message{Role: "user", Content: input.UserMessage}},
	}
	if err := input.Hooks.ApplyTurnStep(ctx, TurnStep{
		Index:    -1,
		Commands: []Command{appendUser},
	}); err != nil {
		return TurnOutput{}, &TurnFailure{Kind: TurnFailureApply, Step: -1, Err: err}
	}

	commands := []Command{appendUser}
	for stepIndex := 0; stepIndex < input.Config.MaxSteps; stepIndex++ {
		if err := ctx.Err(); err != nil {
			return TurnOutput{}, &TurnFailure{Kind: TurnFailureContext, Step: stepIndex, Err: err}
		}
		if err := input.Hooks.CheckBudget(); err != nil {
			return TurnOutput{}, &TurnFailure{Kind: TurnFailureBudget, Step: stepIndex, Err: err}
		}

		stepInput := input.Hooks.CurrentStep(stepIndex)
		if input.Hooks.StepStarted != nil {
			input.Hooks.StepStarted(stepInput)
		}

		stepOutput, err := input.Hooks.ExecuteTurnStep(ctx, stepInput)
		if err != nil {
			if input.Hooks.StepCompleted != nil {
				input.Hooks.StepCompleted(stepIndex, nil, err)
			}
			return TurnOutput{}, &TurnFailure{Kind: TurnFailureStep, Step: stepIndex, Err: err}
		}
		if input.Hooks.StepCompleted != nil {
			input.Hooks.StepCompleted(stepIndex, &stepOutput, nil)
		}
		input.Hooks.AddUsage(stepOutput.Usage)

		action := classifyTurnStep(stepOutput)
		stepCommands := PlanStepCommands(stepInput, stepOutput.StepOutput, stepOutput.Transfer)
		turnStep := TurnStep{
			Index:    stepIndex,
			Input:    stepInput,
			Output:   stepOutput,
			Action:   action,
			Commands: stepCommands,
		}
		if err := input.Hooks.ApplyTurnStep(ctx, turnStep); err != nil {
			return TurnOutput{}, &TurnFailure{Kind: TurnFailureApply, Step: stepIndex, Err: err}
		}
		commands = append(commands, stepCommands...)

		if action == StepActionFinal {
			state := input.State
			if input.Hooks.SnapshotState != nil {
				state = input.Hooks.SnapshotState()
			}
			return TurnOutput{
				FinalText: stepOutput.Text,
				State:     state,
				Commands:  commands,
			}, nil
		}
	}

	return TurnOutput{}, &TurnFailure{
		Kind:     TurnFailureMaxSteps,
		Step:     input.Config.MaxSteps,
		MaxSteps: input.Config.MaxSteps,
	}
}

func validateTurnInput(input TurnInput) error {
	switch {
	case input.Config.MaxSteps <= 0:
		return fmt.Errorf("max steps must be greater than zero")
	case input.Hooks.CheckBudget == nil:
		return fmt.Errorf("check budget hook is nil")
	case input.Hooks.AddUsage == nil:
		return fmt.Errorf("add usage hook is nil")
	case input.Hooks.CurrentStep == nil:
		return fmt.Errorf("current step hook is nil")
	case input.Hooks.ExecuteTurnStep == nil:
		return fmt.Errorf("execute turn step hook is nil")
	case input.Hooks.ApplyTurnStep == nil:
		return fmt.Errorf("apply turn step hook is nil")
	default:
		return nil
	}
}

func classifyTurnStep(output TurnStepOutput) StepAction {
	if output.Transfer != nil {
		return StepActionTransfer
	}
	if len(output.ToolCalls) > 0 {
		return StepActionTools
	}
	return StepActionFinal
}

func IsTurnFailure(err error, kind TurnFailureKind) bool {
	var failure *TurnFailure
	return errors.As(err, &failure) && failure.Kind == kind
}
