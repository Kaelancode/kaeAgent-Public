package agent

import (
	"context"
	"errors"
	"fmt"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
)

func (e *runExecutor) executeEngineTurn(userMessage string, handler Handler, output runOutputAdapter, trace *runTraceState) (agentengine.TurnOutput, error) {
	executeStep := func(ctx context.Context, stepInput agentengine.StepInput) (agentengine.TurnStepOutput, error) {
		runResult, err := handler(ctx, stepFromEngineStepInput(stepInput))
		if err != nil {
			return agentengine.TurnStepOutput{}, err
		}
		if runResult == nil {
			return agentengine.TurnStepOutput{}, fmt.Errorf("nil result")
		}
		return engineTurnStepOutput(normalizeStepResult(runResult)), nil
	}
	return e.executeEngineTurnWithStep(userMessage, executeStep, output, trace)
}

func (e *runExecutor) executeEngineStreamingTurn(userMessage string, handler StreamingHandler, output runOutputAdapter, trace *runTraceState, out chan<- streaming.Event) (agentengine.TurnOutput, error) {
	executeStep := func(ctx context.Context, stepInput agentengine.StepInput) (agentengine.TurnStepOutput, error) {
		streamResult, err := handler(ctx, streamingStepFromEngineStepInput(stepInput), out)
		if err != nil {
			return agentengine.TurnStepOutput{}, err
		}
		if streamResult == nil {
			return agentengine.TurnStepOutput{}, fmt.Errorf("nil result")
		}
		return engineTurnStepOutput(normalizeStreamingStepResult(streamResult)), nil
	}
	return e.executeEngineTurnWithStep(userMessage, executeStep, output, trace)
}

func (e *runExecutor) executeEngineTurnWithStep(userMessage string, executeStep agentengine.ExecuteTurnStepFunc, output runOutputAdapter, trace *runTraceState) (agentengine.TurnOutput, error) {
	input := e.engineTurnInput(userMessage)
	input.Hooks.CheckBudget = e.rs.budget.Check
	input.Hooks.AddUsage = func(usage llm.Usage) {
		e.rs.budget.Add(usage.InputTokens, usage.OutputTokens)
	}
	input.Hooks.CurrentStep = func(stepIndex int) agentengine.StepInput {
		return agentengine.BuildStepInput(
			engineStateFromRunState(e.rs),
			e.availableToolDefs(),
			e.rt.provider.Name(),
			stepIndex,
		)
	}
	input.Hooks.ExecuteTurnStep = executeStep
	input.Hooks.ApplyTurnStep = func(ctx context.Context, step agentengine.TurnStep) error {
		return e.applyEngineTurnStep(ctx, step, output, trace)
	}
	input.Hooks.SnapshotState = func() agentengine.State {
		return engineStateFromRunState(e.rs)
	}
	input.Hooks.StepStarted = func(stepInput agentengine.StepInput) {
		e.recordStepStarted(trace, stepFromEngineStepInput(stepInput))
	}
	input.Hooks.StepCompleted = func(stepIndex int, stepOutput *agentengine.TurnStepOutput, stepErr error) {
		if stepErr != nil {
			e.recordStepCompleted(trace, stepIndex, nil, fmt.Errorf("runtime: step %d: %w", stepIndex, stepErr))
			return
		}
		e.recordStepCompleted(trace, stepIndex, runLoopResultFromEngineTurnOutput(*stepOutput), nil)
	}
	return agentengine.New().ExecuteTurn(trace.ctx, input)
}

func engineTurnStepOutput(result *runLoopResult) agentengine.TurnStepOutput {
	if result == nil {
		return agentengine.TurnStepOutput{}
	}
	return agentengine.TurnStepOutput{
		StepOutput: agentengine.StepOutput{
			Response:  result.Response,
			Text:      result.Text,
			ToolCalls: cloneToolCalls(result.ToolCalls),
			Usage:     result.TokensUsed,
		},
		Transfer: transferPlanFromStep(result.Transfer),
	}
}

func runLoopResultFromEngineTurnOutput(output agentengine.TurnStepOutput) *runLoopResult {
	result := &runLoopResult{
		Response:   output.Response,
		ToolCalls:  cloneToolCalls(output.ToolCalls),
		TokensUsed: output.Usage,
		Text:       output.Text,
	}
	if output.Transfer != nil {
		result.Transfer = &TransferStep{
			Call: output.Transfer.Call,
			Request: TransferRequest{
				AgentName: output.Transfer.TargetAgent,
				Input:     output.Transfer.Input,
				Metadata:  cloneStringMap(output.Transfer.Metadata),
			},
		}
	}
	return result
}

func (e *runExecutor) applyEngineTurnStep(ctx context.Context, step agentengine.TurnStep, output runOutputAdapter, trace *runTraceState) error {
	if step.Index < 0 {
		_, err := e.applyEngineCommands(ctx, step.Commands)
		return err
	}

	result := runLoopResultFromEngineTurnOutput(step.Output)
	switch step.Action {
	case agentengine.StepActionFinal:
		return e.executeFinalCommandPlan(ctx, step.Commands, output, trace, result)
	case agentengine.StepActionTools:
		return e.executeToolCommandPlan(ctx, step.Commands, output, trace, step.Index)
	case agentengine.StepActionTransfer:
		return e.executeTransferCommandPlan(ctx, step.Commands, output, trace, result)
	default:
		return fmt.Errorf("runtime: unsupported engine step action %q", step.Action)
	}
}

func runtimeTurnError(err error) (error, error) {
	var failure *agentengine.TurnFailure
	if !errors.As(err, &failure) {
		wrapped := fmt.Errorf("runtime: engine turn: %w", err)
		return wrapped, wrapped
	}

	switch failure.Kind {
	case agentengine.TurnFailureBudget:
		wrapped := fmt.Errorf("runtime: %w", failure.Err)
		return wrapped, wrapped
	case agentengine.TurnFailureContext:
		wrapped := fmt.Errorf("runtime: context cancelled: %w", failure.Err)
		return wrapped, failure.Err
	case agentengine.TurnFailureStep:
		wrapped := fmt.Errorf("runtime: step %d: %w", failure.Step, failure.Err)
		return wrapped, failure.Err
	case agentengine.TurnFailureApply:
		return failure.Err, failure.Err
	case agentengine.TurnFailureMaxSteps:
		wrapped := fmt.Errorf("runtime: max steps (%d) exceeded", failure.MaxSteps)
		return wrapped, wrapped
	default:
		wrapped := fmt.Errorf("runtime: engine turn: %w", failure.Err)
		return wrapped, wrapped
	}
}
