package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	agentengine "github.com/Kaelancode/kaeAgent-Public/agent/internal/engine"
)

func TestRuntimeTurnErrorPreservesPublicSemantics(t *testing.T) {
	stepCause := errors.New("provider down")
	tests := []struct {
		name       string
		failure    *agentengine.TurnFailure
		want       string
		traceCause error
	}{
		{
			name:       "budget",
			failure:    &agentengine.TurnFailure{Kind: agentengine.TurnFailureBudget, Err: errors.New("budget: token limit exceeded")},
			want:       "runtime: budget: token limit exceeded",
			traceCause: nil,
		},
		{
			name:       "context",
			failure:    &agentengine.TurnFailure{Kind: agentengine.TurnFailureContext, Err: context.Canceled},
			want:       "runtime: context cancelled: context canceled",
			traceCause: context.Canceled,
		},
		{
			name:       "step",
			failure:    &agentengine.TurnFailure{Kind: agentengine.TurnFailureStep, Step: 2, Err: stepCause},
			want:       "runtime: step 2: provider down",
			traceCause: stepCause,
		},
		{
			name:       "max steps",
			failure:    &agentengine.TurnFailure{Kind: agentengine.TurnFailureMaxSteps, MaxSteps: 3},
			want:       "runtime: max steps (3) exceeded",
			traceCause: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, traceErr := runtimeTurnError(tt.failure)
			if got.Error() != tt.want {
				t.Fatalf("expected %q, got %q", tt.want, got)
			}
			if tt.traceCause != nil && !errors.Is(traceErr, tt.traceCause) {
				t.Fatalf("expected trace error to wrap %v, got %v", tt.traceCause, traceErr)
			}
			if tt.traceCause == nil && !strings.Contains(traceErr.Error(), tt.want) {
				t.Fatalf("expected trace error %q, got %v", tt.want, traceErr)
			}
		})
	}
}

func TestRuntimeTurnErrorReturnsApplyFailureDirectly(t *testing.T) {
	want := errors.New("runtime: checkpoint conversation: failed")
	got, traceErr := runtimeTurnError(&agentengine.TurnFailure{
		Kind: agentengine.TurnFailureApply,
		Err:  want,
	})
	if !errors.Is(got, want) || !errors.Is(traceErr, want) {
		t.Fatalf("expected apply failure to pass through, got runtime=%v trace=%v", got, traceErr)
	}
}
