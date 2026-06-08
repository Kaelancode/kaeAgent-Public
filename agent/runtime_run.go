package agent

import (
	"context"
	"fmt"

	"github.com/Kaelancode/kaeAgent-Public/streaming"
)

func (r *Runtime) Run(ctx context.Context, userMessage string) (string, error) {
	if !r.HasProvider() {
		return "", fmt.Errorf("runtime: provider is nil")
	}

	exec := r.newRunExecutor()
	handler := Chain(exec.executeStep, r.middleware...)
	trace := exec.startRunTrace(ctx, userMessage)
	output := noopRunOutputAdapter{}
	turn, err := exec.executeEngineTurn(userMessage, handler, output, trace)
	if err != nil {
		runtimeErr, traceErr := runtimeTurnError(err)
		_ = output.EmitError(trace.ctx, runtimeErr)
		exec.endRunTrace(trace, traceErr)
		return "", runtimeErr
	}
	exec.endRunTrace(trace, nil)
	return turn.FinalText, nil
}

func (r *Runtime) Stream(ctx context.Context, userMessage string) (<-chan streaming.Event, error) {
	if !r.HasProvider() {
		return nil, fmt.Errorf("runtime: provider is nil")
	}

	exec := r.newRunExecutor()
	handler := ChainStreaming(exec.executeStreamingStep, r.streamMiddleware...)

	out := make(chan streaming.Event, 128)
	go func() {
		defer close(out)
		trace := exec.startRunTrace(ctx, userMessage)
		adapter := streamingRunOutputAdapter{rt: r, out: out}
		defer func() {
			if recovered := recover(); recovered != nil {
				runtimeErr := fmt.Errorf("runtime: stream panic: %v", recovered)
				if emitErr := adapter.EmitError(trace.ctx, runtimeErr); emitErr != nil {
					r.logger.Warn().Err(emitErr).Msg("runtime: failed to emit streaming panic event")
				}
				exec.endRunTrace(trace, runtimeErr)
			}
		}()
		if _, err := exec.executeEngineStreamingTurn(userMessage, handler, adapter, trace, out); err != nil {
			runtimeErr, traceErr := runtimeTurnError(err)
			if emitErr := adapter.EmitError(trace.ctx, runtimeErr); emitErr != nil {
				r.logger.Warn().Err(emitErr).Msg("runtime: failed to emit streaming error event")
			}
			exec.endRunTrace(trace, traceErr)
		} else {
			exec.endRunTrace(trace, nil)
		}
	}()

	return out, nil
}
