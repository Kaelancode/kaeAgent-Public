package llm

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type recordingLimiter struct {
	mu       sync.Mutex
	calls    int
	requests []*Request
	err      error
}

func (l *recordingLimiter) Wait(_ context.Context, req *Request) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.calls++
	l.requests = append(l.requests, req)
	return l.err
}

type recordingRetryPolicy struct {
	mu       sync.Mutex
	calls    []retryCall
	wait     time.Duration
	attempts int
}

type retryCall struct {
	req     *Request
	err     error
	attempt int
}

func (p *recordingRetryPolicy) ShouldRetry(_ context.Context, req *Request, err error, attempt int) (time.Duration, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.calls = append(p.calls, retryCall{req: req, err: err, attempt: attempt})
	return p.wait, attempt < p.attempts
}

type blockingProvider struct {
	started chan struct{}
	release chan struct{}
}

func (p *blockingProvider) Complete(ctx context.Context, _ *Request) (*Response, error) {
	close(p.started)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-p.release:
		return &Response{}, nil
	}
}

func (p *blockingProvider) Stream(ctx context.Context, _ *Request) (<-chan Event, error) {
	close(p.started)
	out := make(chan Event, 1)
	go func() {
		defer close(out)
		select {
		case <-ctx.Done():
			out <- Event{Kind: EventError, Err: ctx.Err()}
		case <-p.release:
			out <- Event{Kind: EventDone}
		}
	}()
	return out, nil
}

func (p *blockingProvider) Models(_ context.Context) ([]ModelInfo, error) { return nil, nil }
func (p *blockingProvider) Name() string                                  { return "blocking" }

type staticProvider struct{}

func (p *staticProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	return &Response{}, nil
}

func (p *staticProvider) Stream(_ context.Context, _ *Request) (<-chan Event, error) {
	ch := make(chan Event, 1)
	ch <- Event{Kind: EventDone}
	close(ch)
	return ch, nil
}

func (p *staticProvider) Models(_ context.Context) ([]ModelInfo, error) { return nil, nil }
func (p *staticProvider) Name() string                                  { return "static" }

type flakyProvider struct {
	mu          sync.Mutex
	completeErr []error
	streamErr   []error
	completeN   int
	streamN     int
}

func (p *flakyProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	idx := p.completeN
	p.completeN++
	if idx < len(p.completeErr) && p.completeErr[idx] != nil {
		return nil, p.completeErr[idx]
	}
	return &Response{}, nil
}

func (p *flakyProvider) Stream(_ context.Context, _ *Request) (<-chan Event, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	idx := p.streamN
	p.streamN++
	if idx < len(p.streamErr) && p.streamErr[idx] != nil {
		return nil, p.streamErr[idx]
	}
	ch := make(chan Event, 1)
	ch <- Event{Kind: EventDone}
	close(ch)
	return ch, nil
}

func (p *flakyProvider) Models(_ context.Context) ([]ModelInfo, error) { return nil, nil }
func (p *flakyProvider) Name() string                                  { return "flaky" }

func TestWithRateLimitCompleteWaits(t *testing.T) {
	limiter := &recordingLimiter{}
	provider := WrapProvider(&staticProvider{}, WithRateLimit(limiter))

	_, err := provider.Complete(context.Background(), &Request{Model: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if limiter.calls != 1 {
		t.Fatalf("expected 1 limiter call, got %d", limiter.calls)
	}
	if limiter.requests[0].Model != "test" {
		t.Fatalf("expected request model test, got %q", limiter.requests[0].Model)
	}
}

func TestWithRateLimitStreamWaits(t *testing.T) {
	limiter := &recordingLimiter{}
	provider := WrapProvider(&staticProvider{}, WithRateLimit(limiter))

	stream, err := provider.Stream(context.Background(), &Request{Model: "test-stream"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range stream {
	}

	if limiter.calls != 1 {
		t.Fatalf("expected 1 limiter call, got %d", limiter.calls)
	}
	if limiter.requests[0].Model != "test-stream" {
		t.Fatalf("expected request model test-stream, got %q", limiter.requests[0].Model)
	}
}

func TestWithRateLimitReturnsLimiterError(t *testing.T) {
	limiter := &recordingLimiter{err: errors.New("limited")}
	provider := WrapProvider(&staticProvider{}, WithRateLimit(limiter))

	_, err := provider.Complete(context.Background(), &Request{})
	if err == nil || err.Error() != "llm: rate limit wait: limited" {
		t.Fatalf("expected wrapped limiter error, got %v", err)
	}
}

func TestWithConcurrencyLimitCompleteBlocksSecondCall(t *testing.T) {
	first := &blockingProvider{started: make(chan struct{}), release: make(chan struct{})}
	second := &blockingProvider{started: make(chan struct{}), release: make(chan struct{})}
	call := 0
	base := ProviderFunc(func(ctx context.Context, req *Request) (*Response, error) {
		call++
		if call == 1 {
			return first.Complete(ctx, req)
		}
		return second.Complete(ctx, req)
	})

	provider := WrapProvider(base, WithConcurrencyLimit(1))

	firstDone := make(chan error, 1)
	go func() {
		_, err := provider.Complete(context.Background(), &Request{Model: "one"})
		firstDone <- err
	}()
	<-first.started

	secondDone := make(chan error, 1)
	go func() {
		_, err := provider.Complete(context.Background(), &Request{Model: "two"})
		secondDone <- err
	}()

	select {
	case <-second.started:
		t.Fatal("expected second call to be blocked by concurrency limiter")
	case <-time.After(50 * time.Millisecond):
	}

	close(first.release)
	if err := <-firstDone; err != nil {
		t.Fatalf("unexpected first call error: %v", err)
	}

	select {
	case <-second.started:
	case <-time.After(time.Second):
		t.Fatal("expected second call to start after first completed")
	}

	close(second.release)
	if err := <-secondDone; err != nil {
		t.Fatalf("unexpected second call error: %v", err)
	}
}

func TestWithConcurrencyLimitStreamBlocksUntilStreamCloses(t *testing.T) {
	first := &blockingProvider{started: make(chan struct{}), release: make(chan struct{})}
	second := &blockingProvider{started: make(chan struct{}), release: make(chan struct{})}
	call := 0
	base := &sequenceProvider{
		streamFn: func(ctx context.Context, req *Request) (<-chan Event, error) {
			call++
			if call == 1 {
				return first.Stream(ctx, req)
			}
			return second.Stream(ctx, req)
		},
	}

	provider := WrapProvider(base, WithConcurrencyLimit(1))

	firstStream, err := provider.Stream(context.Background(), &Request{Model: "one"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	<-first.started

	secondReady := make(chan struct{})
	go func() {
		defer close(secondReady)
		_, _ = provider.Stream(context.Background(), &Request{Model: "two"})
	}()

	select {
	case <-second.started:
		t.Fatal("expected second stream to be blocked by concurrency limiter")
	case <-time.After(50 * time.Millisecond):
	}

	close(first.release)
	for range firstStream {
	}

	select {
	case <-second.started:
	case <-time.After(time.Second):
		t.Fatal("expected second stream to start after first stream closed")
	}

	close(second.release)
	<-secondReady
}

func TestWithConcurrencyLimitHonorsContextCancellation(t *testing.T) {
	provider := WrapProvider(&blockingProvider{started: make(chan struct{}), release: make(chan struct{})}, WithConcurrencyLimit(1))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := provider.Complete(ctx, &Request{})
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}

func TestWithRetryCompleteRetriesUntilSuccess(t *testing.T) {
	base := &flakyProvider{
		completeErr: []error{errors.New("first"), errors.New("second"), nil},
	}
	policy := &recordingRetryPolicy{attempts: 2}
	provider := WrapProvider(base, WithRetry(policy))

	_, err := provider.Complete(context.Background(), &Request{Model: "retry"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if base.completeN != 3 {
		t.Fatalf("expected 3 complete attempts, got %d", base.completeN)
	}
	if len(policy.calls) != 2 {
		t.Fatalf("expected 2 retry policy calls, got %d", len(policy.calls))
	}
	if policy.calls[0].attempt != 0 || policy.calls[1].attempt != 1 {
		t.Fatalf("unexpected retry attempts: %#v", policy.calls)
	}
}

func TestWithRetryCompleteStopsWhenPolicySaysNo(t *testing.T) {
	base := &flakyProvider{
		completeErr: []error{errors.New("boom")},
	}
	policy := &recordingRetryPolicy{attempts: 0}
	provider := WrapProvider(base, WithRetry(policy))

	_, err := provider.Complete(context.Background(), &Request{})
	if err == nil || err.Error() != "boom" {
		t.Fatalf("expected original error, got %v", err)
	}
	if base.completeN != 1 {
		t.Fatalf("expected 1 complete attempt, got %d", base.completeN)
	}
}

func TestWithRetryStreamRetriesSetupFailureOnly(t *testing.T) {
	base := &flakyProvider{
		streamErr: []error{errors.New("setup failed"), nil},
	}
	policy := &recordingRetryPolicy{attempts: 1}
	provider := WrapProvider(base, WithRetry(policy))

	stream, err := provider.Stream(context.Background(), &Request{Model: "retry-stream"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for range stream {
	}
	if base.streamN != 2 {
		t.Fatalf("expected 2 stream attempts, got %d", base.streamN)
	}
	if len(policy.calls) != 1 || policy.calls[0].attempt != 0 {
		t.Fatalf("unexpected retry policy calls: %#v", policy.calls)
	}
}

func TestWithRetryHonorsContextCancellationDuringBackoff(t *testing.T) {
	base := &flakyProvider{
		completeErr: []error{errors.New("retry me")},
	}
	policy := &recordingRetryPolicy{attempts: 1, wait: time.Second}
	provider := WrapProvider(base, WithRetry(policy))

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := provider.Complete(ctx, &Request{})
		done <- err
	}()
	time.Sleep(50 * time.Millisecond)
	cancel()

	err := <-done
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected cancellation error, got %v", err)
	}
}

type ProviderFunc func(ctx context.Context, req *Request) (*Response, error)

func (f ProviderFunc) Complete(ctx context.Context, req *Request) (*Response, error) {
	return f(ctx, req)
}

func (f ProviderFunc) Stream(_ context.Context, _ *Request) (<-chan Event, error) {
	return nil, nil
}

func (f ProviderFunc) Models(_ context.Context) ([]ModelInfo, error) { return nil, nil }
func (f ProviderFunc) Name() string                                  { return "func" }

type sequenceProvider struct {
	streamFn func(ctx context.Context, req *Request) (<-chan Event, error)
}

func (p *sequenceProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	return &Response{}, nil
}

func (p *sequenceProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	return p.streamFn(ctx, req)
}

func (p *sequenceProvider) Models(_ context.Context) ([]ModelInfo, error) { return nil, nil }
func (p *sequenceProvider) Name() string                                  { return "sequence" }
