package llm

import (
	"context"
	"fmt"
	"time"
)

type ProviderMiddleware func(Provider) Provider

type RateLimiter interface {
	Wait(ctx context.Context, req *Request) error
}

type RetryPolicy interface {
	ShouldRetry(ctx context.Context, req *Request, err error, attempt int) (wait time.Duration, retry bool)
}

func WrapProvider(base Provider, mw ...ProviderMiddleware) Provider {
	for i := len(mw) - 1; i >= 0; i-- {
		base = mw[i](base)
	}
	return base
}

func WithRateLimit(limiter RateLimiter) ProviderMiddleware {
	return func(next Provider) Provider {
		return &rateLimitedProvider{
			next:    next,
			limiter: limiter,
		}
	}
}

func WithConcurrencyLimit(maxInFlight int) ProviderMiddleware {
	if maxInFlight <= 0 {
		panic("llm: maxInFlight must be > 0")
	}

	return func(next Provider) Provider {
		return &concurrencyLimitedProvider{
			next: next,
			sem:  make(chan struct{}, maxInFlight),
		}
	}
}

func WithRetry(policy RetryPolicy) ProviderMiddleware {
	return func(next Provider) Provider {
		return &retryingProvider{
			next:   next,
			policy: policy,
		}
	}
}

type rateLimitedProvider struct {
	next    Provider
	limiter RateLimiter
}

var _ Provider = (*rateLimitedProvider)(nil)

func (p *rateLimitedProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	if err := p.limiter.Wait(ctx, req); err != nil {
		return nil, fmt.Errorf("llm: rate limit wait: %w", err)
	}
	return p.next.Complete(ctx, req)
}

func (p *rateLimitedProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	if err := p.limiter.Wait(ctx, req); err != nil {
		return nil, fmt.Errorf("llm: rate limit wait: %w", err)
	}
	return p.next.Stream(ctx, req)
}

func (p *rateLimitedProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return p.next.Models(ctx)
}

func (p *rateLimitedProvider) Name() string {
	return p.next.Name()
}

type concurrencyLimitedProvider struct {
	next Provider
	sem  chan struct{}
}

var _ Provider = (*concurrencyLimitedProvider)(nil)

func (p *concurrencyLimitedProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	release, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer release()
	return p.next.Complete(ctx, req)
}

func (p *concurrencyLimitedProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	release, err := p.acquire(ctx)
	if err != nil {
		return nil, err
	}

	source, err := p.next.Stream(ctx, req)
	if err != nil {
		release()
		return nil, err
	}

	out := make(chan Event, 64)
	go func() {
		defer close(out)
		defer release()
		for event := range source {
			out <- event
		}
	}()
	return out, nil
}

func (p *concurrencyLimitedProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return p.next.Models(ctx)
}

func (p *concurrencyLimitedProvider) Name() string {
	return p.next.Name()
}

func (p *concurrencyLimitedProvider) acquire(ctx context.Context) (func(), error) {
	select {
	case <-ctx.Done():
		return nil, fmt.Errorf("llm: concurrency limit cancelled: %w", ctx.Err())
	case p.sem <- struct{}{}:
		return func() { <-p.sem }, nil
	}
}

type retryingProvider struct {
	next   Provider
	policy RetryPolicy
}

var _ Provider = (*retryingProvider)(nil)

func (p *retryingProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	attempt := 0
	for {
		resp, err := p.next.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		wait, retry := p.policy.ShouldRetry(ctx, req, err, attempt)
		if !retry {
			return nil, err
		}
		if err := waitForRetry(ctx, wait); err != nil {
			return nil, err
		}
		attempt++
	}
}

func (p *retryingProvider) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	attempt := 0
	for {
		stream, err := p.next.Stream(ctx, req)
		if err == nil {
			return stream, nil
		}
		wait, retry := p.policy.ShouldRetry(ctx, req, err, attempt)
		if !retry {
			return nil, err
		}
		if err := waitForRetry(ctx, wait); err != nil {
			return nil, err
		}
		attempt++
	}
}

func (p *retryingProvider) Models(ctx context.Context) ([]ModelInfo, error) {
	return p.next.Models(ctx)
}

func (p *retryingProvider) Name() string {
	return p.next.Name()
}

func waitForRetry(ctx context.Context, wait time.Duration) error {
	if wait <= 0 {
		select {
		case <-ctx.Done():
			return fmt.Errorf("llm: retry cancelled: %w", ctx.Err())
		default:
			return nil
		}
	}

	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return fmt.Errorf("llm: retry cancelled: %w", ctx.Err())
	case <-timer.C:
		return nil
	}
}
