// examples/basic/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found, relying on system env")
	}
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	provider, err := resolveProvider()
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	provider = wrapProvider(provider)
	fmt.Printf("Using provider: %s\n", provider.Name())

	registry := tools.NewRegistry()
	registry.Register(tools.NewHTTPTool())
	registry.Register(currentTimeTool())
	registry.Register(currentDateTool())
	registry.Register(unixTimestampTool())

	dispatcher := tools.NewDispatcher(registry)

	budget := streaming.NewBudget(streaming.BudgetConfig{
		MaxTokens:          100000,
		MaxCostUSD:         1.00,
		CostPerInputToken:  0.000003,
		CostPerOutputToken: 0.000015,
	})

	session := agent.NewSession(agent.SessionConfig{
		Model:        modelForProvider(provider.Name()),
		SystemPrompt: "You are a helpful assistant. Use tools when needed to answer questions.",
		MaxTokens:    4096,
		Temperature:  float32Ptr(0.7),
		TrimStrategy: agent.TrimSlidingWindow,
		MaxHistory:   50,
		BudgetConfig: &streaming.BudgetConfig{
			MaxTokens:  100000,
			MaxCostUSD: 1.00,
		},
	})

	tracer := observability.NoopTracer{}

	middleware := []agent.Middleware{
		agent.CostGuardMiddleware(budget),
		agent.RetryMiddleware(3, 500*time.Millisecond),
	}

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:           provider,
		Session:            session,
		Tools:              registry,
		Dispatcher:         dispatcher,
		MaxToolConcurrency: 2,
		Middleware:         middleware,
		MaxSteps:           15,
		Tracer:             tracer,
	})

	userMsg := "Give me a UTC clock summary by calling current_time, current_date, and unix_timestamp."
	if len(os.Args) > 1 {
		userMsg = os.Args[1]
	}

	fmt.Println("Example tools: current_time, current_date, unix_timestamp")
	fmt.Println("MaxToolConcurrency: 2 per run")
	fmt.Printf("User: %s\n", userMsg)
	response, err := rt.Run(ctx, userMsg)
	if err != nil {
		log.Fatalf("Agent error: %v", err)
	}
	fmt.Printf("Assistant: %s\n", response)

	input, output, total, cost := budget.Usage()
	fmt.Printf("\nToken usage: %d input, %d output, %d total (est. $%.4f)\n", input, output, total, cost)
}

func float32Ptr(v float32) *float32 {
	return &v
}

func wrapProvider(provider llm.Provider) llm.Provider {
	return llm.WrapProvider(
		provider,
		llm.WithRateLimit(&intervalLimiter{interval: 100 * time.Millisecond}),
		llm.WithConcurrencyLimit(4),
		llm.WithRetry(retryPolicy{maxAttempts: 2, backoff: 250 * time.Millisecond}),
	)
}

func resolveProvider() (llm.Provider, error) {
	if os.Getenv("OPENAI_API_KEY") != "" {
		return llm.NewOpenAIProvider()
	}
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		return llm.NewClaudeProvider()
	}
	if os.Getenv("GEMINI_API_KEY") != "" {
		return llm.NewGeminiProvider()
	}
	if os.Getenv("DASHSCOPE_API_KEY") != "" {
		return llm.NewQwenProvider()
	}
	return nil, fmt.Errorf("no API key found; set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, or DASHSCOPE_API_KEY")
}

func modelForProvider(name string) string {
	switch name {
	case "openai":
		return "gpt-4o"
	case "anthropic":
		return "claude-sonnet-4-20250514"
	case "gcp.gemini":
		return "gemini-1.5-pro"
	case "qwen":
		return "qwen-plus"
	default:
		return "gpt-4o"
	}
}

func currentTimeTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "current_time",
		Description: "Returns the current date and time in UTC",
		Tags:        []string{"utility", "read-only"},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			if err := simulateToolWork(ctx, "current_time"); err != nil {
				return nil, err
			}
			return map[string]any{
				"time":     time.Now().UTC().Format(time.RFC3339),
				"timezone": "UTC",
			}, nil
		},
	}
}

func currentDateTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "current_date",
		Description: "Returns today's date in UTC",
		Tags:        []string{"utility", "read-only"},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			if err := simulateToolWork(ctx, "current_date"); err != nil {
				return nil, err
			}
			return map[string]any{
				"date":     time.Now().UTC().Format("2006-01-02"),
				"timezone": "UTC",
			}, nil
		},
	}
}

func unixTimestampTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "unix_timestamp",
		Description: "Returns the current Unix timestamp in UTC",
		Tags:        []string{"utility", "read-only"},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			if err := simulateToolWork(ctx, "unix_timestamp"); err != nil {
				return nil, err
			}
			return map[string]any{
				"unix":     time.Now().UTC().Unix(),
				"timezone": "UTC",
			}, nil
		},
	}
}

func simulateToolWork(ctx context.Context, name string) error {
	fmt.Printf("[tool start] %s\n", name)
	timer := time.NewTimer(750 * time.Millisecond)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		fmt.Printf("[tool cancel] %s: %v\n", name, ctx.Err())
		return ctx.Err()
	case <-timer.C:
		fmt.Printf("[tool done]  %s\n", name)
		return nil
	}
}

type intervalLimiter struct {
	mu       sync.Mutex
	interval time.Duration
	last     time.Time
}

func (l *intervalLimiter) Wait(ctx context.Context, _ *llm.Request) error {
	for {
		l.mu.Lock()
		now := time.Now()
		wait := l.interval - now.Sub(l.last)
		if wait <= 0 {
			l.last = now
			l.mu.Unlock()
			return nil
		}
		l.mu.Unlock()

		timer := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

type retryPolicy struct {
	maxAttempts int
	backoff     time.Duration
}

func (p retryPolicy) ShouldRetry(ctx context.Context, _ *llm.Request, _ error, attempt int) (time.Duration, bool) {
	select {
	case <-ctx.Done():
		return 0, false
	default:
	}
	return p.backoff, attempt < p.maxAttempts
}
