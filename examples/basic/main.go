// examples/basic/main.go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/examples/internal/exampleutil"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Println("No .env file found, relying on system env")
	}
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	provider, err := exampleutil.SelectProvider()
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	provider = exampleutil.WrapProvider(provider)
	fmt.Printf("Using provider: %s\n", provider.Name())

	registry := tools.NewRegistry()
	registry.Register(tools.NewHTTPTool())
	registry.Register(exampleutil.CurrentTimeTool())
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
		Model:        exampleutil.ModelForProvider(provider.Name()),
		SystemPrompt: "You are a helpful assistant. Use tools when needed to answer questions.",
		MaxTokens:    4096,
		Temperature:  exampleutil.Float32Ptr(0.7),
		TrimStrategy: agent.TrimSlidingWindow,
		MaxHistory:   50,
		BudgetConfig: &streaming.BudgetConfig{
			MaxTokens:  100000,
			MaxCostUSD: 1.00,
		},
	})

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
