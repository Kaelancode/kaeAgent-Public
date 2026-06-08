package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/examples/internal/exampleutil"
	"github.com/Kaelancode/kaeAgent-Public/store"
	storefile "github.com/Kaelancode/kaeAgent-Public/store/file"
	storeinmem "github.com/Kaelancode/kaeAgent-Public/store/inmem"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
	"github.com/joho/godotenv"
)

func main() {
	godotenv.Load()
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	logger := exampleutil.NewLogger()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	provider, err := exampleutil.SelectProvider()
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	fmt.Printf("Using provider: %s\n", provider.Name())

	tracer, shutdown, err := exampleutil.SetupTracerFromEnv(ctx)
	if err != nil {
		log.Fatalf("Failed to setup tracer: %v", err)
	}
	defer shutdown()

	convStore := setupStore()

	registry := tools.NewRegistry()
	registry.Register(tools.NewHTTPTool())
	registry.Register(exampleutil.CurrentTimeTool())

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
		Provider:          provider,
		Session:           session,
		Tools:             registry,
		Dispatcher:        dispatcher,
		Middleware:        middleware,
		MaxSteps:          15,
		Tracer:            tracer,
		ConversationStore: convStore,
		Logger:            logger,
	})

	userMsg := "What is the current time?"
	if len(os.Args) > 1 {
		userMsg = os.Args[1]
	}

	fmt.Printf("User: %s\n", userMsg)
	response, err := rt.Run(ctx, userMsg)
	if err != nil {
		log.Fatalf("Agent error: %v", err)
	}
	fmt.Printf("Assistant: %s\n", response)

	input, output, total, cost := budget.Usage()
	fmt.Printf("\nToken usage: %d input, %d output, %d total (est. $%.4f)\n", input, output, total, cost)
}

func setupStore() store.ConversationStore {
	backend := os.Getenv("STORE_BACKEND")
	switch strings.ToLower(backend) {
	case "file":
		dir := exampleutil.EnvOrDefault("STORE_DIR", "./data")
		s, err := storefile.NewConversationStore(storefile.Config{Dir: dir})
		if err != nil {
			log.Fatalf("Failed to create file store: %v", err)
		}
		fmt.Printf("Store: file (%s)\n", dir)
		return s
	case "memory":
		fmt.Println("Store: in-memory (volatile)")
		return storeinmem.NewConversationStore()
	default:
		if backend != "" {
			log.Fatalf("Unknown STORE_BACKEND: %s (use: memory, file)", backend)
		}
		fmt.Println("Store: none (set STORE_BACKEND=memory|file to persist)")
		return nil
	}
}
