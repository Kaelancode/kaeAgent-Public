package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	oteltracer "github.com/yourorg/agent-sdk/observability/otel"
	"github.com/yourorg/agent-sdk/store"
	storefile "github.com/yourorg/agent-sdk/store/file"
	storeinmem "github.com/yourorg/agent-sdk/store/inmem"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
)

func main() {
	godotenv.Load()
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	logLevel := parseLogLevel(os.Getenv("LOG_LEVEL"))
	logger := zerolog.New(os.Stderr).Level(logLevel).With().Timestamp().Logger()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	provider, err := resolveProvider()
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	fmt.Printf("Using provider: %s\n", provider.Name())

	tracer, shutdown, err := setupTracer(ctx)
	if err != nil {
		log.Fatalf("Failed to setup tracer: %v", err)
	}
	defer shutdown()

	convStore := setupStore()

	registry := tools.NewRegistry()
	registry.Register(tools.NewHTTPTool())
	registry.Register(currentTimeTool())

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

func float32Ptr(v float32) *float32 {
	return &v
}

func setupTracer(ctx context.Context) (observability.Tracer, func(), error) {
	backend := os.Getenv("TRACE_BACKEND")
	switch strings.ToLower(backend) {
	case "jaeger":
		return setupJaeger(ctx)
	case "mlflow":
		return setupMLflow(ctx)
	case "stdout":
		return setupStdout(), func() {}, nil
	default:
		fmt.Println("TRACE_BACKEND not set — using NoopTracer. Set TRACE_BACKEND=jaeger|mlflow|stdout")
		return observability.NoopTracer{}, func() {}, nil
	}
}

func setupJaeger(ctx context.Context) (observability.Tracer, func(), error) {
	endpoint := envOrDefault("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
	fmt.Printf("Tracing to Jaeger (OTLP gRPC): %s\n", endpoint)

	tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
		Endpoint:     endpoint,
		ServiceName:  envOrDefault("OTEL_SERVICE_NAME", "agent-sdk"),
		Insecure:     true,
		ExporterType: "grpc",
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating Jaeger tracer provider: %w", err)
	}

	tracer := oteltracer.NewTracer(tp, "agent-sdk")
	return tracer, shutdown, nil
}

func setupMLflow(ctx context.Context) (observability.Tracer, func(), error) {
	endpoint := envOrDefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "localhost:5000")
	experimentID := envOrDefault("MLFLOW_EXPERIMENT_ID", "0")
	serviceName := envOrDefault("OTEL_SERVICE_NAME", "agent-sdk")
	username := os.Getenv("MLFLOW_TRACKING_USERNAME")
	password := os.Getenv("MLFLOW_TRACKING_PASSWORD")
	insecure := envOrDefault("OTEL_EXPORTER_OTLP_INSECURE", "false") == "true"
	fmt.Printf("Tracing to MLflow (OTLP HTTP): %s  experiment=%s  insecure=%v\n", endpoint, experimentID, insecure)

	tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
		Endpoint:     endpoint,
		ServiceName:  serviceName,
		Insecure:     insecure,
		ExporterType: "http",
		URLPath:      "/v1/traces",
		Headers:      map[string]string{"x-mlflow-experiment-id": experimentID},
		Username:     username,
		Password:     password,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating MLflow tracer provider: %w", err)
	}

	tracer := oteltracer.NewTracer(tp, serviceName)
	return tracer, shutdown, nil
}

func setupStdout() observability.Tracer {
	fmt.Println("Tracing to stdout")
	return observability.NewStdoutTracer(os.Stderr)
}

func resolveProvider() (llm.Provider, error) {
	if os.Getenv("GEMINI_API_KEY") != "" {
		return llm.NewGeminiProvider()
	}
	if os.Getenv("OPENAI_API_KEY") != "" {
		return llm.NewOpenAIProvider()
	}
	if os.Getenv("ANTHROPIC_API_KEY") != "" {
		return llm.NewClaudeProvider()
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
		return "gemini-2.5-flash"
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
			return map[string]any{
				"time":     time.Now().UTC().Format(time.RFC3339),
				"timezone": "UTC",
			}, nil
		},
	}
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseLogLevel(level string) zerolog.Level {
	switch strings.ToLower(level) {
	case "debug":
		return zerolog.DebugLevel
	case "info":
		return zerolog.InfoLevel
	case "warn", "warning":
		return zerolog.WarnLevel
	case "error":
		return zerolog.ErrorLevel
	default:
		return zerolog.Disabled
	}
}

func setupStore() store.ConversationStore {
	backend := os.Getenv("STORE_BACKEND")
	switch strings.ToLower(backend) {
	case "file":
		dir := envOrDefault("STORE_DIR", "./data")
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
