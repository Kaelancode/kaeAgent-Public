package exampleutil

import (
	"bufio"
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/observability"
	oteltracer "github.com/yourorg/agent-sdk/observability/otel"
	"github.com/yourorg/agent-sdk/streaming"
)

type SwitchableTracer struct {
	mu     sync.Mutex
	active bool
	noop   observability.NoopTracer
	stdout *observability.StdoutTracer
}

func NewSwitchableTracer() *SwitchableTracer {
	return &SwitchableTracer{
		noop:   observability.NoopTracer{},
		stdout: observability.NewStdoutTracer(os.Stderr),
		active: true,
	}
}

func (s *SwitchableTracer) Toggle() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.active = !s.active
	return s.active
}

func (s *SwitchableTracer) current() observability.Tracer {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.active {
		return s.stdout
	}
	return s.noop
}

func (s *SwitchableTracer) StartSpan(ctx context.Context, name string, attrs map[string]string) (context.Context, observability.Span) {
	return s.current().StartSpan(ctx, name, attrs)
}

func (s *SwitchableTracer) EndSpan(ctx context.Context, span observability.Span, err error) {
	s.current().EndSpan(ctx, span, err)
}

func (s *SwitchableTracer) AddEvent(ctx context.Context, span observability.Span, name string, attrs map[string]string) {
	s.current().AddEvent(ctx, span, name, attrs)
}

func (s *SwitchableTracer) SetSpanAttributes(ctx context.Context, span observability.Span, attrs map[string]any) {
	s.current().SetSpanAttributes(ctx, span, attrs)
}

type ProviderEntry struct {
	Name    string
	EnvVar  string
	Factory func() (llm.Provider, error)
}

var providerTable = []ProviderEntry{
	{"openai", "OPENAI_API_KEY", func() (llm.Provider, error) { return llm.NewOpenAIProvider() }},
	{"claude", "ANTHROPIC_API_KEY", func() (llm.Provider, error) { return llm.NewClaudeProvider() }},
	{"gemini", "GEMINI_API_KEY", func() (llm.Provider, error) { return llm.NewGeminiProvider() }},
	{"qwen", "DASHSCOPE_API_KEY", func() (llm.Provider, error) { return llm.NewQwenProvider() }},
}

func SelectProvider() (llm.Provider, error) {
	forced := ""
	for i, arg := range os.Args[1:] {
		if (arg == "--provider" || arg == "-p") && i+2 <= len(os.Args[1:]) {
			forced = os.Args[i+2]
			break
		}
	}

	var available []ProviderEntry
	for _, p := range providerTable {
		if os.Getenv(p.EnvVar) != "" {
			available = append(available, p)
		}
	}
	if len(available) == 0 {
		return nil, fmt.Errorf("no API key found; set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DASHSCOPE_API_KEY")
	}
	if forced != "" {
		for _, p := range available {
			if p.Name == forced {
				return p.Factory()
			}
		}
		names := make([]string, len(available))
		for i, p := range available {
			names[i] = p.Name
		}
		return nil, fmt.Errorf("provider %q not available; have keys for: %s", forced, strings.Join(names, ", "))
	}
	if len(available) == 1 {
		return available[0].Factory()
	}

	fmt.Println("Multiple providers available:")
	for i, p := range available {
		fmt.Printf("  [%d] %s\n", i+1, p.Name)
	}
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Printf("Select provider [1-%d]: ", len(available))
		if !scanner.Scan() {
			return nil, fmt.Errorf("no input received")
		}
		input := strings.TrimSpace(scanner.Text())
		for i, p := range available {
			if input == fmt.Sprintf("%d", i+1) || input == p.Name {
				return p.Factory()
			}
		}
		fmt.Printf("  Invalid choice %q. Enter a number or name.\n", input)
	}
}

func WrapProvider(provider llm.Provider) llm.Provider {
	return llm.WrapProvider(
		provider,
		llm.WithRateLimit(&intervalLimiter{interval: 100 * time.Millisecond}),
		llm.WithConcurrencyLimit(4),
		llm.WithRetry(retryPolicy{maxAttempts: 2, backoff: 250 * time.Millisecond}),
	)
}

func ModelForProvider(name string) string {
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

func SetupMLflowTracer(ctx context.Context) (observability.Tracer, func(), error) {
	endpoint := envOrDefault("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "localhost:5000")
	experimentID := envOrDefault("MLFLOW_EXPERIMENT_ID", "0")
	serviceName := envOrDefault("OTEL_SERVICE_NAME", "agent-sdk-multiagent-mlflow")
	insecure := envOrDefault("OTEL_EXPORTER_OTLP_INSECURE", "true") == "true"
	username := os.Getenv("MLFLOW_TRACKING_USERNAME")
	password := os.Getenv("MLFLOW_TRACKING_PASSWORD")

	fmt.Printf("Tracing to MLflow (OTLP HTTP): %s  experiment=%s\n", endpoint, experimentID)

	headers := map[string]string{
		"x-mlflow-experiment-id": experimentID,
	}

	tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
		Endpoint:     endpoint,
		ServiceName:  serviceName,
		Insecure:     insecure,
		ExporterType: "http",
		URLPath:      "/v1/traces",
		Headers:      headers,
		Username:     username,
		Password:     password,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating MLflow tracer provider: %w", err)
	}

	return oteltracer.NewTracer(tp, serviceName), shutdown, nil
}

func SetupOpikTracer(ctx context.Context) (observability.Tracer, func(), error) {
	endpoint := envOrDefault("OPIK_ENDPOINT", "www.comet.com")
	urlPath := envOrDefault("OPIK_URL_PATH", "/opik/api/v1/private/otel/v1/traces")
	serviceName := envOrDefault("OTEL_SERVICE_NAME", "agent-sdk-multiagent-opik")
	project := envOrDefault("OPIK_PROJECT", "default")
	workspace := envOrDefault("OPIK_WORKSPACE", "default")
	apiKey := os.Getenv("OPIK_API_KEY")
	insecure := envOrDefault("OPIK_INSECURE", "false") == "true"

	headers := map[string]string{
		"projectName":     project,
		"Comet-Workspace": workspace,
	}
	if apiKey != "" {
		headers["Authorization"] = apiKey
	}

	tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
		Endpoint:     endpoint,
		ServiceName:  serviceName,
		Insecure:     insecure,
		ExporterType: "http",
		URLPath:      urlPath,
		Headers:      headers,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating Opik tracer provider: %w", err)
	}

	return oteltracer.NewTracer(tp, serviceName), shutdown, nil
}

func SetupLangfuseTracer(ctx context.Context) (observability.Tracer, func(), error) {
	publicKey := os.Getenv("LANGFUSE_PUBLIC_KEY")
	secretKey := os.Getenv("LANGFUSE_SECRET_KEY")
	if publicKey == "" || secretKey == "" {
		return nil, nil, fmt.Errorf("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
	}

	endpoint := os.Getenv("LANGFUSE_OTEL_TRACES_ENDPOINT")
	if endpoint == "" {
		baseURL := envOrDefault("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
		endpoint = strings.TrimRight(baseURL, "/") + "/api/public/otel/v1/traces"
	}

	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, nil, fmt.Errorf("invalid Langfuse OTLP traces endpoint %q", endpoint)
	}

	serviceName := envOrDefault("OTEL_SERVICE_NAME", "agent-sdk-multiagent-langfuse")
	tp, shutdown, err := oteltracer.NewTracerProvider(ctx, oteltracer.ProviderConfig{
		Endpoint:     parsed.Host,
		ServiceName:  serviceName,
		Insecure:     parsed.Scheme == "http",
		ExporterType: "http",
		URLPath:      parsed.Path,
		Headers: map[string]string{
			"x-langfuse-ingestion-version": "4",
		},
		Username: publicKey,
		Password: secretKey,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("creating Langfuse tracer provider: %w", err)
	}

	return oteltracer.NewTracer(tp, serviceName), shutdown, nil
}

func ParseLogLevel(level string) zerolog.Level {
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

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func HandleCommonCommand(input string, rt *agent.Runtime, budget *streaming.Budget, tracer *SwitchableTracer, registry agent.SubagentResolver, rootAgent string) bool {
	switch input {
	case "/quit", "/exit", "/q":
		in, out, total, cost := budget.Usage()
		fmt.Printf("\nSession stats: %d input, %d output, %d total tokens (est. $%.4f)\n", in, out, total, cost)
		fmt.Println("Goodbye.")
		os.Exit(0)
		return true
	case "/usage":
		in, out, total, cost := budget.Usage()
		remTokens, remCost := budget.Remaining()
		fmt.Printf("  Tokens used:  %d in / %d out / %d total\n", in, out, total)
		fmt.Printf("  Cost:         $%.4f\n", cost)
		fmt.Printf("  Remaining:    %d tokens / $%.4f\n\n", remTokens, remCost)
		return true
	case "/history":
		msgs := rt.ConversationMessages()
		fmt.Printf("  Conversation has %d messages:\n", len(msgs))
		for i, m := range msgs {
			content := strings.ReplaceAll(m.Content, "\n", " ")
			if len(content) > 90 {
				content = content[:87] + "..."
			}
			fmt.Printf("    [%d] %-9s %s\n", i, m.Role, content)
		}
		fmt.Println()
		return true
	case "/clear":
		rt.ClearConversation()
		sessionSnap := rt.SessionSnapshot()
		if sessionSnap.Config.SystemPrompt != "" {
			rt.SetConversationSystem(sessionSnap.Config.SystemPrompt)
		}
		fmt.Println("  Conversation cleared.")
		fmt.Println()
		return true
	case "/trace":
		if tracer.Toggle() {
			fmt.Println("  Tracing ON. Spans and events print to stderr.")
		} else {
			fmt.Println("  Tracing OFF.")
		}
		fmt.Println()
		return true
	case "/active":
		fmt.Printf("  Active agent: %s\n\n", rt.ActiveAgent(registry, rootAgent))
		return true
	}
	return false
}

func Float32Ptr(v float32) *float32 {
	return &v
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
