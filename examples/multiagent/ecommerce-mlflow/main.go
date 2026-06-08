package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/examples/internal/exampleutil"
	"github.com/Kaelancode/kaeAgent-Public/schema"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
)

func main() {
	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, relying on system env")
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	provider, err := exampleutil.SelectProvider()
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}
	provider = exampleutil.WrapProvider(provider)

	tracer, shutdown, err := exampleutil.SetupMLflowTracer(ctx)
	if err != nil {
		log.Fatalf("Failed to setup MLflow tracing: %v", err)
	}
	defer shutdown()

	model := exampleutil.ModelForProvider(provider.Name())
	budget := streaming.NewBudget(streaming.BudgetConfig{
		MaxTokens:          500000,
		MaxCostUSD:         5.00,
		CostPerInputToken:  0.000003,
		CostPerOutputToken: 0.000015,
	})
	logger := zerolog.New(os.Stderr).Level(exampleutil.ParseLogLevel(os.Getenv("LOG_LEVEL"))).With().Timestamp().Logger()

	shopAssistant := agent.NewAgent(agent.AgentConfig{
		Name:  "shop_assistant",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a friendly e-commerce shopping assistant.",
			"You own the final reply to the user.",
			"Use consult_product_expert for product details, comparisons, fit/sizing, and availability questions.",
			"Use consult_order_specialist for order tracking, delivery estimates, returns, exchanges, and refund status.",
			"If a question involves both product choice and order logistics, consult both specialists before replying.",
			"Give a concise, helpful answer. Do not mention internal tool names or delegation details.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.4),
		MaxHistory:  80,
		Subagents:   []string{"product_expert", "order_specialist"},
		MaxSteps:    10,
	})

	productExpert := agent.NewAgent(agent.AgentConfig{
		Name:  "product_expert",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a product knowledge expert advising the shopping assistant.",
			"Use your tools to look up product details, compare options, and check sizing and compatibility.",
			"Return concise product facts and recommendations. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	productExpert.RegisterTool(productCatalogTool())
	productExpert.RegisterTool(sizingGuideTool())

	orderSpecialist := agent.NewAgent(agent.AgentConfig{
		Name:  "order_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are an order and logistics specialist advising the shopping assistant.",
			"Use your tools to track orders, estimate delivery, and look up return and exchange policies.",
			"Return concise logistics facts and next steps. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	orderSpecialist.RegisterTool(orderTrackingTool())
	orderSpecialist.RegisterTool(returnPolicyTool())

	registry := agent.NewRegistry()
	registry.Register(shopAssistant)
	registry.Register(productExpert)
	registry.Register(orderSpecialist)

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:         provider,
		Agent:            shopAssistant,
		SubagentResolver: registry,
		Middleware: []agent.Middleware{
			agent.CostGuardMiddleware(budget),
			agent.RetryMiddleware(3, 500*time.Millisecond),
		},
		MaxSteps:           12,
		MaxToolConcurrency: 2,
		Tracer:             tracer,
		Logger:             logger,
		UserID:             "ecommerce-example-user",
		AgentID:            shopAssistant.Name(),
	})

	fmt.Println("=== Multi-Agent E-Commerce with MLflow Tracing ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Tracing: MLflow OTLP HTTP")
	fmt.Println("Mode: shop_assistant consults product_expert and order_specialist, then replies directly.")
	fmt.Println("Available consults: product_expert, order_specialist")
	fmt.Println("Try: I want to buy running shoes for a half-marathon but I'm not sure about sizing. Also, if I order today, when would they arrive in Brooklyn?")
	fmt.Println("Try: Is the TrailRunner Pro waterproof? And what's the return policy if it doesn't fit?")
	fmt.Println("Commands: /usage, /history, /clear, /quit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if handleCommand(input, rt, budget) {
			if exampleutil.IsQuitCommand(input) {
				break
			}
			continue
		}

		response, err := rt.Run(ctx, input)
		if err != nil {
			fmt.Printf("[error] %v\n\n", err)
			continue
		}
		fmt.Printf("\nShop> %s\n\n", response)
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Input error: %v", err)
	}
	fmt.Println("\nGoodbye.")
}

func handleCommand(input string, rt *agent.Runtime, budget *streaming.Budget) bool {
	if exampleutil.IsQuitCommand(input) {
		in, out, total, cost := budget.Usage()
		fmt.Printf("\nSession stats: %d input, %d output, %d total tokens (est. $%.4f)\n", in, out, total, cost)
		fmt.Println("Goodbye.")
		return true
	}

	switch input {
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
	default:
		return false
	}
}

func productCatalogTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_product",
		Description: "Looks up product details, pricing, and availability by name or SKU.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"product"},
			Properties: map[string]*schema.Schema{
				"product":  {Type: "string", Description: "Product name or SKU."},
				"category": {Type: "string", Description: "Category filter, for example shoes, jackets, accessories."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"product":         input["product"],
				"in_stock":        true,
				"price_usd":       129.99,
				"available_sizes": []string{"7", "8", "9", "10", "11", "12"},
				"material":        "breathable mesh upper, cushioned foam sole",
				"waterproof":      false,
			}, nil
		},
	}
}

func sizingGuideTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_sizing_guide",
		Description: "Checks the sizing guide for a product category and foot length.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"category"},
			Properties: map[string]*schema.Schema{
				"category": {Type: "string", Description: "Product category, for example shoes, jackets, hats."},
				"foot_cm":  {Type: "number", Description: "Foot length in centimeters, if known."},
				"size_us":  {Type: "string", Description: "US shoe size, if known."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"category":       input["category"],
				"recommendation": "If between sizes, go up half a size for running shoes.",
				"width_note":     "Standard width fits most; wide version available in black only.",
				"exchange_note":  "Free exchange within 60 days if the fit is not right.",
			}, nil
		},
	}
}

func orderTrackingTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "track_order",
		Description: "Tracks order status, shipment progress, and estimated delivery.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"order_id"},
			Properties: map[string]*schema.Schema{
				"order_id": {Type: "string", Description: "Order ID or confirmation number."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"order_id":           input["order_id"],
				"status":             "shipped",
				"carrier":            "FedEx",
				"tracking_number":    "FX-77442211",
				"estimated_delivery": "2 business days",
				"destination":        "Brooklyn, NY",
			}, nil
		},
	}
}

func returnPolicyTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_return_policy",
		Description: "Checks return, exchange, and refund policies for a product category.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"category"},
			Properties: map[string]*schema.Schema{
				"category": {Type: "string", Description: "Product category."},
				"reason":   {Type: "string", Description: "Reason for return, for example fit, defect, changed mind."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"category":        input["category"],
				"return_window":   "60 days from delivery",
				"exchange_option": "Free exchange for a different size within 60 days.",
				"refund_method":   "Original payment method, 5-10 business days after we receive the item.",
				"condition":       "Item must be unworn with original tags attached.",
			}, nil
		},
	}
}
