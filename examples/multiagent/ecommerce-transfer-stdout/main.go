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
	"github.com/Kaelancode/kaeAgent-Public/observability"
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

	model := exampleutil.ModelForProvider(provider.Name())
	tracer := observability.NewStdoutTracer(os.Stderr)
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
			"You are a shopping assistant who triages customer requests to the right specialist.",
			"You can transfer the conversation to a specialist who then owns the reply.",
			"Use transfer_to_product_specialist for product details, comparisons, sizing, availability, and fit questions.",
			"Use transfer_to_order_specialist for order tracking, delivery estimates, returns, exchanges, and refund status.",
			"Use transfer_to_returns_specialist for returns, exchanges, warranty claims, and product defects.",
			"Pass a concise input summary, a short reason, and useful metadata such as topic=product, topic=order, or topic=returns.",
			"Do not answer specialist-owned cases yourself after choosing transfer — the transferred specialist owns the reply.",
			"For simple questions that do not need specialist ownership, answer directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		Subagents:   []string{"product_specialist", "order_specialist", "returns_specialist"},
		MaxSteps:    10,
	})

	productSpecialist := agent.NewAgent(agent.AgentConfig{
		Name:  "product_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a product knowledge specialist and now own the user-facing reply.",
			"Help with product details, comparisons, sizing, availability, and fit recommendations.",
			"If the user shifts to an order or delivery issue, call transfer_to_order_specialist.",
			"If the user shifts to a return or warranty issue, call transfer_to_returns_specialist.",
			"If the issue needs general coordination, call transfer_to_shop_assistant.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	productSpecialist.RegisterTool(productCatalogTool())
	productSpecialist.RegisterTool(sizingGuideTool())

	orderSpecialist := agent.NewAgent(agent.AgentConfig{
		Name:  "order_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are an order and logistics specialist and now own the user-facing reply.",
			"Help with order tracking, delivery estimates, address changes, and missing packages.",
			"If the user shifts to a product question, call transfer_to_product_specialist.",
			"If the user shifts to a return or refund issue, call transfer_to_returns_specialist.",
			"If the issue needs general coordination, call transfer_to_shop_assistant.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	orderSpecialist.RegisterTool(orderTrackingTool())
	orderSpecialist.RegisterTool(deliveryActionsTool())

	returnsSpecialist := agent.NewAgent(agent.AgentConfig{
		Name:  "returns_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a returns and warranty specialist and now own the user-facing reply.",
			"Help with returns, exchanges, refunds, warranty claims, and product defects.",
			"If the user shifts to an order or delivery issue, call transfer_to_order_specialist.",
			"If the user shifts to a product question, call transfer_to_product_specialist.",
			"If the issue needs general coordination, call transfer_to_shop_assistant.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	returnsSpecialist.RegisterTool(returnPolicyTool())
	returnsSpecialist.RegisterTool(warrantyLookupTool())

	registry := agent.NewRegistry()
	registry.Register(shopAssistant)
	registry.Register(productSpecialist)
	registry.Register(orderSpecialist)
	registry.Register(returnsSpecialist)

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
		TransferInputFilter: agent.ComposeTransferInput(
			agent.RemoveToolTransferInput(),
			agent.RecentWindowTransferInput(8),
			agent.NestTransferHistory(),
		),
		Tracer:  tracer,
		Logger:  logger,
		UserID:  "stdout-ecommerce-user",
		AgentID: shopAssistant.Name(),
	})

	fmt.Println("=== Multi-Agent E-Commerce Transfer with Stdout Tracing ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Tracing: stdout. Inspect stderr for gen_ai.agent.transfer events.")
	fmt.Println("Mode: shop_assistant transfers to specialist; specialist owns replies and can re-transfer.")
	fmt.Println("Available transfers: product_specialist, order_specialist, returns_specialist")
	fmt.Println("Try: I want running shoes for a half-marathon but not sure about sizing.")
	fmt.Println("Try: My order hasn't arrived and I need to return a defective jacket.")
	fmt.Println("Look for: gen_ai.agent.transfer gen_ai.handoff.from_agent=shop_assistant gen_ai.handoff.to_agent=product_specialist")
	fmt.Println("Commands: /active, /usage, /history, /clear, /quit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		active := rt.ActiveAgent(registry, shopAssistant.Name())
		fmt.Printf("%s You> ", active)
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

		before := rt.ActiveAgent(registry, shopAssistant.Name())
		response, err := rt.Run(ctx, input)
		if err != nil {
			fmt.Printf("[error] %v\n\n", err)
			continue
		}
		after := rt.ActiveAgent(registry, shopAssistant.Name())
		if before != after {
			fmt.Printf("\n[transfer: %s -> %s]\n", before, after)
		}
		fmt.Printf("%s> %s\n\n", after, response)
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
	case "/active":
		fmt.Printf("  Active agent: %s\n\n", rt.ActiveAgent(nil, "shop_assistant"))
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
				"status":             "in_transit",
				"carrier":            "FedEx",
				"tracking_number":    "FX-88332211",
				"estimated_delivery": "2 business days",
				"destination":        "Brooklyn, NY",
			}, nil
		},
	}
}

func deliveryActionsTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_delivery_actions",
		Description: "Looks up available delivery exception actions such as address change, hold, or redirect.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"issue": {Type: "string", Description: "Delivery issue, for example missing, wrong address, damaged."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"available_actions": []string{
					"Request carrier trace.",
					"Update delivery address before final delivery attempt.",
					"Schedule redelivery.",
				},
				"customer_needed": []string{"order ID", "current shipping address"},
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
				"exchange_option": "Free exchange for a different size or color within 60 days.",
				"refund_method":   "Original payment method, 5-10 business days after we receive the item.",
				"condition":       "Item must be unworn with original tags attached.",
			}, nil
		},
	}
}

func warrantyLookupTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_warranty",
		Description: "Looks up warranty coverage and claim options for a product.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"product"},
			Properties: map[string]*schema.Schema{
				"product": {Type: "string", Description: "Product name or SKU."},
				"issue":   {Type: "string", Description: "Issue or defect description."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"product":       input["product"],
				"warranty":      "1-year manufacturer warranty covers defects in materials and workmanship.",
				"claim_options": []string{"Replacement of same item", "Store credit for full purchase price"},
				"required_info": []string{"order ID", "photos of defect", "serial number if applicable"},
			}, nil
		},
	}
}
