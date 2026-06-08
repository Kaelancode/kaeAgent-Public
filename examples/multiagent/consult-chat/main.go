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

	model := exampleutil.ModelForProvider(provider.Name())
	tracer := exampleutil.NewSwitchableTracer()
	budget := streaming.NewBudget(streaming.BudgetConfig{
		MaxTokens:          500000,
		MaxCostUSD:         5.00,
		CostPerInputToken:  0.000003,
		CostPerOutputToken: 0.000015,
	})
	logger := zerolog.New(os.Stderr).Level(exampleutil.ParseLogLevel(os.Getenv("LOG_LEVEL"))).With().Timestamp().Logger()

	support := agent.NewAgent(agent.AgentConfig{
		Name:  "support",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a customer support assistant for an online electronics store.",
			"You own the final reply to the user.",
			"You can consult specialists, but you must choose the right one yourself based on the user's request.",
			"Use consult_policy_specialist for refunds, returns, warranty, damaged items, or eligibility rules.",
			"Use consult_inventory_specialist for stock, shipping dates, replacement availability, or delivery exceptions.",
			"Use consult_troubleshooting_specialist for device setup, compatibility, diagnostics, or repair guidance.",
			"If a request spans multiple domains, consult more than one specialist before replying.",
			"Use consulted results to answer naturally. Do not mention tool names or internal delegation.",
			"Do not transfer ownership. Specialists only advise you.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.4),
		MaxHistory:  80,
		Subagents:   []string{"policy_specialist", "inventory_specialist", "troubleshooting_specialist"},
		MaxSteps:    10,
	})

	policy := agent.NewAgent(agent.AgentConfig{
		Name:  "policy_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a refund, return, and warranty policy specialist.",
			"Return concise policy facts and the next clarification question the support assistant should ask.",
			"Do not address the end user directly. You are advising another agent.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	policy.RegisterTool(returnPolicyTool())
	policy.RegisterTool(warrantyPolicyTool())

	inventory := agent.NewAgent(agent.AgentConfig{
		Name:  "inventory_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are an inventory and fulfillment specialist.",
			"Use your tools to check stock, replacement options, and estimated shipping.",
			"Return concise operational facts to the support assistant. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	inventory.RegisterTool(inventoryLookupTool())
	inventory.RegisterTool(shippingEstimateTool())

	troubleshooting := agent.NewAgent(agent.AgentConfig{
		Name:  "troubleshooting_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a device troubleshooting specialist.",
			"Use your tools for diagnostics, compatibility, and safe next steps.",
			"Return concise technical guidance to the support assistant. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	troubleshooting.RegisterTool(deviceDiagnosticTool())
	troubleshooting.RegisterTool(compatibilityLookupTool())

	registry := agent.NewRegistry()
	registry.Register(support)
	registry.Register(policy)
	registry.Register(inventory)
	registry.Register(troubleshooting)

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:         provider,
		Agent:            support,
		SubagentResolver: registry,
		Middleware: []agent.Middleware{
			agent.CostGuardMiddleware(budget),
			agent.RetryMiddleware(3, 500*time.Millisecond),
		},
		MaxSteps:           12,
		MaxToolConcurrency: 2,
		Tracer:             tracer,
		Logger:             logger,
		UserID:             "example-user",
		AgentID:            support.Name(),
	})

	fmt.Println("=== Multi-Agent Consult Chat ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Mode: support chooses one or more consult_<specialist> tools, then support still replies to the user.")
	fmt.Println("Available consults: policy_specialist, inventory_specialist, troubleshooting_specialist")
	fmt.Println("Try: I bought headphones 45 days ago, the left side stopped working, and I need a replacement before Friday.")
	fmt.Println("Try: Is the 65W USB-C charger in stock, and will it work with my TravelBook 14?")
	fmt.Println("Commands: /active, /usage, /history, /clear, /trace, /quit")
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
		if exampleutil.HandleCommonCommand(input, rt, budget, tracer, registry, support.Name()) {
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
		fmt.Printf("\nSupport> %s\n", response)
		fmt.Printf("[active agent: %s]\n\n", rt.ActiveAgent(registry, support.Name()))
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Input error: %v", err)
	}
	fmt.Println("\nGoodbye.")
}

func returnPolicyTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_return_policy",
		Description: "Looks up refund and return eligibility for a product category and days since purchase.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"category"},
			Properties: map[string]*schema.Schema{
				"category":            {Type: "string", Description: "Product category, for example headphones, laptop, charger."},
				"days_since_purchase": {Type: "integer", Description: "Number of days since purchase, if known."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"standard_window_days":  30,
				"defective_window_days": 60,
				"condition":             "Refunds after 30 days require a confirmed defect; replacements are preferred when inventory exists.",
				"category":              input["category"],
			}, nil
		},
	}
}

func warrantyPolicyTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_warranty_policy",
		Description: "Looks up warranty coverage and next steps for a product category.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"category"},
			Properties: map[string]*schema.Schema{
				"category": {Type: "string", Description: "Product category."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"coverage":        "One-year limited warranty for manufacturing defects.",
				"required_info":   []string{"order ID", "serial number", "short defect description"},
				"preferred_route": "Start warranty replacement if defect appears hardware-related.",
				"category":        input["category"],
			}, nil
		},
	}
}

func inventoryLookupTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_replacement_inventory",
		Description: "Checks stock and replacement availability for a SKU or product name.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"item"},
			Properties: map[string]*schema.Schema{
				"item": {Type: "string", Description: "SKU or product name."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"item":                  input["item"],
				"in_stock":              true,
				"replacement_available": true,
				"warehouse":             "west-region",
				"hold_window_hours":     24,
			}, nil
		},
	}
}

func shippingEstimateTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "estimate_shipping",
		Description: "Estimates shipping options for replacement or new order delivery.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination"},
			Properties: map[string]*schema.Schema{
				"destination": {Type: "string", Description: "Destination region or city."},
				"urgency":     {Type: "string", Description: "Desired delivery timing."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination":       input["destination"],
				"standard_delivery": "3-5 business days",
				"expedited":         "1-2 business days if ordered before 2pm local warehouse time",
				"cutoff":            "2pm local warehouse time",
			}, nil
		},
	}
}

func deviceDiagnosticTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "device_diagnostic_steps",
		Description: "Returns safe troubleshooting steps for common device symptoms.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"device", "symptom"},
			Properties: map[string]*schema.Schema{
				"device":  {Type: "string", Description: "Device or accessory name."},
				"symptom": {Type: "string", Description: "Observed symptom."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"device": input["device"],
				"steps": []string{
					"Test with a second cable or source device.",
					"Reset pairing or power-cycle the device.",
					"Check whether the issue follows the accessory or the source device.",
				},
				"likely_defect_if": "The same side or function fails across multiple source devices.",
			}, nil
		},
	}
}

func compatibilityLookupTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_compatibility",
		Description: "Checks compatibility between an accessory and a device model.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"accessory", "device"},
			Properties: map[string]*schema.Schema{
				"accessory": {Type: "string", Description: "Accessory name."},
				"device":    {Type: "string", Description: "Device model."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"accessory":  input["accessory"],
				"device":     input["device"],
				"compatible": true,
				"notes":      "Requires USB-C PD support; full-speed charging depends on the device accepting 20V profiles.",
			}, nil
		},
	}
}
