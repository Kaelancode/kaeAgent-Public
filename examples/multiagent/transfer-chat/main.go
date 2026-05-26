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

	"github.com/joho/godotenv"
	"github.com/rs/zerolog"
	"github.com/yourorg/agent-sdk/agent"
	"github.com/yourorg/agent-sdk/examples/multiagent/internal/exampleutil"
	"github.com/yourorg/agent-sdk/schema"
	"github.com/yourorg/agent-sdk/streaming"
	"github.com/yourorg/agent-sdk/tools"
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

	triage := agent.NewAgent(agent.AgentConfig{
		Name:  "triage",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a front-line support triage agent.",
			"You can transfer ownership to one specialist, but you must choose the right specialist yourself based on the user's issue.",
			"Use transfer_to_billing_specialist for duplicate charges, invoices, refunds, subscriptions, payment failures, or account credits.",
			"Use transfer_to_technical_support for broken devices, setup failures, connectivity, diagnostics, or compatibility problems.",
			"Use transfer_to_order_success for order status, delivery exceptions, address changes, missing packages, or replacement logistics.",
			"Pass a concise input, a short reason, and useful metadata such as topic=billing, topic=technical, or topic=order.",
			"Do not answer specialist-owned cases yourself after choosing transfer. The transferred specialist owns the user-facing reply.",
			"For simple general questions that do not need specialist ownership, answer directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		Subagents:   []string{"billing_specialist", "technical_support", "order_success"},
		MaxSteps:    10,
	})

	billing := agent.NewAgent(agent.AgentConfig{
		Name:  "billing_specialist",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a billing specialist and now own the user-facing reply.",
			"Help with duplicate charges, invoices, refunds, subscriptions, and payment troubleshooting.",
			"Ask only for safe account details such as order ID, invoice ID, billing email, and last four digits if needed.",
			"Never ask for a full card number, CVV, password, or one-time code.",
			"If the user changes to a technical issue, call transfer_to_technical_support.",
			"If the user changes to an order or delivery issue, call transfer_to_order_success.",
			"If the issue needs coordinator judgment, call transfer_to_triage.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	billing.RegisterTool(invoiceLookupTool())
	billing.RegisterTool(refundEligibilityTool())

	technical := agent.NewAgent(agent.AgentConfig{
		Name:  "technical_support",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a technical support specialist and now own the user-facing reply.",
			"Help with device setup, diagnostics, compatibility, connectivity, and repair routing.",
			"Use tools before giving troubleshooting or replacement guidance when details are available.",
			"If the user changes to a billing issue, call transfer_to_billing_specialist.",
			"If the user changes to an order or delivery issue, call transfer_to_order_success.",
			"If the issue needs coordinator judgment, call transfer_to_triage.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	technical.RegisterTool(diagnosticTool())
	technical.RegisterTool(repairOptionTool())

	orderSuccess := agent.NewAgent(agent.AgentConfig{
		Name:  "order_success",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are an order success specialist and now own the user-facing reply.",
			"Help with order status, delivery exceptions, missing packages, replacement shipments, and address changes.",
			"Use tools to check shipment state and available logistics actions.",
			"If the user changes to a billing issue, call transfer_to_billing_specialist.",
			"If the user changes to a technical issue, call transfer_to_technical_support.",
			"If the issue needs coordinator judgment, call transfer_to_triage.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		MaxSteps:    8,
	})
	orderSuccess.RegisterTool(orderStatusTool())
	orderSuccess.RegisterTool(deliveryActionTool())

	registry := agent.NewRegistry()
	registry.Register(triage)
	registry.Register(billing)
	registry.Register(technical)
	registry.Register(orderSuccess)

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:         provider,
		Agent:            triage,
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
		UserID:  "example-user",
		AgentID: triage.Name(),
	})

	fmt.Println("=== Multi-Agent Transfer Chat ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Mode: triage chooses one transfer_to_<specialist>; after transfer, that specialist owns replies and stays active.")
	fmt.Println("Available transfers: billing_specialist, technical_support, order_success")
	fmt.Println("Transferred specialists can transfer to another specialist or back to triage when the topic changes.")
	fmt.Println("Try: I was charged twice for my subscription this month and need help.")
	fmt.Println("Try: My camera will not connect to Wi-Fi after setup.")
	fmt.Println("Try: My replacement order says delivered, but it never arrived.")
	fmt.Println("Commands: /active, /usage, /history, /clear, /trace, /quit")
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		active := rt.ActiveAgent(registry, triage.Name())
		fmt.Printf("%s You> ", active)
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if exampleutil.HandleCommonCommand(input, rt, budget, tracer, registry, triage.Name()) {
			continue
		}

		before := rt.ActiveAgent(registry, triage.Name())
		response, err := rt.Run(ctx, input)
		if err != nil {
			fmt.Printf("[error] %v\n\n", err)
			continue
		}
		after := rt.ActiveAgent(registry, triage.Name())
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

func invoiceLookupTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_invoice",
		Description: "Looks up invoice and charge status for an order, invoice, or subscription reference.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"reference"},
			Properties: map[string]*schema.Schema{
				"reference": {Type: "string", Description: "Order ID, invoice ID, or subscription reference."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"reference":        input["reference"],
				"status":           "paid",
				"duplicate_charge": true,
				"latest_charge":    "2026-05-18",
				"safe_next_step":   "Verify billing email and last four digits before discussing account-specific details.",
			}, nil
		},
	}
}

func refundEligibilityTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_refund_eligibility",
		Description: "Checks refund or credit eligibility for billing cases.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"reason": {Type: "string", Description: "Reason for refund or credit."},
				"amount": {Type: "number", Description: "Amount, if known."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"eligible":        true,
				"resolution":      "Refund duplicate charge to original payment method or apply account credit.",
				"processing_time": "3-5 business days for card refunds; credit applies immediately.",
			}, nil
		},
	}
}

func diagnosticTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "run_device_diagnostic",
		Description: "Runs a guided diagnostic lookup for a device symptom.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"device", "symptom"},
			Properties: map[string]*schema.Schema{
				"device":  {Type: "string", Description: "Device name or model."},
				"symptom": {Type: "string", Description: "Observed issue."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"device": input["device"],
				"checks": []string{
					"Confirm firmware is current.",
					"Power-cycle device and router.",
					"Test setup on a 2.4GHz network if the device does not support 5GHz onboarding.",
				},
				"escalate_if": "Setup fails after reset and network compatibility is confirmed.",
			}, nil
		},
	}
}

func repairOptionTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_repair_options",
		Description: "Looks up repair, replacement, or warranty options for a device.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"device"},
			Properties: map[string]*schema.Schema{
				"device": {Type: "string", Description: "Device name or model."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"device":             input["device"],
				"warranty_route":     "Advance replacement available if device is under one year old.",
				"repair_route":       "Mail-in repair takes 7-10 business days.",
				"required_materials": []string{"order ID", "serial number", "diagnostic summary"},
			}, nil
		},
	}
}

func orderStatusTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_order_status",
		Description: "Checks order, shipment, and delivery status.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"order_id"},
			Properties: map[string]*schema.Schema{
				"order_id": {Type: "string", Description: "Order ID or shipment reference."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"order_id":       input["order_id"],
				"carrier_status": "delivered",
				"delivery_date":  "2026-05-18",
				"risk_flag":      "possible missing package claim",
			}, nil
		},
	}
}

func deliveryActionTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_delivery_actions",
		Description: "Looks up available delivery exception actions.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"issue": {Type: "string", Description: "Delivery issue."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"available_actions": []string{
					"Open carrier trace.",
					"Start missing package affidavit.",
					"Ship replacement after claim criteria are met.",
				},
				"customer_needed": []string{"shipping address confirmation", "safe drop-off locations checked"},
			}, nil
		},
	}
}
