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

	tracer, shutdown, err := exampleutil.SetupLangfuseTracer(ctx)
	if err != nil {
		log.Fatalf("Failed to setup Langfuse tracing: %v", err)
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

	coordinator := agent.NewAgent(agent.AgentConfig{
		Name:  "support_coordinator",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are the coordinator for a SaaS account support team.",
			"You own the final reply to the user.",
			"Use consult_billing_analyst for invoices, duplicate charges, credits, payment failures, plan changes, or refund eligibility.",
			"Use consult_security_analyst for suspicious logins, MFA resets, audit events, account access, or compliance-sensitive questions.",
			"If the request involves both account access and billing risk, consult both specialists before replying.",
			"Give a concise customer-facing answer with the next safest action. Do not mention internal tool names.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.3),
		MaxHistory:  80,
		Subagents:   []string{"billing_analyst", "security_analyst"},
		MaxSteps:    10,
	})

	billing := agent.NewAgent(agent.AgentConfig{
		Name:  "billing_analyst",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a billing analyst advising the support coordinator.",
			"Use tools to check invoice state and credit/refund policy.",
			"Return operational facts and safe next steps. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	billing.RegisterTool(invoiceStatusTool())
	billing.RegisterTool(accountCreditPolicyTool())

	security := agent.NewAgent(agent.AgentConfig{
		Name:  "security_analyst",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a security analyst advising the support coordinator.",
			"Use tools to inspect login risk and MFA recovery requirements.",
			"Return concise risk facts and verification steps. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	security.RegisterTool(loginRiskTool())
	security.RegisterTool(mfaRecoveryPolicyTool())

	registry := agent.NewRegistry()
	registry.Register(coordinator)
	registry.Register(billing)
	registry.Register(security)

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:         provider,
		Agent:            coordinator,
		SubagentResolver: registry,
		Middleware: []agent.Middleware{
			agent.CostGuardMiddleware(budget),
			agent.RetryMiddleware(3, 500*time.Millisecond),
		},
		MaxSteps:           12,
		MaxToolConcurrency: 2,
		Tracer:             tracer,
		Logger:             logger,
		UserID:             "langfuse-example-user",
		AgentID:            coordinator.Name(),
	})

	fmt.Println("=== Multi-Agent Langfuse Consult Chat ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Tracing: Langfuse OTLP HTTP")
	fmt.Println("Mode: coordinator chooses consult_<specialist> tools, then coordinator replies to the user.")
	fmt.Println("Available consults: billing_analyst, security_analyst")
	fmt.Println("Try: I see two charges this month and also got a suspicious login email.")
	fmt.Println("Try: I lost access to MFA and need to update the card before renewal.")
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
		fmt.Printf("\nSupport> %s\n\n", response)
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
		return true
	}

	switch input {
	case "/usage":
		in, out, total, cost := budget.Usage()
		fmt.Printf("  Tokens used: %d in / %d out / %d total\n", in, out, total)
		fmt.Printf("  Cost:        $%.4f\n\n", cost)
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

func invoiceStatusTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_invoice_status",
		Description: "Looks up recent invoice status and duplicate charge indicators.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"account_id": {Type: "string", Description: "Account ID, if known."},
				"invoice_id": {Type: "string", Description: "Invoice ID, if known."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"latest_invoice_status": "paid",
				"duplicate_charge_risk": "possible",
				"recommended_action":    "Verify billing email and last four digits, then open a duplicate-charge review.",
				"account_id":            input["account_id"],
				"invoice_id":            input["invoice_id"],
			}, nil
		},
	}
}

func accountCreditPolicyTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_credit_policy",
		Description: "Checks whether a billing credit or refund review is available.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"reason": {Type: "string", Description: "Reason for the requested credit or refund."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"eligible_for_review": true,
				"review_window_days":  60,
				"required_evidence":   []string{"invoice ID", "billing email", "charge date"},
				"reason":              input["reason"],
			}, nil
		},
	}
}

func loginRiskTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "inspect_login_risk",
		Description: "Inspects recent sign-in risk for an account.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"account_id": {Type: "string", Description: "Account ID, if known."},
				"email":      {Type: "string", Description: "Account email, if known."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"risk_level":         "medium",
				"recent_event":       "new device sign-in from an unusual region",
				"recommended_action": "Verify identity, revoke unknown sessions, then reset MFA if verified.",
				"account_id":         input["account_id"],
				"email":              input["email"],
			}, nil
		},
	}
}

func mfaRecoveryPolicyTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_mfa_recovery_policy",
		Description: "Looks up safe MFA recovery requirements.",
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"role": {Type: "string", Description: "User role, for example admin or member."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"admin_recovery_requires":  []string{"verified billing contact", "backup admin approval or support review", "session revocation"},
				"member_recovery_requires": []string{"admin approval", "identity verification"},
				"role":                     input["role"],
			}, nil
		},
	}
}
