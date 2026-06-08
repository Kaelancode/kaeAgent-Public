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

	tracer, shutdown, err := exampleutil.SetupOpikTracer(ctx)
	if err != nil {
		log.Fatalf("Failed to setup Opik tracing: %v", err)
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

	travelAdvisor := agent.NewAgent(agent.AgentConfig{
		Name:  "travel_advisor",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a travel planning advisor helping users plan trips.",
			"You own the final reply to the user.",
			"Use consult_destination_expert for destination research, weather, visas, local customs, and attractions.",
			"Use consult_booking_agent for flights, hotels, pricing, and itinerary scheduling.",
			"If the request covers both destination choice and booking logistics, consult both specialists before replying.",
			"Give a concise, helpful answer with clear next steps. Do not mention internal tool names.",
		}, " "),
		MaxTokens:   1200,
		Temperature: exampleutil.Float32Ptr(0.4),
		MaxHistory:  80,
		Subagents:   []string{"destination_expert", "booking_agent"},
		MaxSteps:    10,
	})

	destinationExpert := agent.NewAgent(agent.AgentConfig{
		Name:  "destination_expert",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a destination research expert advising the travel advisor.",
			"Use your tools to look up weather, visa requirements, local customs, and top attractions.",
			"Return concise destination facts and travel tips. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	destinationExpert.RegisterTool(destinationInfoTool())
	destinationExpert.RegisterTool(weatherForecastTool())
	destinationExpert.RegisterTool(visaRequirementsTool())

	bookingAgent := agent.NewAgent(agent.AgentConfig{
		Name:  "booking_agent",
		Model: model,
		SystemPrompt: strings.Join([]string{
			"You are a booking and logistics specialist advising the travel advisor.",
			"Use your tools to search flights, hotels, and build itinerary estimates with pricing.",
			"Return concise logistics options and pricing. Do not address the end user directly.",
		}, " "),
		MaxTokens:   900,
		Temperature: exampleutil.Float32Ptr(0.2),
		MaxHistory:  20,
		MaxSteps:    6,
	})
	bookingAgent.RegisterTool(flightSearchTool())
	bookingAgent.RegisterTool(hotelSearchTool())
	bookingAgent.RegisterTool(itineraryEstimateTool())

	registry := agent.NewRegistry()
	registry.Register(travelAdvisor)
	registry.Register(destinationExpert)
	registry.Register(bookingAgent)

	rt := agent.NewRuntime(agent.RuntimeConfig{
		Provider:         provider,
		Agent:            travelAdvisor,
		SubagentResolver: registry,
		Middleware: []agent.Middleware{
			agent.CostGuardMiddleware(budget),
			agent.RetryMiddleware(3, 500*time.Millisecond),
		},
		MaxSteps:           12,
		MaxToolConcurrency: 2,
		Tracer:             tracer,
		Logger:             logger,
		UserID:             "opik-example-user",
		AgentID:            travelAdvisor.Name(),
	})

	fmt.Println("=== Multi-Agent Travel Planner with Opik Tracing ===")
	fmt.Printf("Provider: %s | Model: %s\n", provider.Name(), model)
	fmt.Println("Tracing: Opik OTLP HTTP")
	fmt.Println("Mode: travel_advisor consults destination_expert and booking_agent, then replies directly.")
	fmt.Println("Available consults: destination_expert, booking_agent")
	fmt.Println("Try: I want to visit Kyoto in early April for 5 days. What should I see and what will it cost?")
	fmt.Println("Try: Compare a week in Lisbon vs Barcelona in September — weather, costs, and top attractions.")
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
		fmt.Printf("\nTravel> %s\n\n", response)
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

func destinationInfoTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "lookup_destination",
		Description: "Looks up destination highlights, top attractions, and travel tips for a city or region.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination"},
			Properties: map[string]*schema.Schema{
				"destination": {Type: "string", Description: "City or region name."},
				"interests":   {Type: "string", Description: "Travel interests, for example history, food, nature."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination":     input["destination"],
				"top_attractions": []string{"historic temples", "local food markets", "scenic gardens", "cultural museums"},
				"best_time":       "March-May and October-November",
				"local_tips":      "Carry cash for small shops, remove shoes in temples, learn basic greetings.",
			}, nil
		},
	}
}

func weatherForecastTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_weather",
		Description: "Checks typical weather conditions for a destination during a given period.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination"},
			Properties: map[string]*schema.Schema{
				"destination": {Type: "string", Description: "City or region name."},
				"month":       {Type: "string", Description: "Month or period, for example early April."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination": input["destination"],
				"period":      input["month"],
				"avg_temp_c":  15,
				"conditions":  "mild, occasional rain, cherry blossom season",
				"packing_tip": "Layered clothing, light rain jacket, comfortable walking shoes.",
			}, nil
		},
	}
}

func visaRequirementsTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "check_visa_requirements",
		Description: "Checks visa and entry requirements for a destination by passport origin.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination"},
			Properties: map[string]*schema.Schema{
				"destination":   {Type: "string", Description: "Destination country."},
				"passport_from": {Type: "string", Description: "Passport issuing country."},
				"purpose":       {Type: "string", Description: "Purpose of visit, for example tourism, business."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination":   input["destination"],
				"passport_from": input["passport_from"],
				"visa_required": false,
				"max_stay_days": 90,
				"entry_note":    "Ensure passport validity exceeds 6 months from entry date.",
				"purpose":       input["purpose"],
			}, nil
		},
	}
}

func flightSearchTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "search_flights",
		Description: "Searches for flights between two cities on given dates.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"origin", "destination", "date"},
			Properties: map[string]*schema.Schema{
				"origin":      {Type: "string", Description: "Departure city or airport code."},
				"destination": {Type: "string", Description: "Arrival city or airport code."},
				"date":        {Type: "string", Description: "Travel date or date range."},
				"flexible":    {Type: "boolean", Description: "Whether dates are flexible (±3 days)."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"origin":           input["origin"],
				"destination":      input["destination"],
				"date":             input["date"],
				"best_option":      "Direct flight, 12h total, $850 round-trip",
				"budget_option":    "1 stop, 16h total, $560 round-trip",
				"booking_deadline": "Book 6-8 weeks ahead for best fares.",
			}, nil
		},
	}
}

func hotelSearchTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "search_hotels",
		Description: "Searches for hotels in a destination for given dates and budget range.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination"},
			Properties: map[string]*schema.Schema{
				"destination": {Type: "string", Description: "City or district name."},
				"check_in":    {Type: "string", Description: "Check-in date or period."},
				"nights":      {Type: "number", Description: "Number of nights."},
				"budget":      {Type: "string", Description: "Budget level: budget, mid-range, luxury."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination": input["destination"],
				"nights":      input["nights"],
				"mid_range":   "$120-180/night, central district, walking distance to transit",
				"luxury":      "$280-400/night, boutique hotel, on-site onsen",
				"budget_stay": "$60-90/night, guesthouse, shared bath",
			}, nil
		},
	}
}

func itineraryEstimateTool() tools.ToolDef {
	return tools.ToolDef{
		Name:        "estimate_itinerary",
		Description: "Estimates total trip cost and builds a day-by-day itinerary outline.",
		Schema: &schema.Schema{
			Type:     "object",
			Required: []string{"destination", "days"},
			Properties: map[string]*schema.Schema{
				"destination": {Type: "string", Description: "Trip destination."},
				"days":        {Type: "number", Description: "Number of trip days."},
				"style":       {Type: "string", Description: "Travel style: budget, mid-range, luxury."},
				"origin":      {Type: "string", Description: "Departure city for flight estimate."},
			},
		},
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			return map[string]any{
				"destination":     input["destination"],
				"days":            input["days"],
				"estimated_total": "$2,200-3,000 per person (mid-range)",
				"breakdown": map[string]string{
					"flights":       "$850 round-trip",
					"accommodation": "$150/night × 5",
					"activities":    "$40-80/day",
					"food":          "$40-60/day",
					"transport":     "$15-25/day",
				},
				"tip": "Book 2 months ahead for best rates; consider a rail pass for day trips.",
			}, nil
		},
	}
}
