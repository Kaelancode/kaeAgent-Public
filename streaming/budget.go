// streaming/budget.go
package streaming

import (
	"fmt"
	"sync"
)

// Budget tracks cumulative token usage and cost across an agent session.
type Budget struct {
	mu           sync.Mutex
	maxTokens    int
	maxCostUSD   float64
	totalInput   int
	totalOutput  int
	totalCostUSD float64

	costPerInputToken  float64
	costPerOutputToken float64
}

// BudgetConfig configures a token/cost budget.
type BudgetConfig struct {
	MaxTokens          int     `json:"max_tokens"`
	MaxCostUSD         float64 `json:"max_cost_usd"`
	CostPerInputToken  float64 `json:"cost_per_input_token"`
	CostPerOutputToken float64 `json:"cost_per_output_token"`
}

// NewBudget creates a budget tracker with the given limits.
func NewBudget(cfg BudgetConfig) *Budget {
	return &Budget{
		maxTokens:          cfg.MaxTokens,
		maxCostUSD:         cfg.MaxCostUSD,
		costPerInputToken:  cfg.CostPerInputToken,
		costPerOutputToken: cfg.CostPerOutputToken,
	}
}

// Add records token usage and updates the running cost.
func (b *Budget) Add(inputTokens, outputTokens int) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.totalInput += inputTokens
	b.totalOutput += outputTokens
	b.totalCostUSD += float64(inputTokens)*b.costPerInputToken + float64(outputTokens)*b.costPerOutputToken
}

// Check returns an error if the budget has been exceeded.
func (b *Budget) Check() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	total := b.totalInput + b.totalOutput
	if b.maxTokens > 0 && total > b.maxTokens {
		return fmt.Errorf("budget: token limit exceeded (%d/%d)", total, b.maxTokens)
	}
	if b.maxCostUSD > 0 && b.totalCostUSD > b.maxCostUSD {
		return fmt.Errorf("budget: cost limit exceeded ($%.4f/$%.4f)", b.totalCostUSD, b.maxCostUSD)
	}
	return nil
}

// Usage returns current cumulative usage.
func (b *Budget) Usage() (inputTokens, outputTokens, totalTokens int, costUSD float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.totalInput, b.totalOutput, b.totalInput + b.totalOutput, b.totalCostUSD
}

// Remaining returns remaining tokens and cost. Returns -1 if no limit is set.
func (b *Budget) Remaining() (tokens int, costUSD float64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	tokens = -1
	if b.maxTokens > 0 {
		tokens = b.maxTokens - (b.totalInput + b.totalOutput)
		if tokens < 0 {
			tokens = 0
		}
	}

	costUSD = -1
	if b.maxCostUSD > 0 {
		costUSD = b.maxCostUSD - b.totalCostUSD
		if costUSD < 0 {
			costUSD = 0
		}
	}

	return tokens, costUSD
}

// Reset clears all accumulated usage.
func (b *Budget) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.totalInput = 0
	b.totalOutput = 0
	b.totalCostUSD = 0
}

// BudgetSnapshot is the serializable representation of a Budget.
type BudgetSnapshot struct {
	MaxTokens          int     `json:"max_tokens"`
	MaxCostUSD         float64 `json:"max_cost_usd"`
	TotalInput         int     `json:"total_input"`
	TotalOutput        int     `json:"total_output"`
	TotalCostUSD       float64 `json:"total_cost_usd"`
	CostPerInputToken  float64 `json:"cost_per_input_token"`
	CostPerOutputToken float64 `json:"cost_per_output_token"`
}

// Snapshot returns a serializable copy of the budget state.
func (b *Budget) Snapshot() BudgetSnapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	return BudgetSnapshot{
		MaxTokens:          b.maxTokens,
		MaxCostUSD:         b.maxCostUSD,
		TotalInput:         b.totalInput,
		TotalOutput:        b.totalOutput,
		TotalCostUSD:       b.totalCostUSD,
		CostPerInputToken:  b.costPerInputToken,
		CostPerOutputToken: b.costPerOutputToken,
	}
}

// Restore restores budget state from a snapshot.
func (b *Budget) Restore(s BudgetSnapshot) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.maxTokens = s.MaxTokens
	b.maxCostUSD = s.MaxCostUSD
	b.totalInput = s.TotalInput
	b.totalOutput = s.TotalOutput
	b.totalCostUSD = s.TotalCostUSD
	b.costPerInputToken = s.CostPerInputToken
	b.costPerOutputToken = s.CostPerOutputToken
}

// NewBudgetFromSnapshot creates a Budget from a snapshot.
func NewBudgetFromSnapshot(s BudgetSnapshot) *Budget {
	return &Budget{
		maxTokens:          s.MaxTokens,
		maxCostUSD:         s.MaxCostUSD,
		totalInput:         s.TotalInput,
		totalOutput:        s.TotalOutput,
		totalCostUSD:       s.TotalCostUSD,
		costPerInputToken:  s.CostPerInputToken,
		costPerOutputToken: s.CostPerOutputToken,
	}
}
