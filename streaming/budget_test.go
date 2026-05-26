package streaming

import (
	"testing"
)

func TestBudget_Add(t *testing.T) {
	b := NewBudget(BudgetConfig{
		MaxTokens:          1000,
		CostPerInputToken:  0.00001,
		CostPerOutputToken: 0.00003,
	})

	b.Add(100, 50)
	input, output, total, cost := b.Usage()
	if input != 100 {
		t.Errorf("expected input=100, got %d", input)
	}
	if output != 50 {
		t.Errorf("expected output=50, got %d", output)
	}
	if total != 150 {
		t.Errorf("expected total=150, got %d", total)
	}
	expectedCost := 100*0.00001 + 50*0.00003
	if cost != expectedCost {
		t.Errorf("expected cost=%f, got %f", expectedCost, cost)
	}
}

func TestBudget_CheckTokenLimit(t *testing.T) {
	b := NewBudget(BudgetConfig{MaxTokens: 100})

	b.Add(60, 50)
	err := b.Check()
	if err == nil {
		t.Fatal("expected budget exceeded error")
	}
}

func TestBudget_CheckCostLimit(t *testing.T) {
	b := NewBudget(BudgetConfig{
		MaxCostUSD:         0.001,
		CostPerInputToken:  0.001,
		CostPerOutputToken: 0.001,
	})

	b.Add(1, 1)
	err := b.Check()
	if err == nil {
		t.Fatal("expected cost limit exceeded error")
	}
}

func TestBudget_CheckUnderLimit(t *testing.T) {
	b := NewBudget(BudgetConfig{MaxTokens: 1000})
	b.Add(10, 10)
	if err := b.Check(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestBudget_Remaining(t *testing.T) {
	b := NewBudget(BudgetConfig{MaxTokens: 1000, MaxCostUSD: 1.0})
	b.Add(100, 200)

	tokens, cost := b.Remaining()
	if tokens != 700 {
		t.Errorf("expected 700 remaining tokens, got %d", tokens)
	}
	if cost != 1.0 {
		t.Errorf("expected 1.0 remaining cost, got %f", cost)
	}
}

func TestBudget_Reset(t *testing.T) {
	b := NewBudget(BudgetConfig{MaxTokens: 1000})
	b.Add(500, 500)
	b.Reset()

	_, _, total, _ := b.Usage()
	if total != 0 {
		t.Errorf("expected 0 total after reset, got %d", total)
	}
}

func TestBudget_NoLimits(t *testing.T) {
	b := NewBudget(BudgetConfig{})
	b.Add(999999, 999999)
	if err := b.Check(); err != nil {
		t.Fatalf("no limits set, should not error: %v", err)
	}

	tokens, cost := b.Remaining()
	if tokens != -1 {
		t.Errorf("expected -1 for unlimited tokens, got %d", tokens)
	}
	if cost != -1 {
		t.Errorf("expected -1 for unlimited cost, got %f", cost)
	}
}
