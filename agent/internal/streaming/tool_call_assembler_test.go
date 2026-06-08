package streaming

import (
	"testing"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

func TestToolCallAssembler(t *testing.T) {
	a := NewToolCallAssembler()

	a.AddFragment(0, &llm.ToolCallDelta{ID: "call_1", Name: "get_weather"})
	a.AddFragment(0, &llm.ToolCallDelta{Input: `{"city": "NYC"}`})

	calls, err := a.Assemble()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].ID != "call_1" {
		t.Errorf("expected ID 'call_1', got %q", calls[0].ID)
	}
	if calls[0].Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %q", calls[0].Name)
	}
	if calls[0].Input["city"] != "NYC" {
		t.Errorf("expected input city=NYC, got %v", calls[0].Input)
	}
}

func TestToolCallAssembler_MultipleFragments(t *testing.T) {
	a := NewToolCallAssembler()

	a.AddFragment(0, &llm.ToolCallDelta{ID: "call_1", Name: "search"})
	a.AddFragment(0, &llm.ToolCallDelta{Input: `{"qu`})
	a.AddFragment(0, &llm.ToolCallDelta{Input: `ery": "test"}`})

	calls, err := a.Assemble()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}
	if calls[0].Input["query"] != "test" {
		t.Errorf("expected query=test, got %v", calls[0].Input)
	}
}

func TestToolCallAssembler_Empty(t *testing.T) {
	a := NewToolCallAssembler()
	calls, err := a.Assemble()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != nil {
		t.Errorf("expected nil calls, got %v", calls)
	}
}

func TestToolCallAssembler_Reset(t *testing.T) {
	a := NewToolCallAssembler()
	a.AddFragment(0, &llm.ToolCallDelta{ID: "call_1", Name: "test"})
	a.Reset()

	calls, err := a.Assemble()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != nil {
		t.Errorf("expected nil calls after reset, got %v", calls)
	}
}
