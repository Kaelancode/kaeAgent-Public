package subagent

import "testing"

func TestParseConsultPayload(t *testing.T) {
	payload, err := ParseConsultPayload("billing", map[string]any{
		"input":  "check invoice",
		"reason": "billing question",
		"metadata": map[string]any{
			"priority": "high",
		},
	})
	if err != nil {
		t.Fatalf("ParseConsultPayload: %v", err)
	}
	if payload.AgentName != "billing" || payload.Input != "check invoice" {
		t.Fatalf("unexpected payload: %#v", payload)
	}
	if payload.Metadata[ConsultReasonMetadataKey] != "billing question" {
		t.Fatalf("expected consult reason metadata, got %#v", payload.Metadata)
	}
	if payload.Metadata["priority"] != "high" {
		t.Fatalf("expected priority metadata, got %#v", payload.Metadata)
	}
}

func TestParseConsultPayloadRejectsMissingInput(t *testing.T) {
	_, err := ParseConsultPayload("billing", map[string]any{})
	if err == nil {
		t.Fatal("expected missing input error")
	}
}

func TestParseTransferPayloadUsesFallbackInput(t *testing.T) {
	payload, err := ParseTransferPayload("billing", map[string]any{
		"reason": "needs ownership",
	}, "fallback")
	if err != nil {
		t.Fatalf("ParseTransferPayload: %v", err)
	}
	if payload.Input != "fallback" {
		t.Fatalf("expected fallback input, got %q", payload.Input)
	}
	if payload.Metadata[TransferReasonMetadataKey] != "needs ownership" {
		t.Fatalf("expected transfer reason metadata, got %#v", payload.Metadata)
	}
}

func TestParseTransferPayloadRejectsNonStringMetadataValue(t *testing.T) {
	_, err := ParseTransferPayload("billing", map[string]any{
		"metadata": map[string]any{"priority": 1},
	}, "")
	if err == nil {
		t.Fatal("expected metadata value type error")
	}
}
