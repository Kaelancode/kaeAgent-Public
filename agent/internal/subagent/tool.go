package subagent

import (
	"fmt"
	"strings"

	"github.com/Kaelancode/kaeAgent-Public/schema"
)

const ConsultToolPrefix = "consult_"
const TransferToolPrefix = "transfer_to_"
const TransferReasonMetadataKey = "transfer_reason"
const ConsultReasonMetadataKey = "consult_reason"

type ConsultPayload struct {
	AgentName string
	Input     string
	Metadata  map[string]string
}

type TransferPayload struct {
	AgentName string
	Input     string
	Metadata  map[string]string
}

var ConsultToolSchema = &schema.Schema{
	Type:     "object",
	Required: []string{"input"},
	Properties: map[string]*schema.Schema{
		"input": {
			Type:        "string",
			Description: "Task, question, or context to send to the consulted subagent.",
		},
		"reason": {
			Type:        "string",
			Description: "Why this consult is being requested.",
		},
		"metadata": {
			Type:        "object",
			Description: "Additional consult metadata as string key-value pairs.",
		},
	},
}

var TransferToolSchema = &schema.Schema{
	Type: "object",
	Properties: map[string]*schema.Schema{
		"input": {
			Type:        "string",
			Description: "Task or context to give the target subagent after transfer.",
		},
		"reason": {
			Type:        "string",
			Description: "Why the transfer is happening.",
		},
		"metadata": {
			Type:        "object",
			Description: "Additional transfer metadata as string key-value pairs.",
		},
	},
}

func ConsultToolName(agentName string) string {
	return ConsultToolPrefix + agentName
}

func TransferToolName(agentName string) string {
	return TransferToolPrefix + agentName
}

func ParseConsultPayload(target string, input map[string]any) (ConsultPayload, error) {
	payload := ConsultPayload{AgentName: target}

	rawInput, ok := input["input"]
	if !ok {
		return ConsultPayload{}, fmt.Errorf("runtime: consult input field is required")
	}
	consultInput, ok := rawInput.(string)
	if !ok {
		return ConsultPayload{}, fmt.Errorf("runtime: consult input field must be a string")
	}
	if strings.TrimSpace(consultInput) == "" {
		return ConsultPayload{}, fmt.Errorf("runtime: consult input field is required")
	}
	payload.Input = consultInput

	metadata, err := parseMetadata(input, "consult", ConsultReasonMetadataKey)
	if err != nil {
		return ConsultPayload{}, err
	}
	payload.Metadata = metadata

	return payload, nil
}

func ParseTransferPayload(target string, input map[string]any, fallbackInput string) (TransferPayload, error) {
	payload := TransferPayload{
		AgentName: target,
		Input:     fallbackInput,
	}

	if rawInput, ok := input["input"]; ok {
		vs, ok := rawInput.(string)
		if !ok {
			return TransferPayload{}, fmt.Errorf("runtime: transfer input field must be a string")
		}
		if strings.TrimSpace(vs) != "" {
			payload.Input = vs
		}
	}

	metadata, err := parseMetadata(input, "transfer", TransferReasonMetadataKey)
	if err != nil {
		return TransferPayload{}, err
	}
	payload.Metadata = metadata

	return payload, nil
}

func parseMetadata(input map[string]any, mode, reasonKey string) (map[string]string, error) {
	metadata := map[string]string{}
	if rawReason, ok := input["reason"]; ok {
		reason, ok := rawReason.(string)
		if !ok {
			return nil, fmt.Errorf("runtime: %s reason field must be a string", mode)
		}
		if strings.TrimSpace(reason) != "" {
			metadata[reasonKey] = reason
		}
	}
	if rawMeta, ok := input["metadata"]; ok {
		meta, ok := rawMeta.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("runtime: %s metadata field must be an object of string values", mode)
		}
		for k, v := range meta {
			if k == "" {
				continue
			}
			vs, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("runtime: %s metadata value for %q must be a string", mode, k)
			}
			metadata[k] = vs
		}
	}
	if len(metadata) == 0 {
		return nil, nil
	}
	return metadata, nil
}
