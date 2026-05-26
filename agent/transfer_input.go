package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yourorg/agent-sdk/llm"
)

const (
	DefaultTransferHistoryStart = "<TRANSFER HISTORY>"
	DefaultTransferHistoryEnd   = "</TRANSFER HISTORY>"
)

type TransferInputData struct {
	Session  SessionSnapshot
	Messages []llm.Message
	Input    string
	Metadata map[string]string
}

type TransferInputFilter func(TransferInputData) (TransferInputData, error)

func ComposeTransferInput(filters ...TransferInputFilter) TransferInputFilter {
	return func(data TransferInputData) (TransferInputData, error) {
		current := data
		var err error
		for _, filter := range filters {
			if filter == nil {
				continue
			}
			current, err = filter(current)
			if err != nil {
				return current, err
			}
		}
		return current, nil
	}
}

func PassThroughTransferInput() TransferInputFilter {
	return func(data TransferInputData) (TransferInputData, error) {
		data.Session = cloneSessionSnapshot(data.Session)
		data.Messages = cloneMessages(data.Messages)
		data.Metadata = cloneStringMap(data.Metadata)
		return data, nil
	}
}

func RemoveToolTransferInput() TransferInputFilter {
	return func(data TransferInputData) (TransferInputData, error) {
		filtered := make([]llm.Message, 0, len(data.Messages))
		for _, msg := range data.Messages {
			if msg.Role == "tool" {
				continue
			}
			if len(msg.ToolCalls) > 0 || msg.ToolCallID != "" {
				continue
			}
			filtered = append(filtered, cloneMessage(msg))
		}
		data.Session = cloneSessionSnapshot(data.Session)
		data.Messages = filtered
		data.Metadata = cloneStringMap(data.Metadata)
		return data, nil
	}
}

func RecentWindowTransferInput(n int) TransferInputFilter {
	return func(data TransferInputData) (TransferInputData, error) {
		data.Session = cloneSessionSnapshot(data.Session)
		data.Metadata = cloneStringMap(data.Metadata)

		if n <= 0 {
			filtered := make([]llm.Message, 0, len(data.Messages))
			for _, msg := range data.Messages {
				if msg.Role == "system" {
					filtered = append(filtered, cloneMessage(msg))
				}
			}
			data.Messages = filtered
			return data, nil
		}

		systemMsgs := make([]llm.Message, 0, len(data.Messages))
		nonSystem := make([]llm.Message, 0, len(data.Messages))
		for _, msg := range data.Messages {
			if msg.Role == "system" {
				systemMsgs = append(systemMsgs, cloneMessage(msg))
				continue
			}
			nonSystem = append(nonSystem, cloneMessage(msg))
		}

		if len(nonSystem) > n {
			nonSystem = nonSystem[len(nonSystem)-n:]
		}

		data.Messages = append(systemMsgs, nonSystem...)
		return data, nil
	}
}

func NestTransferHistory() TransferInputFilter {
	return func(data TransferInputData) (TransferInputData, error) {
		data.Session = cloneSessionSnapshot(data.Session)
		data.Metadata = cloneStringMap(data.Metadata)

		systemMsgs := make([]llm.Message, 0, len(data.Messages))
		transcript := make([]string, 0, len(data.Messages))
		for i, msg := range data.Messages {
			if msg.Role == "system" {
				systemMsgs = append(systemMsgs, cloneMessage(msg))
				continue
			}
			transcript = append(transcript, fmt.Sprintf("%d. %s", i+1, formatTransferTranscriptMessage(msg)))
		}

		if len(transcript) == 0 {
			data.Messages = systemMsgs
			return data, nil
		}

		content := strings.Join([]string{
			"For context, here is the conversation so far between the user and the previous agent:",
			DefaultTransferHistoryStart,
			strings.Join(transcript, "\n"),
			DefaultTransferHistoryEnd,
		}, "\n")

		data.Messages = append(systemMsgs, llm.Message{
			Role:    "assistant",
			Content: content,
		})
		return data, nil
	}
}

func cloneSessionSnapshot(snap SessionSnapshot) SessionSnapshot {
	return SessionSnapshot{
		ID:       snap.ID,
		Config:   cloneSessionConfig(snap.Config),
		Budget:   snap.Budget,
		Metadata: cloneStringMap(snap.Metadata),
	}
}

func formatTransferTranscriptMessage(msg llm.Message) string {
	if msg.Content != "" {
		return fmt.Sprintf("%s: %s", msg.Role, msg.Content)
	}
	if len(msg.ToolCalls) > 0 {
		payload, _ := json.Marshal(msg.ToolCalls)
		return fmt.Sprintf("%s: %s", msg.Role, string(payload))
	}
	if msg.ToolCallID != "" {
		return fmt.Sprintf("%s: tool_call_id=%s", msg.Role, msg.ToolCallID)
	}
	return msg.Role
}
