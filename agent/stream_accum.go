package agent

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yourorg/agent-sdk/llm"
	"github.com/yourorg/agent-sdk/tools"
)

type toolCallAccum struct {
	id    string
	name  strings.Builder
	input strings.Builder
}

type toolCallAssembler struct {
	accumulators map[int]*toolCallAccum
	order        []int
}

func newToolCallAssembler() *toolCallAssembler {
	return &toolCallAssembler{
		accumulators: make(map[int]*toolCallAccum),
	}
}

func (a *toolCallAssembler) addFragment(idx int, delta *llm.ToolCallDelta) {
	acc, ok := a.accumulators[idx]
	if !ok {
		acc = &toolCallAccum{}
		a.accumulators[idx] = acc
		a.order = append(a.order, idx)
	}
	if delta.ID != "" {
		acc.id = delta.ID
	}
	if delta.Name != "" {
		acc.name.WriteString(delta.Name)
	}
	if delta.Input != "" {
		acc.input.WriteString(delta.Input)
	}
}

func (a *toolCallAssembler) assemble() ([]tools.ToolCall, error) {
	if len(a.order) == 0 {
		return nil, nil
	}

	calls := make([]tools.ToolCall, len(a.order))
	for i, idx := range a.order {
		acc := a.accumulators[idx]

		var input map[string]any
		raw := acc.input.String()
		if raw != "" {
			if err := json.Unmarshal([]byte(raw), &input); err != nil {
				return nil, fmt.Errorf("agent: failed to parse tool call input for %q: %w", acc.name.String(), err)
			}
		}

		callID := acc.id
		if callID == "" {
			callID = fmt.Sprintf("call_%d", idx)
		}

		calls[i] = tools.ToolCall{
			ID:    callID,
			Name:  acc.name.String(),
			Input: input,
		}
	}
	return calls, nil
}

func (a *toolCallAssembler) reset() {
	a.accumulators = make(map[int]*toolCallAccum)
	a.order = nil
}
