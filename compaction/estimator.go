package compaction

import (
	"encoding/json"

	"github.com/Kaelancode/kaeAgent-Public/llm"
)

const DefaultCharsPerToken = 4
const defaultMessageOverheadChars = 32
const defaultToolsOverheadChars = 64

type TokenEstimator interface {
	Estimate(messages []llm.Message) int
}

type ApproxTokenEstimator struct {
	charsPerToken int
}

func NewApproxTokenEstimator(charsPerToken int) *ApproxTokenEstimator {
	if charsPerToken <= 0 {
		charsPerToken = DefaultCharsPerToken
	}
	return &ApproxTokenEstimator{charsPerToken: charsPerToken}
}

func (e *ApproxTokenEstimator) Estimate(messages []llm.Message) int {
	totalChars := 0
	for _, m := range messages {
		totalChars += len(m.Content) + len(m.Role) + len(m.Name) + len(m.ToolCallID)
		if len(m.ToolCalls) > 0 {
			data, _ := json.Marshal(m.ToolCalls)
			totalChars += len(data)
		}
	}
	if totalChars == 0 {
		return 0
	}
	return (totalChars + e.charsPerToken - 1) / e.charsPerToken
}

func EstimatePromptTokens(messages []llm.Message, tools []llm.ToolDef, estimator TokenEstimator) int {
	if estimator == nil {
		estimator = NewApproxTokenEstimator(DefaultCharsPerToken)
	}

	totalTokens := estimator.Estimate(messages)
	for range messages {
		totalTokens += charsToTokens(defaultMessageOverheadChars)
	}
	if len(tools) > 0 {
		data, _ := json.Marshal(tools)
		totalTokens += charsToTokens(len(data) + defaultToolsOverheadChars)
	}
	return totalTokens
}

func charsToTokens(totalChars int) int {
	if totalChars <= 0 {
		return 0
	}
	return (totalChars + DefaultCharsPerToken - 1) / DefaultCharsPerToken
}
