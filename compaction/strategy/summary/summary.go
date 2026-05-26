package summary

import (
	"context"
	"fmt"
	"strings"

	"github.com/yourorg/agent-sdk/compaction"
	"github.com/yourorg/agent-sdk/llm"
)

const (
	defaultSummaryRole     = "system"
	defaultSummaryPrefix   = "Conversation summary:"
	defaultMaxSummaryChars = 2000
	defaultPerMessageChars = 240
)

type SummarizerFunc func(ctx context.Context, turns [][]llm.Message) (string, error)

type Strategy struct {
	RecentTurns     int
	Summarizer      SummarizerFunc
	SummaryRole     string
	SummaryPrefix   string
	MaxSummaryChars int
}

func New(recentTurns int, summarizer SummarizerFunc) *Strategy {
	return &Strategy{
		RecentTurns:     recentTurns,
		Summarizer:      summarizer,
		SummaryRole:     defaultSummaryRole,
		SummaryPrefix:   defaultSummaryPrefix,
		MaxSummaryChars: defaultMaxSummaryChars,
	}
}

func (s *Strategy) Name() string {
	return "summary"
}

func (s *Strategy) Compact(ctx context.Context, input compaction.Input) (compaction.Output, error) {
	systemMsgs, turns := splitTurns(input.Messages)
	systemMsgs, priorSummary := s.partitionSystemMessages(systemMsgs)
	if s.RecentTurns <= 0 || len(turns) <= s.RecentTurns {
		return compaction.Output{Messages: cloneMessages(input.Messages)}, nil
	}

	cutoff := len(turns) - s.RecentTurns
	olderTurns := turns[:cutoff]
	recentTurns := turns[cutoff:]

	summaryText, err := s.summarize(ctx, priorSummary, olderTurns)
	if err != nil {
		return compaction.Output{}, err
	}

	out := cloneMessages(systemMsgs)
	if summaryText != "" {
		out = append(out, llm.Message{
			Role:    summaryRole(s.SummaryRole),
			Content: withPrefix(summaryPrefix(s.SummaryPrefix), summaryText),
		})
	}
	for _, turn := range recentTurns {
		out = append(out, turn...)
	}

	return compaction.Output{
		Messages:  out,
		Compacted: len(out) != len(input.Messages),
		Reason:    "older turns summarized",
	}, nil
}

func (s *Strategy) partitionSystemMessages(messages []llm.Message) ([]llm.Message, string) {
	role := summaryRole(s.SummaryRole)
	prefix := summaryPrefix(s.SummaryPrefix)
	retained := make([]llm.Message, 0, len(messages))
	var summaryParts []string

	for _, msg := range messages {
		if msg.Role == role {
			if summaryText, ok := trimSummaryPrefix(msg.Content, prefix); ok {
				if summaryText != "" {
					summaryParts = append(summaryParts, summaryText)
				}
				continue
			}
		}
		retained = append(retained, msg)
	}

	return retained, strings.TrimSpace(strings.Join(summaryParts, "\n"))
}

func (s *Strategy) summarize(ctx context.Context, priorSummary string, turns [][]llm.Message) (string, error) {
	summarizer := s.Summarizer
	if summarizer == nil {
		text, err := s.defaultSummarizer(ctx, turns)
		if err != nil {
			return "", err
		}
		return strings.TrimSpace(mergeSummaryText(priorSummary, text)), nil
	}

	inputTurns := cloneTurns(turns)
	if priorSummary != "" {
		inputTurns = append([][]llm.Message{{
			{
				Role:    summaryRole(s.SummaryRole),
				Content: withPrefix(summaryPrefix(s.SummaryPrefix), priorSummary),
			},
		}}, inputTurns...)
	}

	text, err := summarizer(ctx, inputTurns)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(text), nil
}

func (s *Strategy) defaultSummarizer(_ context.Context, turns [][]llm.Message) (string, error) {
	limit := s.MaxSummaryChars
	if limit <= 0 {
		limit = defaultMaxSummaryChars
	}

	var b strings.Builder
	for i, turn := range turns {
		line := formatTurn(i+1, turn)
		if line == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		if b.Len()+len(line) > limit {
			remaining := limit - b.Len()
			if remaining <= 0 {
				break
			}
			b.WriteString(truncate(line, remaining))
			break
		}
		b.WriteString(line)
	}
	return b.String(), nil
}

func Factory(config map[string]any) (compaction.Strategy, error) {
	recentTurns := intValue(config, "recent_turns", 0)
	if recentTurns == 0 {
		recentTurns = intValue(config, "max_turns", 0)
	}
	strategy := New(recentTurns, nil)
	if role, ok := stringValue(config, "summary_role"); ok {
		strategy.SummaryRole = role
	}
	if prefix, ok := stringValue(config, "summary_prefix"); ok {
		strategy.SummaryPrefix = prefix
	}
	if maxChars := intValue(config, "max_summary_chars", 0); maxChars > 0 {
		strategy.MaxSummaryChars = maxChars
	}
	return strategy, nil
}

func splitTurns(messages []llm.Message) ([]llm.Message, [][]llm.Message) {
	systemMsgs := make([]llm.Message, 0, len(messages))
	turns := make([][]llm.Message, 0)
	current := make([]llm.Message, 0)

	for _, m := range messages {
		if m.Role == "system" {
			systemMsgs = append(systemMsgs, m)
			continue
		}
		if m.Role == "user" {
			if len(current) > 0 {
				turns = append(turns, current)
			}
			current = []llm.Message{m}
			continue
		}
		if len(current) == 0 {
			current = []llm.Message{m}
			continue
		}
		current = append(current, m)
	}

	if len(current) > 0 {
		turns = append(turns, current)
	}

	return systemMsgs, turns
}

func formatTurn(turnNum int, turn []llm.Message) string {
	if len(turn) == 0 {
		return ""
	}

	parts := make([]string, 0, len(turn)+1)
	parts = append(parts, fmt.Sprintf("Turn %d:", turnNum))
	for _, m := range turn {
		label := roleLabel(m)
		content := truncate(strings.TrimSpace(m.Content), defaultPerMessageChars)
		if content == "" && len(m.ToolCalls) > 0 {
			content = fmt.Sprintf("%d tool call(s)", len(m.ToolCalls))
		}
		if content == "" {
			content = "(empty)"
		}
		parts = append(parts, fmt.Sprintf("%s %s", label, content))
	}
	return strings.Join(parts, " ")
}

func roleLabel(m llm.Message) string {
	switch m.Role {
	case "user":
		return "User:"
	case "assistant":
		return "Assistant:"
	case "tool":
		if m.Name != "" {
			return fmt.Sprintf("Tool(%s):", m.Name)
		}
		return "Tool:"
	default:
		return strings.Title(m.Role) + ":"
	}
}

func withPrefix(prefix, text string) string {
	prefix = strings.TrimSpace(prefix)
	text = strings.TrimSpace(text)
	if prefix == "" {
		return text
	}
	if text == "" {
		return prefix
	}
	return prefix + "\n" + text
}

func trimSummaryPrefix(content, prefix string) (string, bool) {
	content = strings.TrimSpace(content)
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return content, false
	}
	if content == prefix {
		return "", true
	}
	if content == "" || !strings.HasPrefix(content, prefix) {
		return "", false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(content, prefix))
	return rest, true
}

func mergeSummaryText(parts ...string) string {
	merged := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		merged = append(merged, part)
	}
	return strings.Join(merged, "\n")
}

func summaryRole(role string) string {
	role = strings.TrimSpace(role)
	if role == "" {
		return defaultSummaryRole
	}
	return role
}

func summaryPrefix(prefix string) string {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return defaultSummaryPrefix
	}
	return prefix
}

func truncate(s string, limit int) string {
	if limit <= 0 || len(s) <= limit {
		return s
	}
	if limit <= 3 {
		return s[:limit]
	}
	return s[:limit-3] + "..."
}

func cloneMessages(messages []llm.Message) []llm.Message {
	out := make([]llm.Message, len(messages))
	copy(out, messages)
	return out
}

func cloneTurns(turns [][]llm.Message) [][]llm.Message {
	out := make([][]llm.Message, len(turns))
	for i, turn := range turns {
		out[i] = cloneMessages(turn)
	}
	return out
}

func intValue(config map[string]any, key string, fallback int) int {
	if config == nil {
		return fallback
	}
	switch v := config[key].(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return fallback
	}
}

func stringValue(config map[string]any, key string) (string, bool) {
	if config == nil {
		return "", false
	}
	v, ok := config[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	return s, true
}
