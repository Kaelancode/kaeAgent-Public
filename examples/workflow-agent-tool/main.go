package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/workflow"
)

func main() {
	ctx := context.Background()
	provider := workflowProvider{}

	outlineTool := workflow.WorkflowAgentTool(workflow.AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "outline",
			Model:        "workflow-demo",
			SystemPrompt: "You write compact outlines.",
			MaxSteps:     1,
		}),
		Name:        "outline",
		Description: "Create a compact outline for a writing task.",
		Tags:        []string{"workflow", "writing"},
	}, provider)
	expandTool := workflow.WorkflowAgentTool(workflow.AgentConfig{
		Agent: agent.NewAgent(agent.AgentConfig{
			Name:         "expand",
			Model:        "workflow-demo",
			SystemPrompt: "You expand outlines into short prose.",
			MaxSteps:     1,
		}),
		Name:        "expand",
		Description: "Expand an outline into short prose.",
		Tags:        []string{"workflow", "writing"},
	}, provider)

	task := "Write a detailed launch note for a new workflow API."
	outline, err := outlineTool.Handler(ctx, map[string]any{"message": task})
	if err != nil {
		panic(err)
	}

	fmt.Printf("Outline:\n%v\n\n", outline)

	if needsExpansion(task) {
		expanded, err := expandTool.Handler(ctx, map[string]any{
			"message": fmt.Sprintf("Expand this outline:\n%v", outline),
		})
		if err != nil {
			panic(err)
		}
		fmt.Printf("Expanded:\n%v\n", expanded)
	}
}

func needsExpansion(task string) bool {
	return strings.Contains(strings.ToLower(task), "detailed")
}

type workflowProvider struct{}

var _ llm.Provider = workflowProvider{}

func (workflowProvider) Complete(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	text := "workflow result"
	switch {
	case strings.Contains(req.Messages[0].Content, "compact outlines"):
		text = "- Problem\n- Workflow API\n- Migration note"
	case strings.Contains(req.Messages[0].Content, "expand outlines"):
		text = "The workflow API separates deterministic orchestration from model-driven subagent calls. Existing callers can migrate gradually while keeping compatibility wrappers."
	}

	return &llm.Response{
		Content:      []llm.ContentBlock{{Type: "text", Text: text}},
		FinishReason: "stop",
	}, nil
}

func (workflowProvider) Stream(ctx context.Context, _ *llm.Request) (<-chan llm.Event, error) {
	ch := make(chan llm.Event, 1)
	if err := ctx.Err(); err != nil {
		close(ch)
		return ch, err
	}
	ch <- llm.Event{Kind: llm.EventDone}
	close(ch)
	return ch, nil
}

func (workflowProvider) Models(context.Context) ([]llm.ModelInfo, error) {
	return []llm.ModelInfo{{ID: "workflow-demo", Name: "workflow-demo", Provider: "workflow"}}, nil
}

func (workflowProvider) Name() string {
	return "workflow"
}
