package workflow

import (
	"context"
	"fmt"
	"sync"

	"github.com/Kaelancode/kaeAgent-Public/agent"
	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/schema"
	"github.com/Kaelancode/kaeAgent-Public/tools"
)

type AgentConfig struct {
	Agent       *agent.Agent
	Name        string
	Description string
	Tags        []string
	MaxSteps    int
}

type JoinResult struct {
	Name   string
	Output string
	Err    error
}

func WorkflowAgentTool(cfg AgentConfig, provider llm.Provider) tools.ToolDef {
	minMsgLen := 1
	return tools.ToolDef{
		Name:        "agent_" + cfg.Name,
		Description: cfg.Description,
		Schema: &schema.Schema{
			Type: "object",
			Properties: map[string]*schema.Schema{
				"message": {
					Type:        "string",
					Description: "The task or question to send to the agent",
					MinLength:   &minMsgLen,
				},
			},
			Required: []string{"message"},
		},
		Tags: cfg.Tags,
		Handler: func(ctx context.Context, input map[string]any) (any, error) {
			message, ok := input["message"].(string)
			if !ok || message == "" {
				return nil, fmt.Errorf("workflow_agent_tool: 'message' field is required and must be a string")
			}

			maxSteps := cfg.MaxSteps
			if maxSteps <= 0 && cfg.Agent != nil {
				maxSteps = cfg.Agent.MaxSteps()
			}
			if maxSteps <= 0 {
				maxSteps = 10
			}

			rt := agent.NewRuntime(agent.RuntimeConfig{
				Provider: provider,
				Agent:    cfg.Agent,
				MaxSteps: maxSteps,
			})

			result, err := rt.Run(ctx, message)
			if err != nil {
				return nil, fmt.Errorf("workflow_agent_tool %q: %w", cfg.Name, err)
			}
			return result, nil
		},
	}
}

func JoinAll(ctx context.Context, tasks map[string]func(ctx context.Context) (string, error)) (map[string]string, error) {
	detailed, err := JoinAllDetailed(ctx, tasks)
	results := make(map[string]string, len(detailed))
	for name, result := range detailed {
		results[name] = result.Output
	}
	return results, err
}

func JoinAllDetailed(ctx context.Context, tasks map[string]func(ctx context.Context) (string, error)) (map[string]JoinResult, error) {
	type result struct {
		name   string
		output string
		err    error
	}

	childCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	ch := make(chan result, len(tasks))
	var wg sync.WaitGroup

	for name, fn := range tasks {
		wg.Add(1)
		go func(n string, f func(ctx context.Context) (string, error)) {
			defer wg.Done()
			out, err := f(childCtx)
			ch <- result{name: n, output: out, err: err}
		}(name, fn)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	results := make(map[string]JoinResult, len(tasks))
	var firstErr error
	for r := range ch {
		results[r.name] = JoinResult{
			Name:   r.name,
			Output: r.output,
			Err:    r.err,
		}
		if r.err != nil && firstErr == nil {
			firstErr = fmt.Errorf("workflow: step %q failed: %w", r.name, r.err)
			cancel()
		}
	}

	return results, firstErr
}
