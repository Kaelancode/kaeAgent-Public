// Package workflow provides deterministic application-owned workflow helpers.
//
// Use workflow when application code or a workflow engine has already selected
// the agent step to execute. Model-driven subagent consult and transfer live in
// package agent, where declared subagents are exposed to the model as synthetic
// tools during Runtime.Run or Runtime.Stream.
package workflow
