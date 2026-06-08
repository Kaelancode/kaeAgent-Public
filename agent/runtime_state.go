package agent

import (
	"encoding/json"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/store"
	"github.com/Kaelancode/kaeAgent-Public/streaming"
	"github.com/Kaelancode/kaeAgent-Public/tools"
	"github.com/google/uuid"
)

type runState struct {
	generation  uint64
	sessionID   string
	runID       string
	config      SessionConfig
	metadata    map[string]string
	budget      *streaming.Budget
	conv        *Conversation
	userID      string
	agentID     string
	activeAgent *Agent
	rootAgent   *Agent
	tools       *tools.Registry
}

type runExecutor struct {
	rt *Runtime
	rs *runState
}

type runLoopResult struct {
	Response   *llm.Response
	ToolCalls  []tools.ToolCall
	Transfer   *TransferStep
	TokensUsed llm.Usage
	Text       string
}

func (r *Runtime) SessionSnapshot() SessionSnapshot {
	r.mu.RLock()
	session := r.session
	r.mu.RUnlock()
	if session == nil {
		return SessionSnapshot{Metadata: map[string]string{}}
	}
	return session.Snapshot()
}

func (r *Runtime) ConversationSnapshot() ConversationState {
	r.mu.RLock()
	conv := r.conv
	r.mu.RUnlock()
	if conv == nil {
		return ConversationState{}
	}
	return conv.Snapshot()
}

func (r *Runtime) ConversationMessages() []llm.Message {
	return r.ConversationSnapshot().Messages
}

func (r *Runtime) ConversationSlice(start, end int) []llm.Message {
	r.mu.RLock()
	conv := r.conv
	r.mu.RUnlock()
	return conv.Slice(start, end)
}

func (r *Runtime) AppendConversationMessage(msg llm.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.Append(msg)
}

func (r *Runtime) AppendConversationMessages(msgs []llm.Message) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.AppendAll(msgs)
}

func (r *Runtime) ClearConversation() {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.Clear()
}

func (r *Runtime) SetConversationSystem(content string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.conv == nil {
		return
	}
	r.generation++
	r.conv.SetSystem(content)
}

func (r *Runtime) SessionStoreData(userID, agentID string) *store.SessionData {
	snap := r.SessionSnapshot()
	configJSON, _ := json.Marshal(snap.Config)
	budgetJSON, _ := json.Marshal(snap.Budget)
	return &store.SessionData{
		ID:       snap.ID,
		UserID:   userID,
		AgentID:  agentID,
		Config:   configJSON,
		Budget:   budgetJSON,
		Metadata: cloneStringMap(snap.Metadata),
	}
}

func (r *Runtime) SetSessionMetadata(key, value string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.session == nil {
		return
	}
	r.generation++
	r.session.SetMeta(key, value)
}

func (r *Runtime) LoadState(sessionSnap SessionSnapshot, convState ConversationState) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.generation++
	r.session = NewSessionFromSnapshot(sessionSnap)
	r.conv = NewConversationFromState(convState)
}

func (r *Runtime) newRunExecutor() *runExecutor {
	return &runExecutor{
		rt: r,
		rs: r.captureRunState(),
	}
}

func (r *Runtime) captureRunState() *runState {
	r.mu.RLock()
	generation := r.generation
	session := r.session
	conv := r.conv
	userID := r.userID
	agentID := r.agentID
	activeAgent := r.agent
	rootAgent := r.rootAgent
	toolRegistry := r.tools
	r.mu.RUnlock()

	sessionSnap := session.Snapshot()
	convState := conv.Snapshot()

	return &runState{
		generation:  generation,
		sessionID:   sessionSnap.ID,
		runID:       uuid.NewString(),
		config:      cloneSessionConfig(sessionSnap.Config),
		metadata:    cloneStringMap(sessionSnap.Metadata),
		budget:      streaming.NewBudgetFromSnapshot(sessionSnap.Budget),
		conv:        NewConversationFromState(convState),
		userID:      userID,
		agentID:     agentID,
		activeAgent: activeAgent,
		rootAgent:   rootAgent,
		tools:       toolRegistry,
	}
}

func (r *Runtime) publishRunState(rs *runState) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.generation != rs.generation {
		return
	}

	if r.session != nil {
		r.session.Restore(SessionSnapshot{
			ID:       rs.sessionID,
			Config:   cloneSessionConfig(rs.config),
			Budget:   rs.budget.Snapshot(),
			Metadata: cloneStringMap(rs.metadata),
		})
	}
	if r.conv != nil {
		r.conv.Restore(rs.conv.Snapshot())
	}
	r.agent = rs.activeAgent
	r.agentID = rs.agentID
	r.tools = rs.tools
	r.dispatcher = tools.NewDispatcher(r.tools)
}
