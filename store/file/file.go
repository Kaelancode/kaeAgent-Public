package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Kaelancode/kaeAgent-Public/llm"
	"github.com/Kaelancode/kaeAgent-Public/store"
)

type ConversationStore struct {
	mu  sync.RWMutex
	dir string
}

var _ store.ConversationStore = (*ConversationStore)(nil)

type Config struct {
	Dir string
}

func NewConversationStore(cfg Config) (*ConversationStore, error) {
	dir := filepath.Join(cfg.Dir, "conversations")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("creating conversations directory: %w", err)
	}
	return &ConversationStore{dir: dir}, nil
}

func (s *ConversationStore) Save(ctx context.Context, convID string, messages []llm.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	data, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("marshal messages: %w", err)
	}

	path, err := s.filePath(convID)
	if err != nil {
		return err
	}
	return atomicWriteContext(ctx, path, data)
}

func (s *ConversationStore) Load(ctx context.Context, convID string) ([]llm.Message, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	path, err := s.filePath(convID)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read file: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var messages []llm.Message
	if err := json.Unmarshal(data, &messages); err != nil {
		return nil, fmt.Errorf("unmarshal messages: %w", err)
	}
	return messages, nil
}

func (s *ConversationStore) Append(ctx context.Context, convID string, messages []llm.Message) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	path, err := s.filePath(convID)
	if err != nil {
		return err
	}
	var existing []llm.Message

	if err := ctx.Err(); err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("read file: %w", err)
	}
	if err == nil {
		if err := json.Unmarshal(data, &existing); err != nil {
			return fmt.Errorf("unmarshal existing messages: %w", err)
		}
	}

	existing = append(existing, messages...)

	merged, err := json.Marshal(existing)
	if err != nil {
		return fmt.Errorf("marshal merged messages: %w", err)
	}
	return atomicWriteContext(ctx, path, merged)
}

func (s *ConversationStore) Delete(ctx context.Context, convID string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	path, err := s.filePath(convID)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("delete file: %w", err)
	}
	return nil
}

func (s *ConversationStore) filePath(convID string) (string, error) {
	if convID == "" {
		return "", fmt.Errorf("invalid conversation id: empty")
	}
	if filepath.IsAbs(convID) {
		return "", fmt.Errorf("invalid conversation id %q: absolute paths are not allowed", convID)
	}
	clean := filepath.Clean(convID)
	if clean == "." || clean == ".." {
		return "", fmt.Errorf("invalid conversation id %q", convID)
	}
	if clean != convID {
		return "", fmt.Errorf("invalid conversation id %q: path traversal is not allowed", convID)
	}
	if strings.ContainsRune(convID, filepath.Separator) {
		return "", fmt.Errorf("invalid conversation id %q: path separators are not allowed", convID)
	}

	path := filepath.Join(s.dir, convID+".json")
	base := s.dir + string(filepath.Separator)
	if path != s.dir && !strings.HasPrefix(path, base) {
		return "", fmt.Errorf("invalid conversation id %q: resolved outside store directory", convID)
	}
	return path, nil
}

func atomicWrite(path string, data []byte) error {
	return atomicWriteContext(context.Background(), path, data)
}

func atomicWriteContext(ctx context.Context, path string, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}
	if err := ctx.Err(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename temp file: %w", err)
	}
	return nil
}
