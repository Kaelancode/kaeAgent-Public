package llm

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func TestOpenAIReadSSE_MissingDoneEmitsErrorWithoutDone(t *testing.T) {
	provider := &OpenAIProvider{}
	events := collectProviderEvents(func(ch chan<- Event) {
		provider.readSSE(context.Background(), io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n")), ch)
	})

	assertTerminalErrorWithoutDone(t, events, "openai: stream ended without [DONE]")
}

func TestQwenReadSSE_MissingDoneEmitsErrorWithoutDone(t *testing.T) {
	provider := &QwenProvider{}
	events := collectProviderEvents(func(ch chan<- Event) {
		provider.readSSE(context.Background(), io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n")), ch)
	})

	assertTerminalErrorWithoutDone(t, events, "qwen: stream ended without [DONE]")
}

func TestClaudeReadSSE_MissingMessageStopEmitsErrorWithoutDone(t *testing.T) {
	provider := &ClaudeProvider{}
	payload := "event: content_block_delta\n" +
		"data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"},\"index\":0}\n"
	events := collectProviderEvents(func(ch chan<- Event) {
		provider.readSSE(context.Background(), io.NopCloser(strings.NewReader(payload)), ch)
	})

	assertTerminalErrorWithoutDone(t, events, "claude: stream ended without message_stop")
}

func TestGeminiReadSSE_ScanErrorDoesNotEmitDone(t *testing.T) {
	provider := &GeminiProvider{}
	events := collectProviderEvents(func(ch chan<- Event) {
		provider.readSSE(context.Background(), &errorReadCloser{
			reader: strings.NewReader("data: {\"candidates\":[]}\n"),
			err:    errors.New("boom"),
		}, ch)
	})

	assertTerminalErrorWithoutDone(t, events, "gemini: sse read: boom")
}

func collectProviderEvents(run func(chan<- Event)) []Event {
	ch := make(chan Event, 16)
	go run(ch)

	var events []Event
	for event := range ch {
		events = append(events, event)
	}
	return events
}

func assertTerminalErrorWithoutDone(t *testing.T, events []Event, wantErr string) {
	t.Helper()

	var gotDone bool
	var gotErr error
	for _, event := range events {
		if event.Kind == EventDone {
			gotDone = true
		}
		if event.Kind == EventError && event.Err != nil {
			gotErr = event.Err
		}
	}

	if gotDone {
		t.Fatal("did not expect EventDone")
	}
	if gotErr == nil {
		t.Fatal("expected EventError, got nil")
	}
	if gotErr.Error() != wantErr {
		t.Fatalf("expected error %q, got %q", wantErr, gotErr.Error())
	}
}

func TestOpenAIReadSSECancellationDoesNotBlockOnUnreadChannel(t *testing.T) {
	provider := &OpenAIProvider{}
	ctx, cancel := context.WithCancel(context.Background())
	ch := make(chan Event)
	done := make(chan struct{})

	go func() {
		defer close(done)
		provider.readSSE(ctx, io.NopCloser(strings.NewReader("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n")), ch)
	}()

	cancel()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("expected readSSE to exit after cancellation")
	}
}

type errorReadCloser struct {
	reader io.Reader
	err    error
}

func (r *errorReadCloser) Read(p []byte) (int, error) {
	if r.reader == nil {
		return 0, r.err
	}

	n, readErr := r.reader.Read(p)
	if errors.Is(readErr, io.EOF) {
		r.reader = nil
		if n > 0 {
			return n, nil
		}
		return 0, r.err
	}
	return n, readErr
}

func (r *errorReadCloser) Close() error { return nil }
