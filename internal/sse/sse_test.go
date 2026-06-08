package sse

import (
	"bufio"
	"errors"
	"strings"
	"testing"
)

func TestReadLineReturnsPartialLineAtEOF(t *testing.T) {
	line, err := ReadLine(bufio.NewReader(strings.NewReader("data: partial")))
	if err != nil {
		t.Fatalf("ReadLine: %v", err)
	}
	if line != "data: partial" {
		t.Fatalf("expected partial line, got %q", line)
	}
}

func TestReadLineRejectsOversizedLine(t *testing.T) {
	line := strings.Repeat("x", MaxLineBytes+1)
	_, err := ReadLine(bufio.NewReader(strings.NewReader(line)))
	if !errors.Is(err, ErrLineTooLong) {
		t.Fatalf("expected ErrLineTooLong, got %v", err)
	}
}
