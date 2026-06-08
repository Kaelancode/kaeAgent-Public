package sse

import (
	"bufio"
	"errors"
	"fmt"
	"io"
)

const MaxLineBytes = 8 * 1024 * 1024

var ErrLineTooLong = errors.New("sse line exceeds max size")

func ReadLine(r *bufio.Reader) (string, error) {
	var line []byte
	for {
		fragment, prefix, err := r.ReadLine()
		if err != nil {
			if errors.Is(err, io.EOF) && len(line) > 0 {
				return string(line), nil
			}
			return "", err
		}
		if len(line)+len(fragment) > MaxLineBytes {
			return "", fmt.Errorf("%w: %d bytes", ErrLineTooLong, MaxLineBytes)
		}
		line = append(line, fragment...)
		if !prefix {
			return string(line), nil
		}
	}
}
