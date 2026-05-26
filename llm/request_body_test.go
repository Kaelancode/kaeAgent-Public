package llm

import "testing"

func TestOpenAIBuildRequestBodyOmitsUnsetTemperature(t *testing.T) {
	provider := &OpenAIProvider{}
	body := provider.buildRequestBody(&Request{
		Model:    "gpt-test",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}, false)

	if _, ok := body["temperature"]; ok {
		t.Fatal("expected temperature to be omitted")
	}
}

func TestOpenAIBuildRequestBodyIncludesZeroTemperature(t *testing.T) {
	provider := &OpenAIProvider{}
	body := provider.buildRequestBody(&Request{
		Model:       "gpt-test",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: float32Ptr(0),
	}, false)

	value, ok := body["temperature"]
	if !ok {
		t.Fatal("expected temperature to be included")
	}
	if value != float32(0) {
		t.Fatalf("expected temperature 0, got %#v", value)
	}
}

func TestClaudeBuildRequestBodyIncludesZeroTemperature(t *testing.T) {
	provider := &ClaudeProvider{}
	body := provider.buildRequestBody(&Request{
		Model:       "claude-test",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		MaxTokens:   64,
		Temperature: float32Ptr(0),
	}, false)

	value, ok := body["temperature"]
	if !ok {
		t.Fatal("expected temperature to be included")
	}
	if value != float32(0) {
		t.Fatalf("expected temperature 0, got %#v", value)
	}
}

func TestGeminiBuildRequestBodyIncludesZeroTemperature(t *testing.T) {
	provider := &GeminiProvider{}
	body := provider.buildRequestBody(&Request{
		Model:       "gemini-test",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: float32Ptr(0),
	})

	genConfig, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("expected generationConfig to be present")
	}
	if value, ok := genConfig["temperature"]; !ok || value != float32(0) {
		t.Fatalf("expected generationConfig temperature 0, got %#v", genConfig["temperature"])
	}
}

func TestGeminiBuildRequestBodyAppliesOptions(t *testing.T) {
	provider := &GeminiProvider{}
	body := provider.buildRequestBody(&Request{
		Model:    "gemini-test",
		Messages: []Message{{Role: "user", Content: "hi"}},
		Options: map[string]any{
			"top_p": 0.9,
			"topK":  32,
			"generationConfig": map[string]any{
				"candidateCount": 2,
			},
			"safetySettings": []map[string]any{
				{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
			},
		},
	})

	genConfig, ok := body["generationConfig"].(map[string]any)
	if !ok {
		t.Fatal("expected generationConfig to be present")
	}
	if value := genConfig["topP"]; value != 0.9 {
		t.Fatalf("expected topP 0.9, got %#v", value)
	}
	if value := genConfig["topK"]; value != 32 {
		t.Fatalf("expected topK 32, got %#v", value)
	}
	if value := genConfig["candidateCount"]; value != 2 {
		t.Fatalf("expected candidateCount 2, got %#v", value)
	}
	if _, ok := body["safetySettings"]; !ok {
		t.Fatal("expected top-level safetySettings to be present")
	}
}

func TestQwenBuildRequestBodyIncludesZeroTemperature(t *testing.T) {
	provider := &QwenProvider{}
	body := provider.buildRequestBody(&Request{
		Model:       "qwen-test",
		Messages:    []Message{{Role: "user", Content: "hi"}},
		Temperature: float32Ptr(0),
	}, false)

	value, ok := body["temperature"]
	if !ok {
		t.Fatal("expected temperature to be included")
	}
	if value != float32(0) {
		t.Fatalf("expected temperature 0, got %#v", value)
	}
}

func float32Ptr(v float32) *float32 {
	return &v
}
