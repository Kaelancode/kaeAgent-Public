package llm

func applyProviderOptions(body map[string]any, options map[string]any, reserved map[string]struct{}) {
	for k, v := range options {
		if _, ok := reserved[k]; ok {
			continue
		}
		body[k] = v
	}
}

func reservedOptions(keys ...string) map[string]struct{} {
	reserved := make(map[string]struct{}, len(keys))
	for _, key := range keys {
		reserved[key] = struct{}{}
	}
	return reserved
}
