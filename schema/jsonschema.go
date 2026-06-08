// schema/jsonschema.go
package schema

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Schema represents a JSON Schema used for tool parameter validation.
type Schema struct {
	Type        string             `json:"type"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Description string             `json:"description,omitempty"`
	Enum        []any              `json:"enum,omitempty"`
	Default     any                `json:"default,omitempty"`
	MinLength   *int               `json:"minLength,omitempty"`
	MaxLength   *int               `json:"maxLength,omitempty"`
	Minimum     *float64           `json:"minimum,omitempty"`
	Maximum     *float64           `json:"maximum,omitempty"`
}

// Validate checks that the given value conforms to this schema.
// It returns a list of validation errors; an empty slice means the value is valid.
func (s *Schema) Validate(value any) []string {
	return validateValue(s, value, "")
}

// ValidateJSON parses raw JSON then validates it against the schema.
func (s *Schema) ValidateJSON(data []byte) (map[string]any, []string) {
	var parsed any
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, []string{fmt.Sprintf("invalid JSON: %v", err)}
	}

	errs := s.Validate(parsed)
	if len(errs) > 0 {
		return nil, errs
	}

	if m, ok := parsed.(map[string]any); ok {
		return m, nil
	}
	return map[string]any{"value": parsed}, nil
}

// CoerceAndValidate attempts type coercion on the input map then validates.
func (s *Schema) CoerceAndValidate(input map[string]any) (map[string]any, []string) {
	coerced := s.coerceObject(input)
	errs := s.Validate(coerced)
	return coerced, errs
}

// ToMap converts the schema to a map[string]any suitable for JSON serialization.
func (s *Schema) ToMap() map[string]any {
	data, err := json.Marshal(s)
	if err != nil {
		return map[string]any{}
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return map[string]any{}
	}
	if m == nil {
		return map[string]any{}
	}
	return m
}

func validateValue(s *Schema, value any, path string) []string {
	if value == nil {
		return []string{fmt.Sprintf("%s: expected %s, got null", pathOrRoot(path), schemaTypeOrValue(s))}
	}

	var errs []string

	switch s.Type {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			return []string{fmt.Sprintf("%s: expected object, got %T", pathOrRoot(path), value)}
		}
		for _, req := range s.Required {
			if _, exists := obj[req]; !exists {
				errs = append(errs, fmt.Sprintf("%s: missing required field %q", pathOrRoot(path), req))
			}
		}
		for key, propSchema := range s.Properties {
			if val, exists := obj[key]; exists {
				childPath := path + "." + key
				if path == "" {
					childPath = key
				}
				errs = append(errs, validateValue(propSchema, val, childPath)...)
			}
		}

	case "array":
		arr, ok := value.([]any)
		if !ok {
			return []string{fmt.Sprintf("%s: expected array, got %T", pathOrRoot(path), value)}
		}
		if s.Items != nil {
			for i, item := range arr {
				childPath := fmt.Sprintf("%s[%d]", pathOrRoot(path), i)
				errs = append(errs, validateValue(s.Items, item, childPath)...)
			}
		}

	case "string":
		str, ok := value.(string)
		if !ok {
			return []string{fmt.Sprintf("%s: expected string, got %T", pathOrRoot(path), value)}
		}
		if s.MinLength != nil && len(str) < *s.MinLength {
			errs = append(errs, fmt.Sprintf("%s: string length %d < minLength %d", pathOrRoot(path), len(str), *s.MinLength))
		}
		if s.MaxLength != nil && len(str) > *s.MaxLength {
			errs = append(errs, fmt.Sprintf("%s: string length %d > maxLength %d", pathOrRoot(path), len(str), *s.MaxLength))
		}

	case "number", "integer":
		num, ok := toFloat64(value)
		if !ok {
			return []string{fmt.Sprintf("%s: expected %s, got %T", pathOrRoot(path), s.Type, value)}
		}
		if s.Type == "integer" {
			if !isIntegerNumber(num) {
				errs = append(errs, fmt.Sprintf("%s: expected integer, got %v", pathOrRoot(path), num))
			}
		}
		if s.Minimum != nil && num < *s.Minimum {
			errs = append(errs, fmt.Sprintf("%s: value %v < minimum %v", pathOrRoot(path), num, *s.Minimum))
		}
		if s.Maximum != nil && num > *s.Maximum {
			errs = append(errs, fmt.Sprintf("%s: value %v > maximum %v", pathOrRoot(path), num, *s.Maximum))
		}

	case "boolean":
		if _, ok := value.(bool); !ok {
			return []string{fmt.Sprintf("%s: expected boolean, got %T", pathOrRoot(path), value)}
		}
	default:
		return []string{fmt.Sprintf("%s: unknown schema type %q", pathOrRoot(path), s.Type)}
	}

	if len(s.Enum) > 0 && !enumContains(s, value) {
		errs = append(errs, fmt.Sprintf("%s: value %v not in enum %v", pathOrRoot(path), value, s.Enum))
	}

	return errs
}

func schemaTypeOrValue(s *Schema) string {
	if s == nil || s.Type == "" {
		return "non-null value"
	}
	return s.Type
}

func (s *Schema) coerceObject(input map[string]any) map[string]any {
	if s.Type != "object" || s.Properties == nil {
		return input
	}

	result := make(map[string]any, len(input))
	for k, v := range input {
		propSchema, exists := s.Properties[k]
		if !exists {
			result[k] = v
			continue
		}
		result[k] = coerceValue(propSchema, v)
	}

	for name, propSchema := range s.Properties {
		if _, exists := result[name]; !exists && propSchema.Default != nil {
			result[name] = propSchema.Default
		}
	}

	return result
}

func coerceValue(s *Schema, value any) any {
	if value == nil {
		return value
	}

	str, isStr := value.(string)

	switch s.Type {
	case "integer":
		if isStr {
			if i, err := strconv.ParseInt(strings.TrimSpace(str), 10, 64); err == nil {
				return float64(i) // JSON numbers are float64
			}
		}
		if f, ok := value.(float64); ok && isIntegerNumber(f) {
			return f
		}
	case "number":
		if isStr {
			if f, err := strconv.ParseFloat(strings.TrimSpace(str), 64); err == nil {
				return f
			}
		}
	case "boolean":
		if isStr {
			switch strings.ToLower(strings.TrimSpace(str)) {
			case "true", "1", "yes":
				return true
			case "false", "0", "no":
				return false
			}
		}
	case "array":
		if arr, ok := value.([]any); ok && s.Items != nil {
			out := make([]any, len(arr))
			for i, item := range arr {
				out[i] = coerceValue(s.Items, item)
			}
			return out
		}
	case "object":
		if m, ok := value.(map[string]any); ok {
			return s.coerceObject(m)
		}
	}
	return value
}

func isIntegerNumber(num float64) bool {
	return !math.IsNaN(num) && !math.IsInf(num, 0) && math.Trunc(num) == num
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int8:
		return float64(n), true
	case int16:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint:
		return float64(n), true
	case uint8:
		return float64(n), true
	case uint16:
		return float64(n), true
	case uint32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func enumContains(s *Schema, value any) bool {
	switch s.Type {
	case "string":
		str, ok := value.(string)
		if !ok {
			return false
		}
		for _, e := range s.Enum {
			if enumStr, ok := e.(string); ok && enumStr == str {
				return true
			}
		}
	case "number", "integer":
		num, ok := toFloat64(value)
		if !ok {
			return false
		}
		for _, e := range s.Enum {
			if enumNum, ok := toFloat64(e); ok && enumNum == num {
				return true
			}
		}
	case "boolean":
		b, ok := value.(bool)
		if !ok {
			return false
		}
		for _, e := range s.Enum {
			if enumBool, ok := e.(bool); ok && enumBool == b {
				return true
			}
		}
	default:
		for _, e := range s.Enum {
			if e == value {
				return true
			}
		}
	}
	return false
}

func pathOrRoot(path string) string {
	if path == "" {
		return "$"
	}
	return path
}
