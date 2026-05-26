package schema

import (
	"testing"
)

func TestSchema_ValidateObject(t *testing.T) {
	s := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"name": {Type: "string"},
			"age":  {Type: "integer"},
		},
		Required: []string{"name"},
	}

	errs := s.Validate(map[string]any{"name": "Alice", "age": float64(30)})
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}

	errs = s.Validate(map[string]any{"age": float64(30)})
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing required field, got %d: %v", len(errs), errs)
	}
}

func TestSchema_ValidateString(t *testing.T) {
	min := 2
	max := 10
	s := &Schema{Type: "string", MinLength: &min, MaxLength: &max}

	errs := s.Validate("ok")
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}

	errs = s.Validate("x")
	if len(errs) != 1 {
		t.Errorf("expected 1 error for minLength, got %d", len(errs))
	}

	errs = s.Validate("this is way too long")
	if len(errs) != 1 {
		t.Errorf("expected 1 error for maxLength, got %d", len(errs))
	}
}

func TestSchema_ValidateNumber(t *testing.T) {
	min := 0.0
	max := 100.0
	s := &Schema{Type: "number", Minimum: &min, Maximum: &max}

	errs := s.Validate(float64(50))
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}

	errs = s.Validate(float64(-1))
	if len(errs) != 1 {
		t.Errorf("expected 1 error for minimum, got %d", len(errs))
	}
}

func TestSchema_ValidateEnum(t *testing.T) {
	s := &Schema{Type: "string", Enum: []any{"GET", "POST", "PUT"}}

	errs := s.Validate("GET")
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}

	errs = s.Validate("DELETE")
	if len(errs) != 1 {
		t.Errorf("expected 1 enum error, got %d", len(errs))
	}
}

func TestSchema_ValidateEnum_NumberIntegerBoolean(t *testing.T) {
	tests := []struct {
		name   string
		schema *Schema
		value  any
		valid  bool
	}{
		{
			name:   "number valid",
			schema: &Schema{Type: "number", Enum: []any{float64(1.5), float64(2.5)}},
			value:  float64(1.5),
			valid:  true,
		},
		{
			name:   "number invalid",
			schema: &Schema{Type: "number", Enum: []any{float64(1.5), float64(2.5)}},
			value:  float64(3.5),
			valid:  false,
		},
		{
			name:   "integer valid",
			schema: &Schema{Type: "integer", Enum: []any{float64(1), float64(2)}},
			value:  float64(2),
			valid:  true,
		},
		{
			name:   "integer invalid",
			schema: &Schema{Type: "integer", Enum: []any{float64(1), float64(2)}},
			value:  float64(3),
			valid:  false,
		},
		{
			name:   "boolean valid",
			schema: &Schema{Type: "boolean", Enum: []any{true}},
			value:  true,
			valid:  true,
		},
		{
			name:   "boolean invalid",
			schema: &Schema{Type: "boolean", Enum: []any{true}},
			value:  false,
			valid:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.schema.Validate(tt.value)
			if tt.valid && len(errs) != 0 {
				t.Fatalf("expected no errors, got %v", errs)
			}
			if !tt.valid && len(errs) == 0 {
				t.Fatalf("expected enum validation failure")
			}
		})
	}
}

func TestSchema_CoerceAndValidate(t *testing.T) {
	s := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"count":  {Type: "integer"},
			"active": {Type: "boolean"},
		},
	}

	input := map[string]any{"count": "42", "active": "true"}
	coerced, errs := s.CoerceAndValidate(input)
	if len(errs) != 0 {
		t.Errorf("expected no errors after coercion, got %v", errs)
	}
	if coerced["count"] != float64(42) {
		t.Errorf("expected count=42, got %v", coerced["count"])
	}
	if coerced["active"] != true {
		t.Errorf("expected active=true, got %v", coerced["active"])
	}
}

func TestSchema_CoerceAndValidateArrayItems(t *testing.T) {
	s := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"values": {
				Type:  "array",
				Items: &Schema{Type: "integer"},
			},
			"flags": {
				Type:  "array",
				Items: &Schema{Type: "boolean"},
			},
			"items": {
				Type: "array",
				Items: &Schema{
					Type: "object",
					Properties: map[string]*Schema{
						"count": {Type: "integer"},
					},
				},
			},
		},
	}

	input := map[string]any{
		"values": []any{"1", "2"},
		"flags":  []any{"true", "0"},
		"items": []any{
			map[string]any{"count": "3"},
		},
	}

	coerced, errs := s.CoerceAndValidate(input)
	if len(errs) > 0 {
		t.Fatalf("expected no errors after array item coercion, got %v", errs)
	}

	values, ok := coerced["values"].([]any)
	if !ok {
		t.Fatalf("expected values array, got %T", coerced["values"])
	}
	if values[0] != float64(1) || values[1] != float64(2) {
		t.Fatalf("expected coerced integer array [1 2], got %#v", values)
	}

	flags, ok := coerced["flags"].([]any)
	if !ok {
		t.Fatalf("expected flags array, got %T", coerced["flags"])
	}
	if flags[0] != true || flags[1] != false {
		t.Fatalf("expected coerced boolean array [true false], got %#v", flags)
	}

	items, ok := coerced["items"].([]any)
	if !ok {
		t.Fatalf("expected items array, got %T", coerced["items"])
	}
	first, ok := items[0].(map[string]any)
	if !ok {
		t.Fatalf("expected first item object, got %T", items[0])
	}
	if first["count"] != float64(3) {
		t.Fatalf("expected nested object item count=3, got %#v", first["count"])
	}
}

func TestSchema_ValidateJSON(t *testing.T) {
	s := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"url": {Type: "string"},
		},
		Required: []string{"url"},
	}

	result, errs := s.ValidateJSON([]byte(`{"url":"https://example.com"}`))
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %v", errs)
	}
	if result["url"] != "https://example.com" {
		t.Errorf("unexpected url: %v", result["url"])
	}

	_, errs = s.ValidateJSON([]byte(`{}`))
	if len(errs) != 1 {
		t.Errorf("expected 1 error for missing required, got %d", len(errs))
	}

	_, errs = s.ValidateJSON([]byte(`{"url":null}`))
	if len(errs) != 1 {
		t.Errorf("expected 1 error for null required field, got %d", len(errs))
	}
}

func TestSchema_ValidateNullFailsTypedField(t *testing.T) {
	tests := []struct {
		name   string
		schema *Schema
	}{
		{name: "string", schema: &Schema{Type: "string"}},
		{name: "integer", schema: &Schema{Type: "integer"}},
		{name: "boolean", schema: &Schema{Type: "boolean"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errs := tt.schema.Validate(nil)
			if len(errs) != 1 {
				t.Fatalf("expected 1 error for null value, got %d", len(errs))
			}
		})
	}
}

func TestSchema_ToMap(t *testing.T) {
	s := &Schema{
		Type: "object",
		Properties: map[string]*Schema{
			"name": {Type: "string"},
		},
	}
	m := s.ToMap()
	if m["type"] != "object" {
		t.Errorf("expected type=object, got %v", m["type"])
	}
}
