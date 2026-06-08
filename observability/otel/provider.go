package otel

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
)

type ProviderConfig struct {
	Endpoint     string
	ServiceName  string
	Insecure     bool
	Headers      map[string]string
	ExporterType string
	Username     string
	Password     string
	URLPath      string
}

func NewTracerProvider(ctx context.Context, cfg ProviderConfig) (*sdktrace.TracerProvider, func(), error) {
	if cfg.Endpoint != "" {
		parsedURL, err := url.Parse(cfg.Endpoint)
		if err == nil && parsedURL.Scheme != "" {
			cfg.Endpoint = parsedURL.Host
			if parsedURL.Path != "" && parsedURL.Path != "/" && cfg.URLPath == "" {
				cfg.URLPath = parsedURL.Path
			}
			if parsedURL.Scheme == "http" {
				cfg.Insecure = true
			}
		}
	}

	cfg.Headers = mergeAuthHeader(cfg.Headers, cfg.Username, cfg.Password)
	client, err := newOTLPClient(ctx, cfg)
	if err != nil {
		return nil, nil, err
	}

	exp, err := otlptrace.New(ctx, client)
	if err != nil {
		return nil, nil, fmt.Errorf("creating OTLP exporter: %w", err)
	}

	res, err := newResource(cfg.ServiceName)
	if err != nil {
		return nil, nil, err
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)

	otel.SetTracerProvider(tp)

	shutdown := func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = tp.Shutdown(shutdownCtx)
	}

	return tp, shutdown, nil
}

func newOTLPClient(ctx context.Context, cfg ProviderConfig) (otlptrace.Client, error) {
	switch strings.ToLower(cfg.ExporterType) {
	case "http", "https":
		return newHTTPClient(ctx, cfg)
	default:
		return newGRPCClient(ctx, cfg)
	}
}

func newGRPCClient(ctx context.Context, cfg ProviderConfig) (otlptrace.Client, error) {
	opts := []otlptracegrpc.Option{
		otlptracegrpc.WithEndpoint(cfg.Endpoint),
	}
	if cfg.Insecure {
		opts = append(opts, otlptracegrpc.WithInsecure())
	}
	if len(cfg.Headers) > 0 {
		opts = append(opts, otlptracegrpc.WithHeaders(cfg.Headers))
	}

	client := otlptracegrpc.NewClient(opts...)
	return client, nil
}

func newHTTPClient(ctx context.Context, cfg ProviderConfig) (otlptrace.Client, error) {
	opts := []otlptracehttp.Option{
		otlptracehttp.WithEndpoint(cfg.Endpoint),
	}
	if cfg.Insecure {
		opts = append(opts, otlptracehttp.WithInsecure())
	}
	if cfg.URLPath != "" {
		opts = append(opts, otlptracehttp.WithURLPath(cfg.URLPath))
	}
	if len(cfg.Headers) > 0 {
		opts = append(opts, otlptracehttp.WithHeaders(cfg.Headers))
	}

	client := otlptracehttp.NewClient(opts...)
	return client, nil
}

func newResource(serviceName string) (*resource.Resource, error) {
	r, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			"",
			semconv.ServiceNameKey.String(serviceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("creating OTLP resource: %w", err)
	}
	return r, nil
}

func mergeAuthHeader(headers map[string]string, username, password string) map[string]string {
	if username == "" && password == "" {
		return headers
	}
	if headers == nil {
		headers = make(map[string]string)
	}
	cred := base64.StdEncoding.EncodeToString([]byte(username + ":" + password))
	headers["Authorization"] = "Basic " + cred
	return headers
}
