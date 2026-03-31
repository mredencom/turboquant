# Contributing to TurboQuant

Thanks for your interest in contributing! This guide covers the essentials.

## Prerequisites

- Go 1.22 or later
- Git

## Setup

```bash
git clone https://github.com/mredencom/turboquant.git
cd turboquant
go mod download
```

## Development Workflow

### Run Tests

```bash
go test -v -race -count=1 ./...
```

### Run Benchmarks

```bash
go test -bench=. -benchmem ./...
```

### Run Linter

```bash
# Install golangci-lint: https://golangci-lint.run/welcome/install/
golangci-lint run
```

### Run Vet

```bash
go vet ./...
```

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Write clear, concise commit messages.
3. Add or update tests for any changed functionality.
4. Ensure all tests pass and the linter reports no issues.
5. Open a pull request against `main` with a description of your changes.

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`).
- Keep exported functions documented with GoDoc comments.
- Property-based tests go alongside unit tests in `*_test.go` files.

## Reporting Issues

Open a GitHub issue with a clear description, steps to reproduce, and expected vs actual behavior.
