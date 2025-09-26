# Tests

This directory contains unit tests, integration tests, and test utilities for the SnitchSystem.

## Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for system components
├── fixtures/       # Test data and mock objects
└── utils/          # Test utilities and helpers
```

## Test Categories

### Unit Tests
- Edge device camera handling
- YOLO detection pipeline
- Event correlation logic
- Server API endpoints
- Database models and operations

### Integration Tests
- Edge-to-server communication
- Database connectivity
- Camera stream processing
- End-to-end detection workflow

### Performance Tests
- Detection latency benchmarks
- Memory usage monitoring
- Concurrent request handling
- Database query performance

## Running Tests

### All Tests
```bash
pytest
```

### Specific Test Categories
```bash
pytest tests/unit/
pytest tests/integration/
```

### With Coverage
```bash
pytest --cov=edge --cov=server
```

## Test Data

- Mock camera streams for testing
- Sample detection events
- Database fixtures
- Configuration templates

## Continuous Integration

- GitHub Actions workflow for automated testing
- Docker-based test environments
- Performance regression detection
- Code coverage reporting
