# Contributing

## Setup

1. Create a virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

## Development Workflow

- Run tests before opening a PR:
  ```bash
  pytest
  ```
- Run linting:
  ```bash
  ruff check .
  ```

## Pull Requests

- Keep PRs focused and small.
- Include test coverage for behavioral changes.
- Update README if interfaces or command usage change.
