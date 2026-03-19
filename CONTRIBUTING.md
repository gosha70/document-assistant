# Contributing to Document Assistant

Thank you for your interest in contributing! This guide will help you get started.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/document-assistant.git
   cd document-assistant
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Branch Naming

Use prefixed branch names:
- `feature/` — new functionality
- `fix/` — bug fixes
- `chore/` — maintenance, dependency updates
- `docs/` — documentation changes

### Code Style

- Python 3.11+
- Linting: `poetry run ruff check src/ tests/`
- All prompts go in `prompts/` as versioned YAML — never inline strings
- No hardcoded secrets, API keys, or model names — use config/env vars

### Running Tests

```bash
poetry run pytest -q --tb=short
```

All tests must pass before submitting a PR. CI also runs:
- `ruff` lint checks
- `bandit` security scan
- `pip-audit` dependency audit

### Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep the summary line under 72 characters
- One logical change per commit

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`
2. Run the full test suite locally
3. Push your branch and open a PR against `main`
4. Fill out the PR template
5. Wait for CI to pass and a maintainer review

## Reporting Bugs

Use the [bug report template](https://github.com/gosha70/document-assistant/issues/new?template=bug_report.yml) to file issues.

## Security Issues

Please report security vulnerabilities privately. See [SECURITY.md](SECURITY.md) for instructions.

## License

By contributing, you agree that your contributions will be licensed under the [CC BY-SA 4.0](LICENSE) license.
