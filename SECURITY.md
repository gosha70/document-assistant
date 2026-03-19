# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use [GitHub's private security advisory feature](https://github.com/gosha70/document-assistant/security/advisories/new) to report vulnerabilities.

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

You should receive an initial response within **72 hours**. A fix will be developed privately and released as a patch before public disclosure.

## Security Practices

This project follows these security practices:

- **No secrets in source** — API keys, tokens, and credentials are loaded from environment variables or `.env` files (which are gitignored)
- **Dependency auditing** — `pip-audit` runs in CI on every push
- **Static analysis** — `bandit` scans for common Python security issues in CI
- **Input validation** — All user-facing inputs are validated at system boundaries
- **Parameterized queries** — No string interpolation in database queries
