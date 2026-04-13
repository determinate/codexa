# Security Policy

## Supported Versions

| Version | Security Updates   |
|---------|--------------------|
| 0.x     | :white_check_mark: |

> codexa is currently in early development. All `0.x` releases receive
> security patches. Once we reach `1.0`, this table will be updated.

---

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

If you found a security issue, report it privately:

### Option 1 — GitHub Private Disclosure (preferred)

Use GitHub's built-in private reporting:  
**Security → Report a vulnerability** (button on the repo page)

This keeps the details private until a fix is released.

### Option 2 — Email

Send details to: **evrey2696@gmail.com**  
Subject: `[codexa] Security vulnerability`

Encrypt with PGP if possible (key: _add your PGP key or keybase link here_).

---

## What to include

A good report helps us fix the issue faster:

- **Description** — what the vulnerability is
- **Impact** — what an attacker could do with it
- **Steps to reproduce** — minimal example or proof of concept
- **Affected versions** — which version(s) you tested
- **Suggested fix** — if you have one (optional but welcome)

---

## What happens after you report

| Timeline   | What we do                                              |
|------------|---------------------------------------------------------|
| 48 hours   | Acknowledge receipt, confirm we received your report   |
| 7 days     | Initial assessment — is it valid? what's the severity? |
| 30 days    | Patch released (critical issues get expedited)         |
| After fix  | Public disclosure with credit to you (if you want)     |

If a vulnerability is **declined** (not a real issue or out of scope),
we'll explain why within 7 days.

---

## Scope

Issues we consider in-scope:

- Code execution via crafted project files (e.g. malicious `codexa.toml`)
- Path traversal when scanning project directories
- Credential leakage (API keys written to logs or temp files)
- SQLite or PostgreSQL injection in the vector store
- Malicious ONNX model loading via `local-embed` feature

Out of scope:

- Vulnerabilities in your OS or Rust toolchain itself
- Issues requiring physical access to the machine
- Social engineering

---

## Credits

Security researchers who responsibly disclose issues will be credited
in the release notes and in [CONTRIBUTORS.md](CONTRIBUTORS.md) (with their permission).

We don't have a bug bounty program yet, but we appreciate every report.
