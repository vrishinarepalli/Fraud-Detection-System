# Security Policy

## Sensitive Files — Never Commit

The following must never be committed to this repository under any circumstances:

| File / Pattern | Reason |
|---|---|
| `.env`, `.env.*` | Contains environment variables and credentials |
| `kaggle.json` | Kaggle API token — treat as a password |
| `data/*.csv` | Raw dataset files — too large and may contain PII |
| `artifacts/*.pkl` | Trained models may encode training data |
| `*.pem`, `*.key`, `*.secret` | Private keys and secrets |
| Any file with hardcoded API keys or passwords | Self-explanatory |

These patterns are enforced via `.gitignore`, but `.gitignore` is not a security boundary — it only prevents accidental staging. Never hardcode credentials in source files.

## If a Secret Is Accidentally Committed

1. **Rotate the credential immediately.** Treat the exposed secret as compromised regardless of how quickly it is removed.
2. Remove the secret from git history using `git filter-repo` (not `git rm` — the file will still exist in prior commits).
3. Force-push the cleaned history and notify any collaborators to re-clone.

## Environment Variables

Use a `.env` file locally (never committed) and load it with `python-dotenv` or your deployment platform's secret manager. An `.env.example` file with placeholder values is safe to commit and documents required variables.

## Reporting Vulnerabilities

Open a GitHub issue marked **[SECURITY]** or contact the repository owner directly.
