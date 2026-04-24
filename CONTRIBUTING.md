# Contributing

Thanks for your interest in contributing to Fusion Upscaler.

## Development Setup

1. Install Python 3.10.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Run desktop app:

```bash
py -3.10 desktop_app.py
```

## Contribution Guidelines

- Keep changes focused and small.
- Follow existing naming and code style.
- Update `README.md` when behavior or setup changes.
- Add/adjust docs for new options or modes.

## Pull Requests

- Use clear PR titles and descriptions.
- Explain what changed and why.
- Include steps to reproduce fixes when relevant.
- Attach before/after screenshots for UI changes.

## Reporting Issues

When reporting a bug, include:

- OS and Python version
- how to reproduce
- expected behavior vs actual behavior
- log files from `%TEMP%\fusion_upscaler_launcher.log` and `%TEMP%\fusion_upscaler_app.log` when startup fails
