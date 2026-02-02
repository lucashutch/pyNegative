# Contributing to pyNegative

Thank you for your interest in contributing to pyNegative! This document covers how to set up your development environment and the conventions used in the project.

## Developer Setup

If you want to contribute or run from source, we recommend using `uv` for dependency management.

### Get the code
```bash
git clone https://github.com/lucashutch/pyNegative.git
cd pyNegative
```

### Using `uv` (Preferred)
`uv` automatically creates a virtual environment and installs dependencies.
```bash
# Install in editable mode
uv sync --all-groups

# Run the UI
uv run pyneg-ui
```

### Using `pip` (Legacy)
If you prefer standard tools, you can use pip, though it's less recommended for modern environments.
```bash
# Install in editable mode
pip install -e .

# Run the UI
pyneg-ui
```

## Development Workflow

### Testing
We use `pytest` for all tests. Please ensure tests pass before submitting changes.
```bash
# Run all tests
uv run pytest
```

### Linting & Formatting
We use `ruff` to keep the codebase clean. 
```bash
# Check for linting issues
uv run ruff check .

# Automatically fix what's possible
uv run ruff check --fix .

# Format the code
uv run ruff format .
```

## AI Agents
If you are using AI coding agents (like Claude Dev, OpenCode, etc.), please refer to [AGENTS.md](AGENTS.md) for project-specific instructions and guidelines for agent behavior.
