# AGENTS.md - AI Agent Instructions

## CRITICAL: Before You Start
- **ASK QUESTIONS** - Never make assumptions. Clarify requirements and intent before coding.
- **NO COMMITS TO MAIN** - Always create a feature branch. Never commit directly to the main branch.
- **Keep commit messages short and concise** - No need for a long explanation.
- **Keep PR descriptions short and concise** - No need for a long explanation or lots of sections about testing. Just a brief summary of what was done.
- **Add tests** - Always add tests for new features.
- **Don't use "fixup" commits.** Instead, amend the relevant commit. Force-pushing to your feature branch is encouraged to keep the history clean.
- **Ask questions, dont make assumptions** - Always ask questions instead of making assumptions

## Quick Commands (Always use `uv`)
- **Install**: `uv sync --all-groups`
- **Run UI**: `uv run pyneg-ui`
- **Run Tests**: `uv run pytest` (or `uv run pytest tests/path/to/test.py::TestClass::test_method`)
- **Format**: `uv run ruff format .`
- **Lint**: `uv run ruff check .` (use `--fix` to auto-fix)

## Git Workflow
- **Branch Strategy**: Feature branch -> Pull Request.
- **Pre-commit**: Run `uv run ruff format .`, `uv run pytest`, and `uv run ruff check . --fix` before any commit.
- **Commits**: Every commit must be buildable and pass tests.

## Project Architecture
- **Core**: `src/pynegative/core.py` - Image processing, tone mapping, RAW loading.
- **Editor**: `src/pynegative/ui/editor.py`, `editingcontrols.py`, `imageprocessing.py` - Main editing workflow.
- **Gallery**: `src/pynegative/ui/gallery.py` - Grid browsing, rating, and filtering.
- **Carousel**: `src/pynegative/ui/carouselmanager.py` - Image navigation logic.
- **Export**: `src/pynegative/ui/export_tab.py`, `exportprocessor.py`, `exportsettingsmanager.py` - Batch export pipeline.
- **Undo/Redo**: `src/pynegative/ui/undomanager.py` - Command pattern for history.
- **Widgets**: `src/pynegative/ui/widgets/` - Reusable UI components (sliders, star ratings, etc.).
- **Tests**: `tests/` - pytest suite mirroring the `src` structure.

## Code Standards
- **Style**: ruff for formatting and linting (PEP 8).
- **Types**: Mandatory type hints for all function signatures.
- **Naming**: `snake_case` (funcs/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants), `_private` prefix.
- **Signals**: camelCase with `Signal` suffix (e.g., `ratingChanged = Signal(int)`).

## PySide6 / UI Patterns
- **Structure**: Inherit from standard QWidgets. Use signals/slots for inter-component communication.
- **Performance**: Use `QTimer` for throttling expensive UI updates (see `editor.py`).
- **Memory**: Ensure proper parent-child relationships for Qt object cleanup.
- **Selection Sync**: When subclassing QListWidget with custom selection tracking, use `itemSelectionChanged` (built-in signal) and sync to custom state. Calling `setCurrentRow()` alone does not trigger selection signals - use `item.setSelected(True)` to ensure proper signal emission.

## Documentation Reference
- `README.md`: User-facing features and installation.
- `CONTRIBUTING.md`: Developer environment setup and contribution guide.
- `TODO.md`: Feature roadmap and testing improvement areas. When a feature is complete remove it from this file and update the readme accordingly if the feature is user facing and worth mentioning.
- `AGENTS.md`: This file (AI agent guidelines).
- `plan.md`: Technical implementation plans (Internal dev use).

## Common Patterns
- **Image Data**: Work on copies. Use numpy for operations. Validate 0.0-1.0 ranges for normalized data.
- **File I/O**: Use `pathlib.Path`. Handle missing/corrupt files and metadata sidecars (`.xmp`) gracefully.
- **Logic/UI Separation**: Keep image processing logic separate from widget state management.
