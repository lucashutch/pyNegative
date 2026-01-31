# pyNegative - Agent Guidelines

This document contains guidelines for agentic coding agents working on the pyNegative repository.

## Project Overview

pyNegative is a cross-platform desktop application for photographers to non-destructively edit, rate, and export RAW files. It's built with Python using PySide6 for the GUI and various image processing libraries.

## Development Setup

### Environment
- Use `uv` for dependency management and virtual environment creation
- Install dependencies: `uv sync --all-groups`
- Source the virtual environment: `source .venv/bin/activate` (helps LSP)

### Running the Application
```bash
uv run pyneg-ui
```

## Build, Lint, and Test Commands

### Core Commands
- **Install dependencies**: `uv sync --all-groups`
- **Run UI**: `uv run pyneg-ui`
- **Run tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/test_file.py::TestClass::test_method`
- **Lint code**: `uv run ruff check .`
- **Format code**: `uv run ruff format .`
- **Fix linting issues**: `uv run ruff check --fix .`

### Pre-commit Workflow
Before pushing any commits to GitHub:
1. Format code: `uv run ruff format .`
2. Run tests: `uv run pytest`
3. Fix any linting issues: `uv run ruff check --fix .`
All commits should be buildable and testable to ensure git bisect works well.

## Git Workflow

- **Branch Strategy**: PR -> merge style workflow
- **No direct commits to main** - all changes go through pull requests
- **CI Requirements**: All tests must pass before merging
- **Commit Philosophy**: Preference for amending commits/rewriting branch history rather than fixup commits
- **Buildable Commits**: Every commit should be buildable and testable

## Code Style Guidelines

### Python Style
- Use **ruff** for linting and formatting
- Follow PEP 8 with ruff's default configuration
- Shebang: Use `#!/usr/bin/env python3` for executable scripts
- Import organization: ruff will handle this automatically

### Type Hints
- Use type hints consistently across the codebase
- Import types from `typing` module as needed
- Function signatures should include parameter and return type hints

### Naming Conventions
- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: Prefix with underscore `_`
- **Signals (PySide6)**: Use camelCase with `Signal` suffix: `ratingChanged = Signal(int)`

### Error Handling
- Use specific exception types where possible
- Log errors appropriately for debugging
- Handle Qt-specific exceptions (e.g., image loading failures)
- Validate inputs in core image processing functions

### Code Organization
- **Core logic**: `src/pynegative/core.py` - image processing, tone mapping
- **UI components**: `src/pynegative/ui/` - all GUI-related code
- **Custom widgets**: `src/pynegative/ui/widgets/` - reusable UI components
- **Tests**: `tests/` directory with `test_*.py` naming

## PySide6/UI Guidelines

### Widget Structure
- Inherit from appropriate Qt widgets (`QtWidgets.QWidget`, `QtWidgets.QMainWindow`, etc.)
- Use signals and slots for communication between components
- Implement proper memory management for Qt objects

### Custom Widgets
- Follow existing patterns in `src/pynegative/ui/widgets/`
- Use `setMouseTracking(True)` for hover interactions
- Implement proper paint events with `QPainter`
- Handle events like `mousePressEvent`, `mouseMoveEvent` for interactivity

### Performance
- Use `QTimer` for throttling UI updates (see `editor.py`)
- Implement efficient image loading and caching
- Consider multi-threading for expensive operations (future enhancement)

## Testing Guidelines

### Test Structure
- Use `pytest` for all testing
- Use `pytest-qt` for Qt-specific testing
- Test files should follow `test_*.py` naming convention
- Place tests in the `tests/` directory

### Key Testing Areas
1. **Core Logic**: Test image processing functions with known inputs/outputs
2. **UI Components**: Test widget behavior with pytest-qt
3. **Integration**: Test end-to-end workflows where feasible

### Running Tests
- All tests: `uv run pytest`
- Specific test: `uv run pytest tests/test_file.py::TestClass::test_method`
- Verbose output: `uv run pytest -v`

## Dependencies

### Core Dependencies
- `numpy`: Array operations for image processing
- `Pillow`: Image manipulation
- `pillow-heif`: HEIF format support
- `rawpy`: RAW file processing
- `PySide6`: GUI framework

### Development Dependencies
- `pytest`: Testing framework
- `pytest-qt`: Qt testing utilities
- `ruff`: Linting and formatting

## File Structure Patterns

### Import Organization
Let ruff handle this automatically, but generally:
1. Standard library imports
2. Third-party imports (PySide6, numpy, etc.)
3. Local imports (pynegative.*)

### Constants
- Define module-level constants in `UPPER_SNAKE_CASE`
- Supported file extensions: `SUPPORTED_EXTS` constant in `core.py`
- Configuration values should be centralized where appropriate

## Common Patterns

### Image Processing
- Always work with copies of image data to avoid in-place modifications
- Use numpy arrays for efficient numerical operations
- Validate image data ranges (typically 0.0-1.0 for normalized data)

### File I/O
- Use `pathlib.Path` for file path operations
- Handle various image formats gracefully
- Implement proper error handling for missing/corrupt files

### State Management
- Use Qt's signal/slot mechanism for state changes
- Keep UI state separate from data processing logic
- Implement proper cleanup for resources

## Performance Considerations

### Image Processing
- Cache expensive operations where possible
- Use vectorized numpy operations instead of loops
- Consider memory usage for large RAW files

### UI Performance
- Throttle expensive UI updates with `QTimer`
- Use lazy loading for large image collections
- Implement proper resource cleanup for Qt objects

## Development Tips

- Always test with actual RAW files when working on image processing
- Use the existing test images in the test suite for consistent testing
- Follow the established widget patterns when creating new UI components
- Check the README.md for feature context and roadmap items
- Refer to the testing improvement areas in README.md for guidance on test coverage