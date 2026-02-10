# Contributing to pyNegative

Thank you for your interest in contributing to pyNegative! This guide covers everything you need to get up and running.

---

## Developer Setup

### Prerequisites
- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip`

### Get the Code
```bash
git clone https://github.com/lucashutch/pyNegative.git
cd pyNegative
```

### Install Dependencies

**Using `uv` (preferred):**
```bash
uv sync --all-groups
```

**Using `pip`:**
```bash
pip install -e .
```

### Run the Application
```bash
# With uv
uv run pynegative

# With pip (after install)
pynegative

# Debug mode (console output)
uv run pynegative-debug
```

---

## Project Architecture

```
src/pynegative/
├── core.py                 # Image processing, tone mapping, RAW decoding, sidecar I/O
├── __init__.py             # Package init, OpenCV cache setup
├── styles.qss              # Global QSS stylesheet
├── gpu/                    # (Reserved for future GPU acceleration)
├── logic/                  # (Reserved for future business logic separation)
├── utils/
│   ├── numba_kernels.py    # Numba kernel re-exports
│   ├── numba_color.py      # Tone mapping kernel
│   ├── numba_detail.py     # Sharpening kernel
│   ├── numba_denoise.py    # NL-Means & Bilateral denoising
│   └── numba_dehaze.py     # Dark channel / dehaze kernels
└── ui/
    ├── __init__.py          # App entry point, main() function
    ├── main_window.py       # Main window shell, tabs, toolbar
    ├── editor.py            # Image editing view (preview + controls + carousel)
    ├── editingcontrols.py   # Sidebar sliders & sections (tone, detail, geometry, etc.)
    ├── imageprocessing.py   # Async image processing pipeline
    ├── gallery.py           # Gallery grid view, sorting, filtering
    ├── loaders.py           # Async thumbnail & RAW loaders
    ├── carouselmanager.py   # Image strip navigation
    ├── export_tab.py        # Export view
    ├── exportprocessor.py   # Batch export job runner
    ├── exportsettingsmanager.py   # Export settings persistence
    ├── exportgallerymanager.py    # Export gallery selection
    ├── renamepreviewdialog.py     # Rename preview dialog
    ├── renamesettingsmanager.py   # Rename pattern settings
    ├── settingsmanager.py   # General settings persistence
    ├── undomanager.py       # Command-pattern undo/redo
    └── widgets/             # Reusable UI components
        ├── collapsiblesection.py
        ├── combobox.py
        ├── comparisonoverlay.py
        ├── crop_item.py
        ├── galleryitemdelegate.py
        ├── gallerylistwidget.py
        ├── histogram.py
        ├── horizontallist.py
        ├── resetableslider.py
        ├── starrating.py
        ├── toast.py
        ├── zoomablegraphicsview.py
        └── zoomcontrols.py
```

### Key Concepts

- **Non-destructive editing**: All edits are stored as JSON sidecar files (`.pyNegative/<filename>.json`). Original images are never modified.
- **Numba acceleration**: All pixel-level processing (tone mapping, sharpening, denoising, dehazing) uses Numba JIT-compiled kernels for near-native performance.
- **Async loading**: Thumbnail and RAW image loading happens on a `QThreadPool` to keep the UI responsive.
- **Thumbnail caching**: Resized thumbnails are persisted to disk (`.pyNegative/thumbnails/`) and loaded on subsequent visits for fast gallery startup.
- **Signal/slot architecture**: Components communicate via Qt signals. Camel case with descriptive names (e.g., `ratingChanged`, `settingChanged`).

---

## Development Workflow

### Testing
We use `pytest` for all tests. The test suite is in `tests/`, mirroring the `src/` structure.

```bash
# Run all tests
uv run pytest

# Run a specific test file or test
uv run pytest tests/core/test_thumbnail_cache.py
uv run pytest tests/core/test_tone_mapping.py::TestToneMapping::test_exposure
```

### Linting & Formatting
We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting (PEP 8).

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Pre-Commit Checklist
Before pushing, always run:
```bash
uv run ruff format .
uv run ruff check . --fix
uv run pytest
```

### Git Workflow
- **Never commit directly to `main`**. Always create a feature branch and open a Pull Request.
- Keep commit messages short and descriptive.
- Every commit should be buildable and pass tests.

---

## Code Standards

| Convention | Style |
|---|---|
| Functions / Variables | `snake_case` |
| Classes | `PascalCase` |
| Constants | `UPPER_SNAKE_CASE` |
| Private members | `_leading_underscore` |
| Qt Signals | `camelCase` (e.g., `ratingChanged = Signal(int)`) |
| Type hints | Required on all function signatures |

### Image Data Conventions
- Image arrays are NumPy `float32`, normalized to `0.0–1.0`.
- Always work on copies to protect input arrays.
- Use `pathlib.Path` for all file I/O.
- Handle missing or corrupt files and sidecar data gracefully.

### UI Patterns (PySide6)
- Inherit from standard `QWidget` subclasses.
- Use `QTimer` for throttling expensive UI updates.
- Ensure proper parent-child relationships for Qt object cleanup.
- When subclassing `QListWidget`, use `itemSelectionChanged` and `item.setSelected(True)` to ensure proper signal emission (calling `setCurrentRow()` alone does not trigger selection signals).

---

## AI Agents

If you are using AI coding agents (like Claude Dev, OpenCode, etc.), please refer to [AGENTS.md](AGENTS.md) for project-specific instructions and guidelines for agent behavior.
