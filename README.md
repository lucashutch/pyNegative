# pyRawRoom

A collection of Python tools for processing RAW image files, now organized as a proper Python package.

## Installation

```bash
# Clone the repository
git clone https://github.com/lucashutch/pyRawRoom.git
cd pyRawRoom

# Install in editable mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

## Tools

### UI Editor (`pyraw-ui`)

A graphical user interface (GUI) for interactively editing RAW files.

- Live preview of changes.
- Adjust exposure, contrast, blacks, whites, shadows, and highlights.
- Sharpening controls.
- Save edits as a JSON sidecar file.

Run with:
```bash
pyraw-ui
```

### Batch Processor (`pyraw-batch`)

A command-line tool for batch processing RAW files.

- Convert RAW files to JPEG or HEIF.
- Apply tone mapping and sharpening.
- Process a whole directory of RAW files in parallel.
- Move original RAW files to a separate directory after conversion.

Run with:
```bash
pyraw-batch /path/to/raws -o /path/to/output
```

## Project Structure

```text
pyRawRoom/
├── src/
│   └── pyrawroom/          # Core package
│       ├── core.py         # Image processing logic
│       ├── cli.py          # Batch CLI
│       └── ui.py           # PySide6 GUI
├── tests/                  # Unit tests
└── pyproject.toml          # Build system & dependencies
```

## Development

Run tests with:
```bash
pytest
```
