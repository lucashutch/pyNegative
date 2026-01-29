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

## Roadmap / TODO

Track planned features and project goals.

### User Requested Features
- [x] **Gallery View**: Browse all photos in a folder in a grid or one-by-one.
- [x] **Sidecar Files**: Store edit settings in sidecars natively next to raw originals.
- [ ] **Carousel View**: edit carousel should scroll horizontally
- [ ] **Udate edit sliders** Edit sliders should be: Exposure, Contrast, Blacks, Whites, Shadows, Highlights, and Sharpening, with the sharpening section being in a new sub-section under details, where there should also be a de-noise option.
- [ ] **Zoom Support**: Zoom into photos during editing for fine control.
- [ ] **Rating System**: Rate photos with stars; store in sidecars/metadata.
- [ ] **Batch Export**: Select and process multiple files at once.
- [ ] **Gallery Filtering**: Filter shown images by rating/metadata.
- [ ] **Sync Settings**: Copy/paste edit changes from one photo to others.
- [ ] **Auto-Enhance Mode**: Automatically adjust tone-mapping to look "good" (auto-exposure/auto-levels).
- [ ] **Reset slider double click**: add double click to restore default values on a slider

### AI Suggested Additions
- [ ] **Live Histogram**: Real-time luminance histogram display in the UI.
- [ ] **Archival Export Preset**: Reduce storage footprint (e.g., 16-bit DNG or high-quality HEIF) while retaining latitude.
- [ ] **Film Simulations / LUTs**: Apply built-in or custom Look-Up Tables.
- [ ] **Virtual Copies**: Allow multiple different edits for a single RAW file.
- [ ] **Batch Renaming**: Sequence-based renaming for folder exports.
