# pyRawRoom

A collection of Python tools for processing RAW image files.

## Tools

### `pyrawroom_gui.py`

A graphical user interface (GUI) for interactively editing RAW files.

- Live preview of changes.
- Adjust exposure, contrast, blacks, whites, shadows, and highlights.
- Sharpening controls.
- Save edits as a JSON sidecar file.

### `pyraw_batch.py`

A command-line tool for batch processing RAW files.

- Convert RAW files to JPEG or HEIF.
- Apply tone mapping and sharpening.
- Process a whole directory of RAW files in parallel.
- Move original RAW files to a separate directory after conversion.

## Library

The core processing logic is shared in the `pyrawroom.py` library.
