# pyNegative Installers

Cross-platform installers for pyNegative.

## Quick Start

### Windows
1. Download `scripts/install-pynegative.bat`
2. Double-click it

### macOS/Linux
```bash
curl -fsSL https://raw.githubusercontent.com/lucashutch/pyNegative/main/scripts/install-pynegative.sh | bash
```

Or manually download `scripts/install-pynegative.sh` and run: `bash install-pynegative.sh`

## Silent Mode

**Windows:** `install-pynegative.bat --silent`

**macOS/Linux:** `bash install-pynegative.sh --silent` (or `-s`, `--yes`)

## What It Does

1. Installs uv (Python package manager) if needed
2. Downloads latest pyNegative release (or main branch)
3. Installs dependencies with `uv sync`
4. Generates platform-specific icons from source
5. Creates Start Menu/Desktop shortcuts

Installations are tracked in a `.version` file - re-running the installer only downloads new versions when available.

## Files

- `install-pynegative.bat` - Windows installer
- `install-pynegative.sh` - macOS/Linux installer  
- `generate_icons.py` - Icon generation (run automatically during install)
- `icons/` - Generated icons directory (created during install, in `.gitignore`)

## Requirements

- **Windows:** Windows 10+, internet connection
- **macOS:** 10.12+, internet connection  
- **Linux:** Modern distro with curl/wget, internet connection

No need to install Python or git - the installer handles everything.

## Updating

Re-run the installer. It checks your installed version against GitHub releases and only re-downloads when needed.

## Troubleshooting

**Windows blocked download:** Right-click the file → Properties → Check "Unblock" → OK

**macOS "App can't be opened":** Right-click the app → Open → Open

**Linux app not in menu:** Run `update-desktop-database ~/.local/share/applications`

**Permission denied:** `chmod +x install-pynegative.sh`

## License

Part of the pyNegative project.
