# pyNegative Installers

Cross-platform installers for pyNegative.

## Quick Start

**Only ONE file needed to install!**

### Windows
1. Download `scripts/install-pynegative.bat`
2. Double-click it

### macOS/Linux
```bash
curl -fsSL https://raw.githubusercontent.com/lucashutch/pyNegative/main/scripts/install-pynegative.sh | bash
```

Or manually download `scripts/install-pynegative.sh` and run: `bash install-pynegative.sh`

> **Note**: The installer automatically downloads any additional components needed during installation. You only need to download the single installer file.

## Silent Mode

**Windows:** `install-pynegative.bat --silent` (or `--unattended`, `-y`)

**macOS/Linux:** `bash install-pynegative.sh --silent` (or `-s`, `--yes`, `--unattended`)

Note: When the installer is run non-interactively (for example piped via `curl | bash`), it will automatically update an existing pyNegative installation and print a short status message so users see progress. If the installer fails for any reason in silent/non-interactive mode, it will print a concise failure message including the exit code and the failing command.

## What It Does

1. Installs uv (Python package manager) if needed
2. Resolves latest tagged release from GitHub
3. Installs pyNegative using `uv tool install`
4. Downloads lens database using `download-lens-db` command
5. Creates Start Menu/Desktop shortcuts

## Files

- `install-pynegative.bat` - Windows installer
- `install-pynegative.sh` - macOS/Linux installer
- `generate_icons.py` - Icon generation tool
- `run_all_benchmarks.py` - Performance testing suite

## Requirements

- **Windows:** Windows 10+, internet connection
- **macOS:** 10.12+, internet connection
- **Linux:** Modern distro with curl/wget, internet connection

No need to install Python or git - the single installer file handles everything, including downloading additional components automatically.

## Updating

Re-run the installer. It checks your installed version against GitHub releases and only re-downloads when needed.

## Troubleshooting

**Windows blocked download:** Right-click the file → Properties → Check "Unblock" → OK

**macOS "App can't be opened":** Right-click the app → Open → Open

**Linux app not in menu:** Run `update-desktop-database ~/.local/share/applications`

**Permission denied:** `chmod +x install-pynegative.sh`

## License

Part of the pyNegative project.
