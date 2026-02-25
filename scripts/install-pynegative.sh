#!/usr/bin/env bash
#
# pyNegative Installer for macOS and Linux
# Uses uv tool to install directly from GitHub
#

set -e
set -o pipefail

# Configuration
APP_NAME="pyNegative"
REPO="lucashutch/pyNegative"
GITHUB_URL="https://github.com/$REPO.git"

# Detect OS
OS=""
case "$(uname -s)" in
Linux*) OS="linux" ;;
Darwin*) OS="macos" ;;
*) OS="unknown" ;;
esac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check for silent mode
SILENT=false
for arg in "$@"; do
	case $arg in
	--silent | -silent | --yes | -yes | -s | --unattended | -y)
		SILENT=true
		;;
	esac
done

if [ ! -t 0 ]; then
	SILENT=true
fi

print_info() { [ "$SILENT" = false ] && echo -e "${CYAN}$1${NC}" >&2 || true; }
print_success() { [ "$SILENT" = false ] && echo -e "${GREEN}$1${NC}" >&2 || true; }
print_error() { echo -e "${RED}$1${NC}" >&2 || true; }
print_warning() { [ "$SILENT" = false ] && echo -e "${YELLOW}$1${NC}" >&2 || true; }

# Global error handler: always print a clear failure message (even in silent mode)
on_error() {
	rc=$?
	cmd="${BASH_COMMAND:-unknown}"
	echo -e "${RED}Install failed (exit ${rc}) while running: ${cmd}${NC}" >&2
	exit ${rc}
}
trap 'on_error' ERR

# Check/Install uv
check_uv() { command -v uv &>/dev/null; }

install_uv() {
	print_info "\nInstalling uv (Python package manager)..."
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
	if ! check_uv; then
		# Try to find it in common install locations
		if [ -f "$HOME/.local/bin/uv" ]; then
			export PATH="$HOME/.local/bin:$PATH"
		elif [ -f "$HOME/.cargo/bin/uv" ]; then
			export PATH="$HOME/.cargo/bin:$PATH"
		fi
	fi
	check_uv
}

# Fetch latest tag
get_latest_tag() {
	print_info "Fetching latest release tag..." >&2
	TAG=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
	if [ -z "$TAG" ]; then
		print_warning "Could not fetch latest tag, falling back to main branch." >&2
		echo "main"
	else
		echo "$TAG"
	fi
}

show_welcome() {
	if [ "$SILENT" = true ]; then return; fi
	clear 2>/dev/null || true
	echo "========================================"
	print_info "     pyNegative Installer for Unix     "
	echo "========================================"
	echo ""
	echo "This installer will:"
	echo "  1. Install uv (Python package manager) if needed"
	echo "  2. Resolve the latest pyNegative release"
	echo "  3. Install pyNegative using 'uv tool install'"
	echo "  4. Download the lens database"
	echo "  5. Create application menu entries"
	echo ""
}

create_linux_desktop() {
	print_info "\nCreating application menu entry..."
	mkdir -p "$HOME/.local/share/applications"

	# Try to find the binary
	BIN_PATH=$(command -v pynegative || echo "$HOME/.local/bin/pynegative")

	cat >"$HOME/.local/share/applications/pynegative.desktop" <<EOF
[Desktop Entry]
Name=pyNegative
Comment=RAW Image Processor
Exec=$BIN_PATH
Icon=pynegative
Type=Application
Categories=Graphics;Photography;
Terminal=false
StartupNotify=true
EOF
	print_success "Application menu entry created!"
}

create_macos_app() {
	print_info "\nCreating macOS app bundle..."
	APP_BUNDLE="$HOME/Applications/pyNegative.app"
	mkdir -p "$APP_BUNDLE/Contents/MacOS"
	mkdir -p "$APP_BUNDLE/Contents/Resources"

	# Try to find the binary
	BIN_PATH=$(command -v pynegative || echo "$HOME/.local/bin/pynegative")

	cat >"$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>pynegative_launch</string>
    <key>CFBundleIdentifier</key>
    <string>com.pynegative.app</string>
    <key>CFBundleName</key>
    <string>pyNegative</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
</dict>
</plist>
EOF

	cat >"$APP_BUNDLE/Contents/MacOS/pynegative_launch" <<EOF
#!/bin/bash
exec "$BIN_PATH"
EOF
	chmod +x "$APP_BUNDLE/Contents/MacOS/pynegative_launch"
	print_success "macOS app bundle created at: $APP_BUNDLE"
}

do_install() {
	TAG=$(get_latest_tag)
	INSTALL_URL="git+$GITHUB_URL"
	[ "$TAG" != "main" ] && INSTALL_URL="$INSTALL_URL@$TAG"

	print_info "\nInstalling pyNegative ($TAG) via uv tool..."
	uv tool install "$INSTALL_URL" --force

	print_info "\nDownloading lens database..."
	# Determine data directory
	DATA_DIR=$(uv tool run --from "$INSTALL_URL" python3 -c "from platformdirs import user_data_dir; print(user_data_dir('pyNegative'))" 2>/dev/null || echo "$HOME/.local/share/pyNegative")
	LENS_DIR="$DATA_DIR/data/lensfun"
	mkdir -p "$LENS_DIR"

	uv tool run --from "$INSTALL_URL" download-lens-db --output-dir "$LENS_DIR" --quiet

	if [ "$OS" = "macos" ]; then
		create_macos_app
	else
		create_linux_desktop
	fi

	print_success "\npyNegative installed successfully!"

	# Path warning
	if ! command -v pynegative &>/dev/null; then
		print_warning "\nWarning: The installation directory is not in your PATH."
		print_warning "You may need to add ~/.local/bin to your PATH or run 'uv tool update-shell'."
	fi
}

do_uninstall() {
	print_warning "\nUninstalling pyNegative..."
	uv tool uninstall pynegative || true

	if [ "$OS" = "macos" ]; then
		rm -rf "$HOME/Applications/pyNegative.app"
	else
		rm -f "$HOME/.local/share/applications/pynegative.desktop"
	fi
	print_success "pyNegative has been uninstalled."
}

show_installed_menu() {
	if [ "$SILENT" = true ]; then
		# Non-interactive mode (e.g. curl | bash): automatically update to latest
		# Print a short status message even when running non-interactively so
		# users piping the installer get visible feedback instead of a silent error.
		>&2 echo -e "${CYAN}Non-interactive mode detected: updating pyNegative to the latest version...${NC}"
		do_install
		return
	fi

	echo ""
	print_info "pyNegative is already installed."
	echo ""
	echo "What would you like to do?"
	echo "  1) Update to the latest version"
	echo "  2) Reinstall pyNegative"
	echo "  3) Uninstall pyNegative"
	echo "  4) Cancel"
	echo ""

	read -p "Enter your choice (1-4): " choice
	case $choice in
	1 | 2) do_install ;;
	3)
		read -p "Are you sure you want to uninstall? (y/n) " -n 1 -r
		echo ""
		[[ $REPLY =~ ^[Yy]$ ]] && do_uninstall
		;;
	*) echo "Operation cancelled." ;;
	esac
}

main() {
	if [ "$OS" = "unknown" ]; then
		print_error "Unsupported OS"
		exit 1
	fi

	show_welcome

	if ! check_uv; then
		install_uv || {
			print_error "Failed to install uv"
			exit 1
		}
	fi

	# Check if already installed
	if uv tool list | grep -q "pynegative"; then
		show_installed_menu
	else
		if [ "$SILENT" = false ]; then
			read -p "Continue with installation? (y/n) " -n 1 -r
			echo ""
			if [[ ! $REPLY =~ ^[Yy]$ ]]; then
				echo "Installation cancelled."
				exit 0
			fi
		fi
		do_install
	fi
}

main
