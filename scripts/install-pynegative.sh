#!/usr/bin/env bash
#
# pyNegative Installer for macOS and Linux
# Downloads latest release (or main) using uv's Python
#

set -e

# Configuration
APP_NAME="pyNegative"
REPO="lucashutch/pyNegative"
INSTALL_DIR="$HOME/.local/share/pyNegative"
VERSION_FILE="$INSTALL_DIR/.version"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
	--silent | -silent | --yes | -yes | -s)
		SILENT=true
		;;
	esac
done

# Print functions
print_info() {
	if [ "$SILENT" = false ]; then
		echo -e "${CYAN}$1${NC}"
	fi
}

print_success() {
	if [ "$SILENT" = false ]; then
		echo -e "${GREEN}$1${NC}"
	fi
}

print_error() {
	echo -e "${RED}$1${NC}"
}

print_warning() {
	if [ "$SILENT" = false ]; then
		echo -e "${YELLOW}$1${NC}"
	fi
}

# Show welcome message
show_welcome() {
	if [ "$SILENT" = true ]; then
		return
	fi

	clear 2>/dev/null || true
	echo "========================================"
	print_info "     pyNegative Installer for Unix     "
	echo "========================================"
	echo ""
	echo "This installer will:"
	echo "  1. Install uv (Python package manager) if needed"
	echo "  2. Download latest pyNegative release (or main branch)"
	echo "  3. Install all dependencies"
	echo "  4. Create application menu entries"
	echo ""
	echo "Installation location: $INSTALL_DIR"
	echo "Detected OS: $OS"
	echo ""
}

# Check if uv is installed
check_uv() {
	if command -v uv &>/dev/null; then
		return 0
	else
		return 1
	fi
}

# Install uv
install_uv() {
	print_info "\nInstalling uv (Python package manager)..."

	if [ "$OS" = "macos" ]; then
		# macOS installation
		if command -v brew &>/dev/null; then
			brew install uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
		else
			curl -LsSf https://astral.sh/uv/install.sh | sh
		fi
	else
		# Linux installation
		curl -LsSf https://astral.sh/uv/install.sh | sh
	fi

	# Add to PATH for this session
	export PATH="$HOME/.local/bin:$PATH"

	if check_uv; then
		print_success "uv installed successfully!"
		return 0
	else
		print_error "Failed to install uv"
		return 1
	fi
}

# Download and install pyNegative using Python
download_install() {
	print_info "\nChecking for latest release..."

	# Python script to handle download and extraction
	# Uses uv to provide Python
	uv run --python 3 python3 -c "
import urllib.request
import json
import zipfile
import io
import os
import sys
import shutil

REPO = '$REPO'
INSTALL_DIR = '$INSTALL_DIR'
VERSION_FILE = os.path.join(INSTALL_DIR, '.version')

def get_latest_version():
    try:
        req = urllib.request.Request(
            f'https://api.github.com/repos/{REPO}/releases/latest',
            headers={'User-Agent': 'pyNegative-Installer', 'Accept': 'application/vnd.github.v3+json'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            return data.get('tag_name', 'main')
    except Exception as e:
        print(f'Could not check releases: {e}')
        return 'main'

def download_and_extract(version, dest_dir):
    if version == 'main':
        url = f'https://github.com/{REPO}/archive/refs/heads/main.zip'
        print(f'Downloading main branch...')
    else:
        url = f'https://github.com/{REPO}/archive/refs/tags/{version}.zip'
        print(f'Downloading {version}...')
    
    try:
        # Download
        with urllib.request.urlopen(url, timeout=60) as response:
            zip_data = io.BytesIO(response.read())
        
        # Extract
        temp_extract = dest_dir + '.extracting'
        if os.path.exists(temp_extract):
            shutil.rmtree(temp_extract)
        
        with zipfile.ZipFile(zip_data, 'r') as zf:
            zf.extractall(temp_extract)
        
        # Find the extracted directory (usually like pyNegative-main/ or pyNegative-v1.0.0/)
        subdirs = [d for d in os.listdir(temp_extract) if os.path.isdir(os.path.join(temp_extract, d))]
        if not subdirs:
            print('Error: No directory found in zip')
            sys.exit(1)
        
        source_dir = os.path.join(temp_extract, subdirs[0])
        
        # Remove old install if exists
        if os.path.exists(dest_dir):
            print('Removing old installation...')
            shutil.rmtree(dest_dir)
        
        # Move to final location
        shutil.move(source_dir, dest_dir)
        shutil.rmtree(temp_extract)
        
        # Save version
        with open(VERSION_FILE, 'w') as f:
            f.write(version)
        
        print(f'Successfully installed {version} to {dest_dir}')
        return True
        
    except Exception as e:
        print(f'Error downloading: {e}')
        sys.exit(1)

# Main logic
latest = get_latest_version()
current = None

if os.path.exists(VERSION_FILE):
    try:
        with open(VERSION_FILE, 'r') as f:
            current = f.read().strip()
    except:
        pass

if current == latest and os.path.exists(os.path.join(INSTALL_DIR, 'pyproject.toml')):
    print(f'Already on latest version: {latest}')
    print('Skipping download, updating dependencies only...')
    sys.exit(2)  # Exit code 2 = already on latest, skip to deps sync
elif current:
    print(f'Update available: {current} -> {latest}')
else:
    print(f'Installing {latest}...')

download_and_extract(latest, INSTALL_DIR)
sys.exit(0)
"

	EXIT_CODE=$?

	if [ $EXIT_CODE -eq 0 ]; then
		print_success "Download and extraction complete!"
		return 0
	elif [ $EXIT_CODE -eq 2 ]; then
		# Already on latest
		return 2
	else
		print_error "Download failed"
		return 1
	fi
}

# Install dependencies
install_dependencies() {
	local update=$1

	if [ "$update" = true ]; then
		print_info "\nUpdating dependencies..."
	else
		print_info "\nInstalling dependencies..."
	fi

	cd "$INSTALL_DIR"

	if uv sync --all-groups; then
		if [ "$update" = true ]; then
			print_success "Dependencies updated successfully!"
		else
			print_success "Dependencies installed successfully!"
		fi
		return 0
	else
		print_error "Failed to install dependencies"
		return 1
	fi
}

# Create Linux .desktop file
create_linux_desktop() {
	print_info "\nCreating application menu entry..."

	# Create applications directory
	mkdir -p "$HOME/.local/share/applications"

	# Create .desktop file
	cat >"$HOME/.local/share/applications/pynegative.desktop" <<EOF
[Desktop Entry]
Name=pyNegative
Comment=RAW Image Processor
Exec=uv run --directory $INSTALL_DIR pyneg-ui
Icon=$INSTALL_DIR/pynegative_icon.png
Type=Application
Categories=Graphics;Photography;
Terminal=false
StartupNotify=true
EOF

	# Update desktop database
	if command -v update-desktop-database &>/dev/null; then
		update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
	fi

	print_success "Application menu entry created!"
}

# Create macOS app bundle
create_macos_app() {
	print_info "\nCreating macOS app bundle..."

	APP_BUNDLE="$HOME/Applications/pyNegative.app"

	# Create app bundle structure
	mkdir -p "$APP_BUNDLE/Contents/MacOS"
	mkdir -p "$APP_BUNDLE/Contents/Resources"

	# Create Info.plist
	cat >"$APP_BUNDLE/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>pynegative</string>
    <key>CFBundleIdentifier</key>
    <string>com.pynegative.app</string>
    <key>CFBundleName</key>
    <string>pyNegative</string>
    <key>CFBundleDisplayName</key>
    <string>pyNegative</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.photography</string>
</dict>
</plist>
EOF

	# Create executable script
	cat >"$APP_BUNDLE/Contents/MacOS/pynegative" <<EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec uv run pyneg-ui
EOF
	chmod +x "$APP_BUNDLE/Contents/MacOS/pynegative"

	# Copy icon if available
	if [ -f "$INSTALL_DIR/pynegative_icon.png" ]; then
		cp "$INSTALL_DIR/pynegative_icon.png" "$APP_BUNDLE/Contents/Resources/pynegative.png"
		plutil -insert CFBundleIconFile -string "pynegative" "$APP_BUNDLE/Contents/Info.plist" 2>/dev/null || true
	fi

	print_success "macOS app bundle created at: $APP_BUNDLE"
}

# Create shortcuts based on OS
create_shortcuts() {
	if [ "$OS" = "macos" ]; then
		create_macos_app
	else
		create_linux_desktop
	fi

	# Ask about desktop shortcut (only in interactive mode)
	if [ "$SILENT" = false ]; then
		echo ""
		read -p "Create desktop shortcut? (y/n) " -n 1 -r
		echo ""
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			if [ "$OS" = "macos" ]; then
				# macOS: create alias on desktop
				ln -sf "$HOME/Applications/pyNegative.app" "$HOME/Desktop/pyNegative.app"
				print_success "Desktop alias created!"
			else
				# Linux: copy .desktop to desktop
				cp "$HOME/.local/share/applications/pynegative.desktop" "$HOME/Desktop/"
				chmod +x "$HOME/Desktop/pynegative.desktop"
				print_success "Desktop shortcut created!"
			fi
		fi
	fi
}

# Uninstall pyNegative
uninstall_pynegative() {
	print_warning "\nUninstalling pyNegative..."

	# Remove installation directory
	if [ -d "$INSTALL_DIR" ]; then
		rm -rf "$INSTALL_DIR"
		print_success "Removed installation directory: $INSTALL_DIR"
	fi

	# Remove Linux desktop entry
	if [ "$OS" != "macos" ]; then
		if [ -f "$HOME/.local/share/applications/pynegative.desktop" ]; then
			rm "$HOME/.local/share/applications/pynegative.desktop"
			print_success "Removed application menu entry"
		fi

		if [ -f "$HOME/Desktop/pynegative.desktop" ]; then
			rm "$HOME/Desktop/pynegative.desktop"
			print_success "Removed desktop shortcut"
		fi
	fi

	# Remove macOS app bundle
	if [ "$OS" = "macos" ]; then
		if [ -d "$HOME/Applications/pyNegative.app" ]; then
			rm -rf "$HOME/Applications/pyNegative.app"
			print_success "Removed app bundle"
		fi

		if [ -L "$HOME/Desktop/pyNegative.app" ]; then
			rm "$HOME/Desktop/pyNegative.app"
			print_success "Removed desktop alias"
		fi
	fi

	print_success "\npyNegative has been uninstalled successfully!"
}

# Install pyNegative
do_install() {
	show_welcome

	if [ "$SILENT" = false ]; then
		read -p "Continue with installation? (y/n) " -n 1 -r
		echo ""
		if [[ ! $REPLY =~ ^[Yy]$ ]]; then
			echo "Installation cancelled."
			exit 0
		fi
	fi

	# Check/install uv
	if ! check_uv; then
		if ! install_uv; then
			print_error "Cannot continue without uv. Please install it manually from https://github.com/astral-sh/uv"
			exit 1
		fi
	else
		print_success "uv is already installed"
	fi

	# Download and install
	if ! download_install; then
		# Check if already on latest (exit code 2)
		if [ $? -eq 2 ]; then
			print_success "Already on latest version!"
		else
			print_error "Installation failed"
			exit 1
		fi
	fi

	# Install dependencies
	if ! install_dependencies false; then
		exit 1
	fi

	# Create shortcuts
	create_shortcuts

	# Success message
	echo ""
	echo "========================================"
	print_success "  pyNegative installed successfully!    "
	echo "========================================"
	echo ""
	echo "You can now launch pyNegative from:"
	if [ "$OS" = "macos" ]; then
		echo "  - Applications folder: ~/Applications/pyNegative.app"
		echo "  - Or by running: uv run pyneg-ui"
	else
		echo "  - Application menu (search for 'pyNegative')"
		echo "  - Or by running: uv run pyneg-ui"
	fi
	echo ""

	if [ "$SILENT" = false ]; then
		read -p "Launch pyNegative now? (y/n) " -n 1 -r
		echo ""
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			cd "$INSTALL_DIR"
			uv run pyneg-ui &
		fi
	fi
}

# Update pyNegative
do_update() {
	print_info "\nUpdating pyNegative..."

	if ! check_uv; then
		print_error "uv is not installed. Please reinstall pyNegative."
		exit 1
	fi

	# Download (will check version and skip if same)
	if ! download_install; then
		# Check if already on latest (exit code 2)
		if [ $? -eq 2 ]; then
			print_success "Already on latest version!"
		else
			print_error "Update failed"
			exit 1
		fi
	fi

	# Update dependencies
	if ! install_dependencies true; then
		exit 1
	fi

	print_success "\npyNegative has been updated successfully!"
}

# Show installed menu
show_installed_menu() {
	echo ""
	print_warning "pyNegative is already installed at:"
	echo "$INSTALL_DIR"

	# Show current version
	if [ -f "$VERSION_FILE" ]; then
		CURRENT_VERSION=$(cat "$VERSION_FILE")
		echo "Current version: $CURRENT_VERSION"
	fi

	echo ""
	echo "What would you like to do?"
	echo "  1) Update pyNegative to the latest version"
	echo "  2) Uninstall pyNegative"
	echo "  3) Cancel"
	echo ""

	if [ "$SILENT" = true ]; then
		# In silent mode, default to update
		do_update
		return
	fi

	read -p "Enter your choice (1-3): " choice

	case $choice in
	1)
		do_update
		;;
	2)
		read -p "Are you sure you want to uninstall? (y/n) " -n 1 -r
		echo ""
		if [[ $REPLY =~ ^[Yy]$ ]]; then
			uninstall_pynegative
		else
			echo "Uninstall cancelled."
		fi
		;;
	3)
		echo "Operation cancelled."
		;;
	*)
		print_error "Invalid choice"
		show_installed_menu
		;;
	esac
}

# Main execution
main() {
	# Check for unsupported OS
	if [ "$OS" = "unknown" ]; then
		print_error "Unsupported operating system: $(uname -s)"
		print_error "This installer supports Linux and macOS only."
		exit 1
	fi

	# Check if already installed
	if [ -f "$INSTALL_DIR/pyproject.toml" ]; then
		show_installed_menu
	else
		do_install
	fi
}

# Run main function
main
