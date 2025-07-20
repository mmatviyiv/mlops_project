#!/bin/bash
set -e

# This script builds the PyQt5 application into a macOS .app bundle.
# It should be run from the root of the 'repo' directory.

APP_NAME="RefactorApp"
SCRIPT_PATH="app/main.py"
OUTPUT_DIR="build/dist"

echo "--- Generating hidden imports from requirements.txt ---"
# Start with a list of base arguments for PyInstaller
PYINSTALLER_ARGS=(
    --name "$APP_NAME"
    --windowed
    --noconfirm
    --clean
    --add-data "app:app"
)

# Read requirements.txt and add each package as a hidden import
if [ -f "app/requirements.txt" ]; then
    while IFS= read -r requirement || [ -n "$requirement" ]; do
        # Skip empty lines, comments, and flags
        if [[ -z "$requirement" ]] || [[ "$requirement" == \#* ]] || [[ "$requirement" == -* ]] || [[ "$requirement" == pyinstaller* ]]; then
            continue
        fi
        # Extract package name (part before any special characters like ==, >=, [, etc.)
        package_name=$(echo "$requirement" | sed -e 's/[~=<>!\[].*//')
        
        # Handle known cases where pip package name differs from import name
        if [[ "$package_name" == "mlflow-skinny" ]]; then
            package_name="mlflow"
        elif [[ "$package_name" == "PyQt5" ]]; then
            package_name="PyQt5"
        elif [[ "$package_name" == "python-dotenv" ]]; then
            package_name="dotenv"
        # Add other mappings here if needed
        fi

        echo "Adding hidden import for top-level package: $package_name"
        PYINSTALLER_ARGS+=(--hidden-import "$package_name")
    done < "app/requirements.txt"
else
    echo "Warning: app/requirements.txt not found."
fi

# Add specific sub-modules that PyInstaller's static analysis might miss.
# The top-level packages are already added by the loop above, but these explicit
# sub-module imports act as a safeguard for runtime reliability.
echo "--- Adding specific sub-module imports for reliability ---"
PYINSTALLER_ARGS+=(--hidden-import "PyQt5.sip")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.logging")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.loops")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.protocols.http")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.protocols.websockets")
PYINSTALLER_ARGS+=(--hidden-import "websockets.legacy.protocol")

echo "--- Building macOS Application ---"
pyinstaller "${PYINSTALLER_ARGS[@]}" "$SCRIPT_PATH"


echo "--- Packaging Application ---"

# Move the final .app bundle to the clean output directory
mv "dist/$APP_NAME.app" "$OUTPUT_DIR/"

# Clean up temporary files
rm -rf "build/$APP_NAME"
rm -f "$APP_NAME.spec"

echo "✅ macOS application built successfully at $OUTPUT_DIR/$APP_NAME.app" 