#!/bin/bash
set -e

# This script builds the PyQt5 application into a macOS .app bundle.
# It should be run from the root of the 'repo' directory.

APP_NAME="RefactorApp"
SCRIPT_PATH="app/main.py"

# Clean up previous PyInstaller artifacts
echo "--- Cleaning up previous PyInstaller artifacts ---"
rm -rf "dist"
rm -rf "build/$APP_NAME"
rm -f "$APP_NAME.spec"


echo "--- Generating hidden imports from requirements.txt ---"
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
        # Skip empty lines, comments, and pyinstaller itself
        if [[ -z "$requirement" ]] || [[ "$requirement" == \#* ]] || [[ "$requirement" == -* ]] || [[ "$requirement" == pyinstaller* ]]; then
            continue
        fi
        # Extract package name (part before any special characters like ==, >=, [, etc.)
        package_name=$(echo "$requirement" | sed -e 's/[~=<>!\[].*//')
        
        # Handle known cases where pip package name differs from import name
        if [[ "$package_name" == "mlflow-skinny" ]]; then
            package_name="mlflow"
        elif [[ "$package_name" == "python-dotenv" ]]; then
            package_name="dotenv"
        fi

        echo "Adding hidden import for top-level package: $package_name"
        PYINSTALLER_ARGS+=(--hidden-import "$package_name")
    done < "app/requirements.txt"
else
    echo "Warning: app/requirements.txt not found."
fi

# Add specific sub-modules that PyInstaller's static analysis might miss.
echo "--- Adding specific sub-module imports for reliability ---"
PYINSTALLER_ARGS+=(--hidden-import "PyQt5.sip")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.logging")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.loops")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.protocols.http")
PYINSTALLER_ARGS+=(--hidden-import "uvicorn.protocols.websockets")
PYINSTALLER_ARGS+=(--hidden-import "websockets.legacy.protocol")

echo "--- Building macOS Application ---"
pyinstaller "${PYINSTALLER_ARGS[@]}" "$SCRIPT_PATH"

echo "✅ macOS application built successfully in the 'dist' directory." 