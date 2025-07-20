#!/bin/bash
set -e

OUTPUT_DIR="build/dist"
OUTPUT_FILENAME="refactor-extension.vsix"

echo "--- Building VS Code Extension ---"

mkdir -p "$OUTPUT_DIR"

cd "plugin"

npm install

npm install --prefix ./ vsce

./node_modules/.bin/vsce package --out "../$OUTPUT_DIR/$OUTPUT_FILENAME"

cd ..

echo "✅ VS Code extension built successfully."
