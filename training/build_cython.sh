#!/bin/bash
# Build the Cython fast_features module
set -e

cd "$(dirname "$0")"

echo "Installing Cython if needed..."
uv pip install cython

echo "Building fast_features..."
uv run python setup_cython.py build_ext --inplace

echo "Done! Test with:"
echo "  uv run python -c 'import fast_features; print(fast_features)'"
