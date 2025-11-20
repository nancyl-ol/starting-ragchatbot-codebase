#!/bin/bash
# Code formatting script - Formats code using isort and black

set -e

echo "🔧 Running code formatters..."
echo ""

echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py
echo "✅ Imports sorted"
echo ""

echo "🎨 Formatting code with black..."
uv run black backend/ main.py
echo "✅ Code formatted"
echo ""

echo "✨ All formatting complete!"
