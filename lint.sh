#!/bin/bash
# Linting script - Runs code quality checks without modifying files

set -e

echo "🔍 Running code quality checks..."
echo ""

echo "📋 Checking code style with flake8..."
if uv run flake8 backend/ main.py; then
    echo "✅ flake8 passed"
else
    echo "❌ flake8 found issues"
    exit 1
fi
echo ""

echo "🔎 Checking types with mypy..."
if uv run mypy backend/ main.py; then
    echo "✅ mypy passed"
else
    echo "⚠️  mypy found type issues (informational - not blocking)"
fi
echo ""

echo "📦 Checking import sorting with isort..."
if uv run isort --check-only backend/ main.py; then
    echo "✅ Import sorting is correct"
else
    echo "❌ Imports need sorting (run ./format.sh to fix)"
    exit 1
fi
echo ""

echo "🎨 Checking code formatting with black..."
if uv run black --check backend/ main.py; then
    echo "✅ Code formatting is correct"
else
    echo "❌ Code needs formatting (run ./format.sh to fix)"
    exit 1
fi
echo ""

echo "✨ All quality checks passed!"
