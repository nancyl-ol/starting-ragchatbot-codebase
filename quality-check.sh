#!/bin/bash
# Complete quality check script - Runs all quality checks and tests

set -e

echo "🚀 Running complete quality checks..."
echo ""
echo "================================"
echo "1. CODE QUALITY CHECKS"
echo "================================"
echo ""

./lint.sh

echo ""
echo "================================"
echo "2. RUNNING TESTS"
echo "================================"
echo ""

cd backend
if uv run pytest; then
    echo "✅ All tests passed"
else
    echo "❌ Some tests failed"
    exit 1
fi
cd ..

echo ""
echo "================================"
echo "✨ ALL CHECKS PASSED! ✨"
echo "================================"
