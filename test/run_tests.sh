#!/usr/bin/env bash
# Run all UCI expect tests
# Usage: ./run_tests.sh [options] <path-to-uci-binary>

verbose=0
exitfirst=0
binary=""

for arg in "$@"; do
    case "$arg" in
        -v|--verbose) verbose=1 ;;
        --exitfirst) exitfirst=1 ;;
        -*) echo "Unknown option: $arg"; exit 1 ;;
        *) binary="$arg" ;;
    esac
done

if test -z "$binary"; then
    echo "Usage: $0 [options] <path-to-uci-binary>"
    echo "Options:"
    echo "  -v, --verbose   Show detailed test output"
    echo "  --exitfirst     Stop on first failure"
    exit 1
fi

export UCI_BINARY="$(realpath "$binary")"

if test ! -e "$UCI_BINARY"; then
    echo "ERROR: '$binary' does not exist."
    exit 1
fi

if test ! -x "$UCI_BINARY"; then
    echo "ERROR: '$binary' is not executable."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

passed=0
failed=0

for test_file in test_*.exp; do
    if test -f "$test_file"; then
        if test $verbose -eq 1; then
            if expect "$test_file"; then
                passed=$((passed + 1))
                echo "PASS: $test_file"
            else
                failed=$((failed + 1))
                echo "FAIL: $test_file"
                test $exitfirst -eq 1 && exit 1
            fi
        else
            if expect "$test_file" >/dev/null 2>&1; then
                passed=$((passed + 1))
                echo "PASS: $test_file"
            else
                failed=$((failed + 1))
                echo "FAIL: $test_file"
                test $exitfirst -eq 1 && exit 1
            fi
        fi
    fi
done

if test $failed -gt 0; then
    echo "$passed passed, $failed failed"
    exit 1
fi
exit 0
