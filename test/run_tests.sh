#!/bin/bash

# Dark_Circle Test Runner Script
# Provides convenient options for running tests with various configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
COVERAGE=false
PARALLEL=false
VERBOSE=false
MARKERS=""
TEST_PATH="test/"
HTML_REPORT=false
SLOW_TESTS=true

# Help message
show_help() {
    cat << EOF
Usage: ./run_tests.sh [OPTIONS]

Options:
    -h, --help              Show this help message
    -c, --coverage          Run with coverage report
    -p, --parallel          Run tests in parallel
    -v, --verbose           Run with verbose output
    -m, --markers MARKERS   Run tests with specific markers (e.g., "unit", "integration")
    -s, --skip-slow         Skip slow tests
    -f, --file FILE         Run specific test file
    --html                  Generate HTML coverage report
    
Examples:
    ./run_tests.sh                          # Run all tests
    ./run_tests.sh -c                       # Run with coverage
    ./run_tests.sh -p -v                    # Run in parallel with verbose output
    ./run_tests.sh -m unit                  # Run only unit tests
    ./run_tests.sh -f test_models.py        # Run specific test file
    ./run_tests.sh -c --html                # Generate HTML coverage report
    ./run_tests.sh -s                       # Skip slow tests

Markers:
    unit            Unit tests
    integration     Integration tests
    slow            Slow-running tests
    db              Database tests
    gpu             GPU-dependent tests

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -s|--skip-slow)
            SLOW_TESTS=false
            shift
            ;;
        -f|--file)
            TEST_PATH="test/$2"
            shift 2
            ;;
        --html)
            HTML_REPORT=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add test path
PYTEST_CMD="$PYTEST_CMD $TEST_PATH"

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add markers
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKERS"
fi

# Skip slow tests
if [ "$SLOW_TESTS" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add coverage
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=../model-train --cov=../ensemble-train --cov=../server-load"
    
    if [ "$HTML_REPORT" = true ]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html --cov-report=term"
    else
        PYTEST_CMD="$PYTEST_CMD --cov-report=term-missing"
    fi
fi

# Print configuration
echo -e "${GREEN}=== Dark_Circle Test Runner ===${NC}"
echo -e "Test path:     $TEST_PATH"
echo -e "Coverage:      $COVERAGE"
echo -e "Parallel:      $PARALLEL"
echo -e "Verbose:       $VERBOSE"
echo -e "Markers:       ${MARKERS:-none}"
echo -e "Skip slow:     $([ "$SLOW_TESTS" = false ] && echo "yes" || echo "no")"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found${NC}"
    echo "Please install test dependencies:"
    echo "  pip install -r test/test_requirements.txt"
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running command: $PYTEST_CMD${NC}"
echo ""

# Execute
$PYTEST_CMD

# Capture exit code
EXIT_CODE=$?

# Print summary
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    
    if [ "$HTML_REPORT" = true ]; then
        echo -e "${GREEN}HTML coverage report generated: htmlcov/index.html${NC}"
    fi
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

exit $EXIT_CODE
