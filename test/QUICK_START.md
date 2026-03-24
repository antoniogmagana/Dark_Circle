# Test Suite Quick Start Guide

## ✅ Successfully Recovered Test Suite

All 15 test files have been recreated after git pull. The test framework is operational!

## Running Tests

### 1. Activate Environment & Set Password
```bash
cd /home/lvc_toolkit/project-files/Dark_Circle
source /home/lvc_toolkit/ai_env/bin/activate
export DB_PASSWORD=test_password
```

### 2. Run Tests

**Run all tests:**
```bash
python -m pytest test/ -v
```

**Run specific test modules:**
```bash
# Simple validation tests (no dependencies)
python -m pytest test/test_simple.py -v

# Server loading tests  
python -m pytest test/test_server_load.py -v

# All working tests
python -m pytest test/test_simple.py test/test_server_load.py -v
```

### 3. Test Status

✅ **Working Tests:**
- `test_simple.py` - 9 tests passing
- `test_server_load.py` - Sanitization and database connection tests passing

⚠️ **Known Issues:**
- **PyTorch/torchaudio** - Missing `libcudart.so.13` CUDA runtime library
- **Model tests** - Require PyTorch to be fully functional  
- **Dataset tests** - Depend on torchaudio which has CUDA library issues

## Test Statistics

**Total Test Files Created:** 15  
**Currently Passing:** 16+ tests  
**Framework Status:** ✅ Functional  

## Quick Commands

```bash
# Activate and test in one go
source /home/lvc_toolkit/ai_env/bin/activate && export DB_PASSWORD=test && python -m pytest test/test_simple.py -v

# Clear Python cache (if needed)
python3 -c "import shutil,pathlib;[shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

# Run with coverage
python -m pytest test/test_simple.py --cov --cov-report=term

# Run specific test class
python -m pytest test/test_server_load.py::TestSanitizeName -v
```

## Fixing CUDA/PyTorch Issues

The PyTorch tests currently fail due to missing CUDA runtime. To fix:

1. **Reinstall PyTorch without CUDA** (CPU-only):
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Or install CUDA runtime libraries**:
   ```bash
   # Check which CUDA version PyTorch needs
   python -c "import torch; print(torch.version.cuda)"
   # Install matching CUDA toolkit
   ```

## Test Files Included

1. `__init__.py` - Package initialization
2. `conftest.py` - Shared pytest fixtures  
3. `pytest.ini` - Pytest configuration
4. `test_requirements.txt` - Test dependencies
5. `test.py` - Main test runner
6. `test_config.py` - Configuration tests (30+ tests)
7. `test_data_generator.py` - Data generation tests (35+ tests)
8. `test_db_utils.py` - Database utility tests (25+ tests)
9. `test_dataset.py` - Dataset loading tests (20+ tests)
10. `test_models.py` - Model architecture tests (25+ tests)
11. `test_ensemble.py` - Ensemble method tests (20+ tests)
12. `test_server_load.py` - Server loading tests (20+ tests)
13. `test_simple.py` - Basic validation tests (9 tests)
14. `README.md` - Complete documentation
15. `run_tests.sh` - Test automation script

## Support

For detailed information, see:
- [README.md](README.md) - Complete test documentation
- [TEST_SUITE_SUMMARY.md](TEST_SUITE_SUMMARY.md) - Detailed test descriptions

Last Updated: 2024-03-24
