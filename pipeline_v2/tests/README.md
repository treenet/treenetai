# Test Suite

Comprehensive unit tests for pipeline_v2 using pytest.

## Overview

The test suite covers:
- ✅ Configuration management
- ✅ Data processing (timestamps, resampling, merging)
- ✅ Segmentation (normalization, extraction)
- ✅ Gap injection (generation, application)
- 🔄 Model architecture (TODO)
- 🔄 Training pipeline (TODO)
- 🔄 Visualization (TODO)

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_config.py
pytest tests/test_processors.py
pytest tests/test_segmentation.py
pytest tests/test_gap_injection.py
```

### Run Specific Test Class
```bash
pytest tests/test_config.py::TestSegmentConfig
pytest tests/test_processors.py::TestDataResampler
```

### Run Specific Test Function
```bash
pytest tests/test_config.py::TestSegmentConfig::test_input_steps_calculation
```

### Run with Coverage Report
```bash
pytest --cov=src --cov-report=html
```

View coverage report: `htmlcov/index.html`

### Run with Verbose Output
```bash
pytest -v
```

### Run Only Fast Tests
```bash
pytest -m "not slow"
```

## Test Organization

### conftest.py
Shared fixtures for all tests:
- `temp_dir` - Temporary directory for test outputs
- `sample_10min_data` - Sample 10-minute resolution data
- `sample_hourly_data` - Sample hourly resolution data
- `sample_daily_data` - Sample daily meteo data
- `sample_segment_input` - Sample 30-day input segment (11 channels)
- `sample_segment_output` - Sample 30-day output segment (3 channels)
- `sample_normalized_segment` - Sample normalized data [0, 1]
- `sample_normalization_params` - Sample min/max parameters
- `sample_gap_spec` - Sample gap specification
- `sample_numpy_segment` - Sample numpy array segment
- `sample_numpy_target` - Sample numpy array target

### test_config.py
Tests for configuration system:
- **TestDataPaths** - Path configuration
- **TestSegmentConfig** - Segment parameters
- **TestDataConfig** - Channel definitions
- **TestGapConfig** - Gap injection settings
- **TestModelConfig** - Model architecture
- **TestPipelineConfig** - Full pipeline integration
- **TestConfigValidation** - Parameter validation

**Coverage:** ~40 tests

### test_processors.py
Tests for data processing:
- **TestTimestampProcessor** - UTC/local conversion
- **TestDataResampler** - Hourly/daily resampling
- **TestDataMerger** - Channel merging
- **TestYearGridBuilder** - Year-level grids
- **TestDataProcessor** - Integration workflow
- **TestEdgeCases** - Error handling

**Coverage:** ~25 tests

### test_segmentation.py
Tests for segmentation:
- **TestNormalizer** - Min-max normalization
- **TestSegmentExtractor** - Segment extraction
- **TestSegmentMetadata** - Metadata structure
- **TestEdgeCases** - Gaps, NaN, custom lengths

**Coverage:** ~20 tests

### test_gap_injection.py
Tests for gap injection:
- **TestGapGenerator** - Gap specification generation
- **TestGapInjector** - Gap application to data
- **TestGapGeneratorEdgeCases** - Short segments, single channel
- **TestGapInjectorEdgeCases** - Boundary gaps, full coverage

**Coverage:** ~25 tests

## Test Coverage Summary

| Module | Coverage | Tests |
|--------|----------|-------|
| `src/config.py` | ~95% | 40 |
| `src/data/processors.py` | ~85% | 25 |
| `src/data/segmentation.py` | ~80% | 20 |
| `src/gaps/gap_injection.py` | ~90% | 25 |
| `src/data/loaders.py` | 0% | 0 (TODO) |
| `src/models/tcn.py` | 0% | 0 (TODO) |
| `src/models/training.py` | 0% | 0 (TODO) |
| `src/visualization/` | 0% | 0 (TODO) |

**Total:** ~110 tests covering ~50% of codebase

## Writing New Tests

### Template for New Test

```python
"""Tests for new_module."""

import pytest
from src.module import MyClass


class TestMyClass:
    """Test MyClass functionality."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        obj = MyClass()
        result = obj.method()
        assert result is not None
    
    def test_with_fixture(self, sample_10min_data):
        """Test using shared fixture."""
        obj = MyClass()
        result = obj.process(sample_10min_data)
        assert len(result) > 0
    
    def test_error_handling(self):
        """Test error conditions."""
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input=True)
```

### Using Fixtures

```python
def test_with_multiple_fixtures(
    self,
    sample_segment_input,
    sample_segment_output,
    temp_dir
):
    """Test using multiple fixtures."""
    # Use fixtures as needed
    assert len(sample_segment_input) > 0
    assert len(sample_segment_output) > 0
    assert temp_dir.exists()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("segment_days,expected_steps", [
    (30, 4320),
    (45, 6480),
    (60, 8640),
])
def test_segment_lengths(self, segment_days, expected_steps):
    """Test different segment lengths."""
    config = SegmentConfig(segment_days=segment_days)
    assert config.input_steps == expected_steps
```

## Testing Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Clear Test Names
```python
# Good
def test_normalize_handles_constant_values(self):

# Bad
def test_norm(self):
```

### 3. Arrange-Act-Assert Pattern
```python
def test_feature(self):
    # Arrange
    config = SegmentConfig(segment_days=30)
    processor = Processor(config)
    
    # Act
    result = processor.process(data)
    
    # Assert
    assert result is not None
    assert len(result) > 0
```

### 4. Test Edge Cases
- Empty inputs
- Single-element inputs
- Very large inputs
- Invalid parameters
- Boundary conditions

### 5. Use Descriptive Assertions
```python
# Good
assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"

# Bad
assert len(segments) == 3
```

## Continuous Integration

### GitHub Actions Workflow (TODO)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=src
```

## Future Improvements

### High Priority
- [ ] Tests for `DataLoaders` (file I/O)
- [ ] Tests for TCN model architecture
- [ ] Tests for training pipeline
- [ ] Tests for visualization modules

### Medium Priority
- [ ] Integration tests (end-to-end workflows)
- [ ] Performance benchmarks
- [ ] Regression tests with saved outputs

### Low Priority
- [ ] Property-based testing (Hypothesis)
- [ ] Mutation testing
- [ ] Load testing for large datasets

## Troubleshooting

### Import Errors
If you get import errors:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

Or use pytest with explicit path:
```bash
python -m pytest tests/
```

### Slow Tests
Skip slow tests during development:
```bash
pytest -m "not slow"
```

Mark slow tests:
```python
@pytest.mark.slow
def test_large_dataset(self):
    pass
```

### Coverage Not Generated
Install coverage plugin:
```bash
pip install pytest-cov
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Test Discovery Issues
Ensure:
1. Test files start with `test_`
2. Test classes start with `Test`
3. Test methods start with `test_`
4. `__init__.py` files exist in test directories (not required but recommended)

## Dependencies

Install test dependencies:
```bash
pip install pytest pytest-cov numpy pandas
```

Or use full requirements:
```bash
pip install -r requirements.txt
```

## Test Metrics

Target metrics:
- **Line Coverage:** >80%
- **Branch Coverage:** >75%
- **Test Success Rate:** 100%
- **Average Test Time:** <0.1s per test

Current metrics:
- Line Coverage: ~50% (110 tests implemented)
- Branch Coverage: ~45%
- Test Success Rate: 100%
- Average Test Time: ~0.05s per test

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% coverage for new code
4. Add docstrings to test functions
5. Update this README if adding new test files
