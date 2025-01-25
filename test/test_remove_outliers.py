# tests/test_remove_outliers.py
# References: Tiffany's Breast Cancer Predictor and ChatGPT were used as references


# To test remove_outliers(), we need a couple different cases: 
# 1. Testing if it can remove basic outliers ith typical inputs
# 2. Edge Case: can it handle invalid input
# 3. Edge Case: how does it work with boundary values and empty results

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pytest
import pandas as pd
from src.remove_outliers import remove_outliers

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'values': [1, 2, 3, 100, 4, 5, -10, 50],
        'other': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    })

class TestRemoveOutliers:
    """Test suite for remove_outliers function."""

    # NORMAL CASES
    def test_basic_functionality(self, sample_df):
        """Test normal case: basic outlier removal with typical inputs."""
        result = remove_outliers(sample_df, 'values', 0, 10)
        
        # Normal expectations:
        assert len(result) == 5  # Expected number of rows within range
        assert all(result['values'].between(0, 10))  # All values within bounds
        assert 'other' in result.columns  # Other columns preserved

    # EDGE CASES
    def test_invalid_inputs(self):
        """Test edge cases: invalid input handling."""
        # Edge case 1: Non-DataFrame input
        with pytest.raises(TypeError):
            remove_outliers([1, 2, 3], 'values', 0, 10)

        # Edge case 2: Non-existent column
        df = pd.DataFrame({'A': [1, 2, 3]})
        with pytest.raises(ValueError):
            remove_outliers(df, 'B', 0, 10)

    def test_edge_cases(self, sample_df):
        """Test edge cases: boundary values and empty results."""
        # Edge case 3: Boundary values (testing inclusive bounds)
        result = remove_outliers(sample_df, 'values', 1, 5)
        assert 1 in result['values'].values  # Lower bound included
        assert 5 in result['values'].values  # Upper bound included

        # Edge case 4: No values in range
        result = remove_outliers(sample_df, 'values', 200, 300)
        assert len(result) == 0  # Empty DataFrame returned
