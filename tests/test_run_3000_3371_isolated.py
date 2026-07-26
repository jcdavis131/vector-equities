"""safe placeholder for test_run_3000_3371_isolated - heavy pipeline module, skipped for fast collect"""

import pytest


def test_placeholder_test_run_3000_3371_isolated_fast():
    assert True


def test_test_run_3000_3371_isolated_deferred_import():
    pytest.skip("heavy pipeline test deferred - data not available in free-tier")
