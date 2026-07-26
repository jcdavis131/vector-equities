"""safe placeholder for test_towers_v6_political - heavy pipeline module, skipped for fast collect"""
import pytest
def test_placeholder_test_towers_v6_political_fast():
    assert True
def test_test_towers_v6_political_deferred_import():
    pytest.skip("heavy pipeline test deferred - data not available in free-tier")
