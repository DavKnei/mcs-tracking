# tests/conftest.py
import pytest
import os


@pytest.fixture(scope="session", params=["Test1", "Test2", "Test3", "Test4"])
def test_case_dir(request):
    base_dir = os.path.dirname(__file__)
    case_name = request.param
    case_dir = os.path.join(base_dir, case_name)
    config_file = os.path.join(case_dir, f"config_{case_name.lower()}.yaml")
    return {"case_name": case_name, "case_dir": case_dir, "config_file": config_file}
