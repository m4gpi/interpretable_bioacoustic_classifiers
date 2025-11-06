import pytest
import os
import pathlib
import yaml

@pytest.fixture(scope="session")
def fixtures_path() -> pathlib.Path:
    return pathlib.Path(os.path.dirname(__file__)) / "fixtures"
