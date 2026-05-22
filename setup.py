# Compatibility shim — lets Streamlit Community Cloud auto-install
# the local `telco_churn` package via `pip install -e .`.
# All real configuration lives in pyproject.toml.
from setuptools import setup
setup()
