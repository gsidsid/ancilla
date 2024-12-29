# ancilla/tests/confest.py
import pytest
from unittest.mock import Mock
import os
import json

@pytest.fixture
def sample_responses():
    """Load sample API responses for testing"""
    path = os.path.join(os.path.dirname(__file__), 'test_data/sample_responses')
    responses = {}
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            with open(os.path.join(path, filename)) as f:
                responses[filename[:-5]] = json.load(f)
    return responses

@pytest.fixture
def mock_client():
    """Create a mock Polygon REST client"""
    return Mock()
