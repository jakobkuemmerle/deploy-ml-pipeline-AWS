import pytest
import yaml
import logging
from unittest import mock
from src.load_config import get_config

def test_get_config_success(tmp_path):
    config_data = {'key': 'value'}
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    result = get_config(config_file)
    
    assert result == config_data

def test_get_config_file_not_found():
    non_existent_file = "non_existent_config.yaml"
    
    result = get_config(non_existent_file)
    
    assert result is None
