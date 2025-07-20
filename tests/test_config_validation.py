from pydantic import ValidationError
import pytest

def test_empty_column_set_error(load_config):
    with pytest.raises(ValidationError, match="Column set is empty"):
        load_config("empty_col_config")


def test_duplicate_column_error(load_config):
    with pytest.raises(ValidationError, match="Duplicate column names found"):    
        load_config("duplicate_col_config")


def test_tag_limit_exceeded_error(load_config):
    with pytest.raises(ValidationError, match="Tag exceeded limit"):
        load_config("tag_limit_exceeded_config")


def test_tag_compatibility_error(load_config):
    with pytest.raises(ValidationError, match="Incompatible tags found"):
        load_config("tag_compatibility_error_config")
