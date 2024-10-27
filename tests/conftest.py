import pytest


@pytest.fixture
def column_labels():
    from smart_pandas.label import build_column_label, COLUMN_LABEL_NAMES

    return [build_column_label(name) for name in COLUMN_LABEL_NAMES]
