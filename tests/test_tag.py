from itertools import combinations
import pytest
from smart_pandas.config.tag import Tag, get_compatible_tags
from smart_pandas.config.tag_set import TagSet


@pytest.mark.parametrize("tag_name, output", [
    ("a", {"b", "d"}),
    ("b", {"a", "c"}),
    ("c", {"b", "d"}),
    ("d", {"a", "c"}),
    ("e", set()),
])
def test_get_compatible_tags(tag_name, output):
    tag_pairs = [("a", "b"), ("a", "d"), ("b", "c"), ("c", "d")]
    assert get_compatible_tags(tag_name, tag_pairs) == output


def test_compatability(column_tags):
    for column_tag_pair in combinations(column_tags, 2):
        if (column_tag_pair[0] in column_tag_pair[1].compatible_with) and (
            column_tag_pair[1] not in column_tag_pair[0].compatible_with
        ):
            raise ValueError(
                f"Tags {column_tag_pair[0].name} and {column_tag_pair[1].name} have asymmetric compatibility"
            )


def test_tag_set_creation():
    tag_set = TagSet(tags=["raw_feature", "model_feature"])

    assert all(isinstance(tag, Tag) for tag in tag_set.tags)
