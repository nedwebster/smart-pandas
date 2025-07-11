from itertools import combinations
from smart_pandas.config.tag import create_tag, TAG_CONFIGS, Tag
from smart_pandas.config.tag_set import TagSet


def test_tag_configs_creation():
    for tag_name in TAG_CONFIGS.keys():
        tag = create_tag(tag_name)
        assert tag.name == tag_name


def test_tag_equality():
    tag1 = create_tag("raw_feature")
    tag2 = create_tag("raw_feature")
    assert tag1 == tag2


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
