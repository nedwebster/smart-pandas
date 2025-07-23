from pydantic import BaseModel


class Tag(BaseModel):
    """Class to represent a specific tag assigned to a column.

    Parameters
    ----------
    name: str
        The name of the tag.
    data_attribute_name: str
        The name of the data attribute that the tag is associated with.
    config_limit: tuple[int | None, int | None] = (None, None)
        The upper and lower limit for the count of this tag in the config. None represents no limit. Eg, a tag limit of (1, None) means that there must be at
        least one column with this tag in the config, but there can be more.
    required: bool = False
        Whether a column with this tag is required for a valid dataset.
    compatible_with: set[str] = set()
        A set of tag names that this tag is compatible with.

    """

    name: str
    data_attribute_name: str
    config_limit: tuple[int | None, int | None] = (None, None)
    required: bool = False
    compatible_with: set[str] = set()

    def __eq__(self, other: str) -> bool:
        return self.name == other

    def __repr__(self) -> str:
        return f"Tag(name='{self.name}')"

    def __hash__(self):
        return hash(repr(self))


def get_compatible_tags(tag_name: str, tag_pairs: list[tuple[str, str]]) -> set[str]:
    """Get all tags compatible with a given tag name, based on a list of compatible tag tuples."""
    relevant_tag_pairs = [pair for pair in tag_pairs if tag_name in pair]
    compatible_tags = set([tag for pair in relevant_tag_pairs for tag in pair if tag != tag_name])
    return compatible_tags


# List of tag pairs that are compatible with each other, ensuring that the tag pairs are symmetric.
TAG_COMPATIBILITY_PAIRS = [
    ("raw_feature", "model_feature"),
    ("raw_feature", "row_timestamp"),
    ("derived_feature", "model_feature"),
    ("metadata", "unique_identifier"),
    ("metadata", "row_timestamp"),
]

# Collection of predefined tag objects
TAGS: dict[str, Tag] = {
    "target": Tag(
        name="target",
        data_attribute_name="target",
        config_limit=(1, 1),
        compatible_with=get_compatible_tags("target", TAG_COMPATIBILITY_PAIRS),
    ),
    "raw_feature": Tag(
        name="raw_feature",
        data_attribute_name="raw_features",
        config_limit=(1, None),
        compatible_with=get_compatible_tags("raw_feature", TAG_COMPATIBILITY_PAIRS),
    ),
    "derived_feature": Tag(
        name="derived_feature",
        data_attribute_name="derived_features",
        compatible_with=get_compatible_tags("derived_feature", TAG_COMPATIBILITY_PAIRS),
    ),
    "metadata": Tag(
        name="metadata",
        data_attribute_name="metadata",
        compatible_with=get_compatible_tags("metadata", TAG_COMPATIBILITY_PAIRS),
    ),
        "unique_identifier": Tag(
        name="unique_identifier",
        data_attribute_name="unique_identifier",
        config_limit=(1, 1),
        required=True,
        compatible_with=get_compatible_tags("unique_identifier", TAG_COMPATIBILITY_PAIRS),
    ),
    "model_feature": Tag(
        name="model_feature",
        data_attribute_name="model_features",
        config_limit=(1, None),
        compatible_with=get_compatible_tags("model_feature", TAG_COMPATIBILITY_PAIRS),
    ),
    "row_timestamp": Tag(
        name="row_timestamp",
        data_attribute_name="row_timestamp",
        config_limit=(1, 1),
        required=True,
        compatible_with=get_compatible_tags("row_timestamp", TAG_COMPATIBILITY_PAIRS),
    ),
    "weight": Tag(
        name="weight",
        data_attribute_name="weight",
        config_limit=(None, 1),
        compatible_with=get_compatible_tags("weight", TAG_COMPATIBILITY_PAIRS),
    )
}
