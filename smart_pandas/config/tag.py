from typing import Any
from pydantic import BaseModel, field_validator


class Tag(BaseModel):
    """Class to represent a specific tag assigned to a column.

    Parameters
    ----------
    name: str
        The name of the tag.
    compatible_with: List[str]
        A list of tags that this tag is compatible with.
    dataset_limit: Optional[int]
        The maximum number of columns that can be assigned this tag in a dataset. If set to None, there is no limit.

    """

    name: str
    compatible_with: list[str] = []
    dataset_limit: int | None = None

    def __eq__(self, other: str) -> bool:
        return self.name == other

    def __repr__(self) -> str:
        return f"Tag(name='{self.name}')"

    def __hash__(self):
        return hash(repr(self))


TAG_COMPATIBILITY_PAIRS = [
    ("raw_feature", "model_feature"),
    ("raw_feature", "row_timestamp"),
    ("derived_feature", "model_feature"),
    ("metadata", "unique_identifier"),
    ("metadata", "row_timestamp"),
]

def get_compatible_tags(tag_name: str, tag_pairs: list[tuple[str, str]]) -> list[str]:
    """Get all tags compatible with a given tag name, based on a list of compatible tag tuples."""
    relevant_tag_pairs = [pair for pair in tag_pairs if tag_name in pair]
    compatible_tags = [tag for pair in relevant_tag_pairs for tag in pair if tag != tag_name]
    return compatible_tags

# Configuration dictionary for predefined tags
TAG_CONFIGS: dict[str, dict[str, Any]] = {
    "target": {
        "name": "target",
        "compatible_with": get_compatible_tags("target", TAG_COMPATIBILITY_PAIRS),
        "dataset_limit": 1
    },
    "raw_feature": {
        "name": "raw_feature",
        "compatible_with": get_compatible_tags("raw_feature", TAG_COMPATIBILITY_PAIRS),
    },
    "derived_feature": {
        "name": "derived_feature",
        "compatible_with": get_compatible_tags("derived_feature", TAG_COMPATIBILITY_PAIRS),
    },
    "metadata": {
        "name": "metadata",
        "compatible_with": get_compatible_tags("metadata", TAG_COMPATIBILITY_PAIRS),
    },
    "unique_identifier": {
        "name": "unique_identifier",
        "compatible_with": get_compatible_tags("unique_identifier", TAG_COMPATIBILITY_PAIRS),
        "dataset_limit": 1
    },
    "model_feature": {
        "name": "model_feature",
        "compatible_with": get_compatible_tags("model_feature", TAG_COMPATIBILITY_PAIRS),
    },
    "row_timestamp": {
        "name": "row_timestamp",
        "compatible_with": get_compatible_tags("row_timestamp", TAG_COMPATIBILITY_PAIRS),
        "dataset_limit": 1
    },
    "weight": {
        "name": "weight",
        "compatible_with": get_compatible_tags("weight", TAG_COMPATIBILITY_PAIRS),
        "dataset_limit": 1
    }
}


def create_tag(name: str) -> Tag:
    """Factory function to create a Tag instance.
    
    Parameters
    ----------
    name: str
        The name of the tag to create.
    **kwargs
        Additional parameters to override default configuration.
        
    Returns
    -------
    Tag
        A configured Tag instance.
        
    Raises
    ------
    ValueError
        If the tag name is not found in TAG_CONFIGS.
    """
    if name not in TAG_CONFIGS:
        raise ValueError(f"Unknown tag name: {name}. Available tags: {list(TAG_CONFIGS.keys())}")
    
    config = TAG_CONFIGS[name].copy()
    return Tag(**config)
