from typing import Any
from pydantic import BaseModel


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


# Configuration dictionary for predefined tags
TAG_CONFIGS: dict[str, dict[str, Any]] = {
    "target": {
        "name": "target",
        "compatible_with": [],
        "dataset_limit": 1
    },
    "raw_feature": {
        "name": "raw_feature",
        "compatible_with": ["model_feature", "row_timestamp"]
    },
    "derived_feature": {
        "name": "derived_feature",
        "compatible_with": ["model_feature"]
    },
    "metadata": {
        "name": "metadata",
        "compatible_with": ["unique_identifier", "row_timestamp"]
    },
    "unique_identifier": {
        "name": "unique_identifier",
        "compatible_with": ["metadata"],
        "dataset_limit": 1
    },
    "model_feature": {
        "name": "model_feature",
        "compatible_with": ["raw_feature", "derived_feature"]
    },
    "row_timestamp": {
        "name": "row_timestamp",
        "compatible_with": ["raw_feature", "metadata"],
        "dataset_limit": 1
    },
    "weight": {
        "name": "weight",
        "compatible_with": [],
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
