import sys
import inspect
from abc import ABC
from typing import List, Optional
from pydantic import BaseModel


class Tag(BaseModel, ABC):
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
    compatible_with: List[str] = []
    dataset_limit: Optional[int] = None

    def __eq__(self, other: str) -> bool:
        return self.name == other

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __hash__(self):
        return hash(repr(self))


class Target(Tag):
    name: str = "target"
    dataset_limit: int = 1


class RawFeature(Tag):
    name: str = "raw_feature"
    compatible_with: List[str] = ["model_feature", "row_timestamp"]


class DerivedFeature(Tag):
    name: str = "derived_feature"
    compatible_with: List[str] = ["model_feature"]


class Metadata(Tag):
    name: str = "metadata"
    compatible_with: List[str] = ["unique_identifier", "row_timesamp"]


class UniqueIdentifier(Tag):
    name: str = "unique_identifier"
    compatible_with: List[str] = ["metadata"]
    dataset_limit: int = 1


class ModelFeature(Tag):
    name: str = "model_feature"
    compatible_with: List[str] = ["raw_feature", "derived_feature"]


class RowTimestamp(Tag):
    name: str = "row_timestamp"
    comaptible_with: List[str] = ["raw_feature", "metadata"]
    dataset_limit: int = 1


class Weight(Tag):
    name: str = "weight"
    dataset_limit: int = 1


TAG_MAP = {
    obj().name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if name != "Tag" and Tag in obj.__mro__
} 