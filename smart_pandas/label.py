import sys
import inspect
from abc import ABC
from typing import List, Optional
from pydantic import BaseModel


class Label(BaseModel, ABC):
    """Class to represent a specific label assigned to a column.

    Parameters
    ----------
    name: str
        The name of the label.
    compatible_with: List[str]
        A list of labels that this label is compatible with.
    dataset_limit: Optional[int]
        The maximum number of columns that can be assigned this label in a dataset. If set to None, there is no limit.

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


class Target(Label):
    name: str = "target"
    dataset_limit: int = 1


class RawFeature(Label):
    name: str = "raw_feature"
    compatible_with: List[str] = ["model_feature", "row_timestamp"]


class DerivedFeature(Label):
    name: str = "derived_feature"
    compatible_with: List[str] = ["model_feature"]


class Metadata(Label):
    name: str = "metadata"
    compatible_with: List[str] = ["unique_identifier", "row_timesamp"]


class UniqueIdentifier(Label):
    name: str = "unique_identifier"
    compatible_with: List[str] = ["metadata"]
    dataset_limit: int = 1


class ModelFeature(Label):
    name: str = "model_feature"
    compatible_with: List[str] = ["raw_feature", "derived_feature"]


class RowTimestamp(Label):
    name: str = "row_timestamp"
    comaptible_with: List[str] = ["raw_feature", "metadata"]
    dataset_limit: int = 1


LABEL_MAP = {
    obj().name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
    if name != "Label" and Label in obj.__mro__
}
