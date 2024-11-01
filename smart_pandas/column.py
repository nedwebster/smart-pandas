from typing import Optional
from dataclasses import dataclass
import pandera as pa

from smart_pandas.label import LABEL_MAP
from smart_pandas.label_set import LabelSet


@dataclass
class Column:
    """Class to represent a column in a dataframe.

    Parameters
    ----------
    name: str
        The name of the column.
    schema: pa.Column
        The schema of the column.
    labels: LabelSet
        The labels associated with the column.
    description: Optional[str]
        A description of the column.

    """

    name: str
    schema: pa.Column
    labels: LabelSet
    description: Optional[str] = None

    def __post_init__(
        self,
    ):
        for label_name in LABEL_MAP.keys():
            setattr(self, label_name, any(label == label_name for label in self.labels))
