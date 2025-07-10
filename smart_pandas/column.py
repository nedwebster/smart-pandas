from typing import Optional
from dataclasses import dataclass
import pandera as pa

from smart_pandas.tag import TAG_MAP
from smart_pandas.tag_set import TagSet


@dataclass
class Column:
    """Class to represent a column in a dataframe.

    Parameters
    ----------
    name: str
        The name of the column.
    schema: pa.Column
        The schema of the column.
    tags: TagSet
        The tags associated with the column.
    description: Optional[str]
        A description of the column.

    """

    name: str
    schema: pa.Column
    tags: TagSet
    description: Optional[str] = None

    def __post_init__(
        self,
    ):
        for tag_name in TAG_MAP.keys():
            setattr(self, tag_name, any(tag == tag_name for tag in self.tags))
